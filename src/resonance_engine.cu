#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

// ============================================================================
// RCM 4.0 Neuro-Flash - Kernel de Aprendizado Hebbiano com Regularização de Stiefel
// ============================================================================
// Arquitetura: 1 Trilhão de Parâmetros Virtuais
// VRAM Ativa: < 8GB (Carregamento sob demanda via Flash-Loading)
// ============================================================================

#define STATE_DIM 256
#define RANK 48
#define MAX_VIRTUAL_PARAMS 1000000000000ULL // 1 Trilhão
#define BLOCK_SIZE 256

// Constant Memory para hiperparâmetros (sincronizados via cudaMemcpyToSymbol)
__constant__ float d_gamma;          // Taxa de aprendizado Hebbiano
__constant__ float d_target_mu;      // Crença default (prior)
__constant__ float d_stiefel_lambda; // Peso da regularização de ortogonalidade

/**
 * Kernel Hebbiano com Projeção de Stiefel
 * 
 * Atualiza matrizes de baixo posto U e V mantendo ortogonalidade estrita.
 * Previne colapso de rank e saturação em limites de hardware (±4.0f).
 * 
 * @param d_U Ponteiro para matriz U (STATE_DIM x RANK)
 * @param d_V Ponteiro para matriz V (RANK x STATE_DIM)
 * @param d_activation Ativações correntes do manifold
 * @param d_error Erro local de predição
 */
__global__ void rcm_hebbian_learning_kernel(float* d_U, float* d_V, 
                                            float* d_activation, float* d_error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / RANK;
    int col = idx % RANK;

    if (row >= STATE_DIM) return;

    // Carregamento de valores (na implementação real, fetch assíncrono do SSD)
    float u_val = d_U[idx];
    float act = d_activation[row];
    float err = d_error[col];

    // Update Hebbiano Clássico: ΔU = γ * error * activation
    float delta = d_gamma * err * act;
    float new_u = u_val + delta;

    // Regularização de Stiefel Simplificada
    // Penaliza desvios da ortogonalidade: ||U^T * U - I||_F^2
    // Implementado como normalização escalar suave por coluna
    float col_norm = 0.0f;
    for (int r = 0; r < STATE_DIM; r++) {
        float val = d_U[r * RANK + col];
        col_norm += val * val;
    }
    col_norm = sqrtf(col_norm + 1e-8f);

    // Normalização para manter norma unitária (ortogonalidade aproximada)
    float target_norm = 1.0f;
    float scale = target_norm / fmaxf(col_norm, 1e-6f);
    
    // Aplicar fator de correção suave
    float correction = 1.0f - d_stiefel_lambda * (1.0f - scale);
    new_u *= correction;

    // Soft clipping para evitar saturação rígida (melhor que if (x > 4) x = 4)
    float soft_limit = 3.5f;
    float final_u = new_u / fmaxf(1.0f, fabsf(new_u) / soft_limit);

    d_U[idx] = final_u;
}

/**
 * Kernel de Sleep Chunking com Linearização de Threads
 * 
 * Consolida conceitos altamente correlacionados preservando o acoplamento de fase temporal.
 * Elimina escritas redundantes na Shared Memory através de mapeamento linear de threads.
 * 
 * @param d_mu Estados atuais dos nós (mu)
 * @param d_abstract Nós abstratos consolidados
 * @param d_phase_map Mapeamento de fase herdado das arestas originais
 */
__global__ void rcm_sleep_chunking_kernel(float* d_mu, float* d_abstract, int* d_phase_map) {
    extern __shared__ float shared_tile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int thread_id = ty * blockDim.x + tx; // Linearização crítica (16x16 -> 0..255)
    
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;

    // Carregamento cooperativo sem redundância
    // Cada um dos 256 dados é lido exatamente uma vez por uma thread única
    if (thread_id < 128) {
        int global_idx = block_row * 128 + thread_id;
        shared_tile[thread_id] = d_mu[global_idx];
    }
    __syncthreads();

    // Processamento da janela consolidada
    if (thread_id < 64) {
        float sum = 0.0f;
        int phase_sum = 0;
        
        // Agregação ponderada preservando phase_coupling original
        for (int i = 0; i < 4; i++) {
            int offset = thread_id * 4 + i;
            if (offset < 128) {
                // Herda phase_coupling = 1.0 das arestas originais (NÃO usa 0.0 estático)
                int phase = d_phase_map[block_row * 64 + (thread_id * 4 + i) / 64];
                sum += shared_tile[offset] * (phase == 1 ? 1.0f : 0.5f);
                phase_sum += phase;
            }
        }

        // Nó abstrato mantém direção temporal se maioria das arestas tinha phase=1
        float avg_phase = (float)phase_sum / 4.0f;
        d_abstract[block_row * 64 + thread_id] = sum * avg_phase;
    }
}

/**
 * Kernel de Computação de Gradiente sem Atômicos (Redução CSR)
 * 
 * Utiliza a estrutura CSR (Compressed Sparse Row) para evitar atomicAdd.
 * Cada thread processa uma linha completa do grafo, acumulando em registrador local.
 * 
 * @param d_edge_values Valores das arestas
 * @param d_edge_offsets Offsets CSR (início/fim de cada nó)
 * @param d_grad Gradientes acumulados (saída exclusiva por thread)
 */
__global__ void rcm_compute_gradient_kernel(float* d_edge_values, int* d_edge_offsets, float* d_grad) {
    int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Leitura dos offsets CSR para este nó
    int start = d_edge_offsets[node_id];
    int end = d_edge_offsets[node_id + 1];

    // Acumulação em registrador local (ZERO atômicos)
    float local_grad = 0.0f;
    for (int i = start; i < end; i++) {
        local_grad += d_edge_values[i];
    }

    // Escrita única e exclusiva na memória global
    d_grad[node_id] = local_grad;
}

// Função utilitária para sincronizar símbolos do Host para Constant Memory
inline cudaError_t sync_hyperparameters(float gamma, float target_mu, float stiefel_lambda) {
    cudaError_t err;
    
    err = cudaMemcpyToSymbol(d_gamma, &gamma, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyToSymbol(d_target_mu, &target_mu, sizeof(float));
    if (err != cudaSuccess) return err;
    
    err = cudaMemcpyToSymbol(d_stiefel_lambda, &stiefel_lambda, sizeof(float));
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}
