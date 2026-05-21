#include "resonance_engine.h"
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace rcm {

#define LEAKY_RELU(x) ((x) > 0.0f ? (x) : (x) * 0.01f)

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_val = (err); \
        if (err_val != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << cudaGetErrorString(err_val) \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(-1); \
        } \
    } while (0)

// Prior precision (decaimento elástico para a crença default)
__constant__ float d_gamma = 0.0f;


// 1. Kernel para zerar gradientes e acumuladores
__global__ void rcm_zero_grad_kernel(float* d_grad, float* d_free_energy, uint32_t nodes_count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nodes_count) {
        for (uint32_t d = 0; d < STATE_DIM; ++d) {
            d_grad[idx * STATE_DIM + d] = 0.0f;
        }
    }
    if (idx == 0) {
        *d_free_energy = 0.0f;
    }
}

// 2. Kernel de cálculo do gradiente e energia livre com tensores LoRA (U * V)
__global__ void rcm_compute_gradient_kernel(
    const float* d_mu,
    const float* d_target_mu,
    const uint32_t* d_edge_offsets,
    const uint32_t* d_edge_targets,
    const float* d_U,
    const float* d_V,
    const float* d_precisions,
    const float* d_phase_couplings,
    float* d_grad,
    float* d_free_energy,
    uint32_t nodes_count
) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nodes_count) return;

    uint32_t start_edge = d_edge_offsets[k];
    uint32_t end_edge = d_edge_offsets[k + 1];
    float local_energy = 0.0f;

    // Termo de Complexidade (D_KL) - Desvio em relação à crença default (priors)
    for (uint32_t d = 0; d < STATE_DIM; ++d) {
        float default_mu = d_target_mu[k * STATE_DIM + d];
        float mu_val = d_mu[k * STATE_DIM + d];
        float diff_prior = mu_val - default_mu;
        local_energy += 0.5f * d_gamma * diff_prior * diff_prior;
        // Acumula diretamente em d_grad (sem contendência pois k é exclusivo de cada thread)
        d_grad[k * STATE_DIM + d] += d_gamma * diff_prior;
    }

    // Iterar sobre as arestas de saída (k prediz os nós de destino i)
    for (uint32_t e = start_edge; e < end_edge; ++e) {
        uint32_t i = d_edge_targets[e];
        float pi_ki = d_precisions[e];
        float phase_coupling = d_phase_couplings[e];
        int shift = __float2int_rn(phase_coupling);

        // 1. Calcular vetor intermediário z em registradores locais: z = V * mu_k_shifted (loop-swap optimization)
        float z[RANK] = {0.0f};
        for (uint32_t j = 0; j < STATE_DIM; ++j) {
            int idx_src = (static_cast<int>(j) - shift) % STATE_DIM;
            if (idx_src < 0) idx_src += STATE_DIM;
            float mu_val = d_mu[k * STATE_DIM + idx_src];
            for (uint32_t r = 0; r < RANK; ++r) {
                z[r] += d_V[e * RANK * STATE_DIM + r * STATE_DIM + j] * mu_val;
            }
        }

        // 2. Calcular predição do target, erro, local_energy, target grad, e w_temp em um único loop
        float w_temp[RANK] = {0.0f};
        for (uint32_t d = 0; d < STATE_DIM; ++d) {
            float mu_hat = 0.0f;
            uint32_t u_row_offset = e * STATE_DIM * RANK + d * RANK;
            for (uint32_t r = 0; r < RANK; ++r) {
                mu_hat += d_U[u_row_offset + r] * z[r];
            }
            float mu_i_d = d_mu[i * STATE_DIM + d];
            float err = mu_i_d - mu_hat;

            local_energy += 0.5f * pi_ki * err * err;

            // Gradiente em relação ao destino: dF/dmu_i[d] += pi_ki * err
            float val_target = pi_ki * err;
            if (val_target != 0.0f) {
                atomicAdd(&d_grad[i * STATE_DIM + d], val_target);
            }

            for (uint32_t r = 0; r < RANK; ++r) {
                w_temp[r] += d_U[u_row_offset + r] * err;
            }
        }

        // 3. Calcular gradiente em relação à origem com loop invertido otimizado (16x menos escritas)
        for (uint32_t j = 0; j < STATE_DIM; ++j) {
            float sum_v_w = 0.0f;
            for (uint32_t r = 0; r < RANK; ++r) {
                uint32_t v_row_offset = e * RANK * STATE_DIM + r * STATE_DIM;
                sum_v_w += d_V[v_row_offset + j] * w_temp[r];
            }
            int idx_src = (static_cast<int>(j) - shift) % STATE_DIM;
            if (idx_src < 0) idx_src += STATE_DIM;
            
            float val_src = -pi_ki * sum_v_w;
            if (val_src != 0.0f) {
                atomicAdd(&d_grad[k * STATE_DIM + idx_src], val_src);
            }
        }
    }

    // Acumula energia livre total na VRAM
    if (local_energy > 0.0f) {
        atomicAdd(d_free_energy, local_energy);
    }
}

// 3. Kernel de atualização de estados mu em STATE_DIM dimensões com LeakyReLU e esparsidade
__global__ void rcm_update_states_kernel(
    float* d_mu,
    const float* d_target_mu,
    const float* d_is_input,
    const float* d_grad,
    float alpha,
    uint32_t nodes_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodes_count) return;

    for (uint32_t d = 0; d < STATE_DIM; ++d) {
        if (d_is_input[idx] > 0.5f && d_target_mu[idx * STATE_DIM + d] > 0.0f) {
            // Apenas trava dimensões de input que possuem excitação ativa
            d_mu[idx * STATE_DIM + d] = d_target_mu[idx * STATE_DIM + d];
        } else {
            float grad = d_grad[idx * STATE_DIM + d];
            
            // Estabilização numérica do gradiente (clipping)
            if (grad > 8.0f) grad = 8.0f;
            if (grad < -8.0f) grad = -8.0f;

            float val = d_mu[idx * STATE_DIM + d] - alpha * grad;
            float new_val = LEAKY_RELU(val);
            if (new_val < 0.001f) {
                new_val = 0.0f; // Barreira de esparsidade
            }
            d_mu[idx * STATE_DIM + d] = new_val;
        }
    }
}

// 4. Kernel de Aprendizado Hebbiano Preditivo em STATE_DIM dimensões com tensores LoRA (U * V)
__global__ void rcm_hebbian_learning_kernel(
    const float* d_mu,
    const uint32_t* d_edge_offsets,
    const uint32_t* d_edge_targets,
    float* d_U,
    float* d_V,
    const float* d_precisions,
    const float* d_phase_couplings,
    float learning_rate,
    float weight_decay,
    uint32_t nodes_count
) {
    uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nodes_count) return;

    uint32_t start_edge = d_edge_offsets[k];
    uint32_t end_edge = d_edge_offsets[k + 1];

    for (uint32_t e = start_edge; e < end_edge; ++e) {
        uint32_t i = d_edge_targets[e];
        float pi_ki = d_precisions[e];
        float phase_coupling = d_phase_couplings[e];
        int shift = __float2int_rn(phase_coupling);

        // 1. Calcular z = V * mu_k_shifted (loop-swap optimization)
        float z[RANK] = {0.0f};
        for (uint32_t j = 0; j < STATE_DIM; ++j) {
            int idx_src = (static_cast<int>(j) - shift) % STATE_DIM;
            if (idx_src < 0) idx_src += STATE_DIM;
            float mu_val = d_mu[k * STATE_DIM + idx_src];
            for (uint32_t r = 0; r < RANK; ++r) {
                z[r] += d_V[e * RANK * STATE_DIM + r * STATE_DIM + j] * mu_val;
            }
        }

        // 2 & 3. Calcular predição, erro, atualizar U e acumular w_temp
        float w_temp[RANK] = {0.0f};
        for (uint32_t d = 0; d < STATE_DIM; ++d) {
            float mu_hat = 0.0f;
            uint32_t u_row_offset = e * STATE_DIM * RANK + d * RANK;
            for (uint32_t r = 0; r < RANK; ++r) {
                mu_hat += d_U[u_row_offset + r] * z[r];
            }
            float mu_i_d = d_mu[i * STATE_DIM + d];
            float err_d = mu_i_d - mu_hat;

            for (uint32_t r = 0; r < RANK; ++r) {
                float U_val = d_U[u_row_offset + r];
                float delta_U = learning_rate * pi_ki * err_d * z[r];
                float new_U = U_val + delta_U - learning_rate * weight_decay * U_val;
                if (new_U > 4.0f) new_U = 4.0f;
                if (new_U < -4.0f) new_U = -4.0f;
                d_U[u_row_offset + r] = new_U;
                w_temp[r] += new_U * err_d;
            }
        }

        // 4. Atualizar V (loop-swap optimization)
        for (uint32_t j = 0; j < STATE_DIM; ++j) {
            int idx_src = (static_cast<int>(j) - shift) % STATE_DIM;
            if (idx_src < 0) idx_src += STATE_DIM;
            float mu_val = d_mu[k * STATE_DIM + idx_src];
            for (uint32_t r = 0; r < RANK; ++r) {
                uint32_t v_idx = e * RANK * STATE_DIM + r * STATE_DIM + j;
                float V_val = d_V[v_idx];
                float delta_V = learning_rate * pi_ki * w_temp[r] * mu_val;
                float new_V = V_val + delta_V - learning_rate * weight_decay * V_val;
                if (new_V > 4.0f) new_V = 4.0f;
                if (new_V < -4.0f) new_V = -4.0f;
                d_V[v_idx] = new_V;
            }
        }
    }
}

// 5. Kernel de cálculo da correlação para Sleep Chunking (Tiled Matrix Multiplication em Shared Memory)
__global__ void rcm_sleep_chunking_kernel(
    const float* d_mu,
    float* d_similarity_matrix,
    uint32_t nodes_count
) {
    // Dimensões do bloco para Tiling de Shared Memory (16x16)
    __shared__ float tile_x[16][128]; // Blocos de 16 threads, fatias de 128 dimensões de estado
    __shared__ float tile_y[16][128];

    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;
    uint32_t row = blockIdx.y * 16 + ty;
    uint32_t col = blockIdx.x * 16 + tx;

    float acc = 0.0f;
    float norm_x = 0.0f;
    float norm_y = 0.0f;

    // Loop externo dividindo STATE_DIM (1024) em fatias de tamanho 128
    for (uint32_t slice = 0; slice < STATE_DIM; slice += 128) {
        // Carregamento cooperativo de 128 floats de d_mu por thread no bloco para a Shared Memory
        // Cada bloco processa 16 nós (row e col) e carrega 128 floats por nó
        for (uint32_t d_offset = tx; d_offset < 128; d_offset += 16) {
            uint32_t d = slice + d_offset;
            if (row < nodes_count) {
                tile_y[ty][d_offset] = d_mu[row * STATE_DIM + d];
            } else {
                tile_y[ty][d_offset] = 0.0f;
            }
            if (col < nodes_count) {
                tile_x[tx][d_offset] = d_mu[col * STATE_DIM + d];
            } else {
                tile_x[tx][d_offset] = 0.0f;
            }
        }
        __syncthreads();

        // Acumulação local nos registradores a partir do Tile
        for (uint32_t d_offset = 0; d_offset < 128; ++d_offset) {
            float val_y = tile_y[ty][d_offset];
            float val_x = tile_x[tx][d_offset];
            acc += val_x * val_y;
            norm_x += val_x * val_x;
            norm_y += val_y * val_y;
        }
        __syncthreads();
    }

    if (row < nodes_count && col < nodes_count) {
        if (row == col) {
            d_similarity_matrix[row * nodes_count + col] = 1.0f;
        } else {
            float amp_x = sqrtf(norm_x);
            float amp_y = sqrtf(norm_y);
            if (amp_x > 0.0f && amp_y > 0.0f) {
                d_similarity_matrix[row * nodes_count + col] = acc / (amp_x * amp_y);
            } else {
                d_similarity_matrix[row * nodes_count + col] = 0.0f;
            }
        }
    }
}

ResonanceEngine::ResonanceEngine() {}

ResonanceEngine::~ResonanceEngine() {
    free_buffers();
}

void ResonanceEngine::ensure_buffers(size_t nodes_count) {
    if (allocated_nodes_ < nodes_count) {
        free_buffers();
        cudaMalloc(&d_grad_, nodes_count * STATE_DIM * sizeof(float));
        cudaMalloc(&d_free_energy_, sizeof(float));
        allocated_nodes_ = nodes_count;
    }
}

void ResonanceEngine::free_buffers() {
    if (d_grad_) { cudaFree(d_grad_); d_grad_ = nullptr; }
    if (d_free_energy_) { cudaFree(d_free_energy_); d_free_energy_ = nullptr; }
    allocated_nodes_ = 0;
}

ResonanceStats ResonanceEngine::run_resonance(VRAMCache& vram_cache, 
                                             float alpha, 
                                             uint32_t max_iterations, 
                                             float tolerance) {
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    size_t nodes_count = vram_cache.get_active_nodes_count();
    ensure_buffers(nodes_count);

    uint32_t threads_per_block = 256;
    uint32_t blocks = (static_cast<uint32_t>(nodes_count) + threads_per_block - 1) / threads_per_block;

    ResonanceStats stats = {0.0f, 0.0f, 0};
    float prev_energy = 999999.0f;

    std::cout << "[ResonanceEngine] Iniciando run_resonance com " << nodes_count << " nos...\n";
    for (uint32_t iter = 0; iter < max_iterations; ++iter) {
        rcm_zero_grad_kernel<<<blocks, threads_per_block>>>(d_grad_, d_free_energy_, static_cast<uint32_t>(nodes_count));
        
        rcm_compute_gradient_kernel<<<blocks, threads_per_block>>>(
            vram_cache.get_d_mu(),
            vram_cache.get_d_target_mu(),
            vram_cache.get_d_edge_offsets(),
            vram_cache.get_d_edge_targets(),
            vram_cache.get_d_U(),
            vram_cache.get_d_V(),
            vram_cache.get_d_precisions(),
            vram_cache.get_d_phase_couplings(),
            d_grad_,
            d_free_energy_,
            static_cast<uint32_t>(nodes_count)
        );

        rcm_update_states_kernel<<<blocks, threads_per_block>>>(
            vram_cache.get_d_mu(),
            vram_cache.get_d_target_mu(),
            vram_cache.get_d_is_input(),
            d_grad_,
            alpha,
            static_cast<uint32_t>(nodes_count)
        );
        
        CUDA_CHECK(cudaGetLastError());

        float h_free_energy = 0.0f;
        CUDA_CHECK(cudaMemcpy(&h_free_energy, d_free_energy_, sizeof(float), cudaMemcpyDeviceToHost));

        if (iter == 0) {
            stats.initial_energy = h_free_energy;
        }

        float energy_diff = std::abs(prev_energy - h_free_energy);
        if (iter > 5 && energy_diff < tolerance) {
            stats.final_energy = h_free_energy;
            stats.iterations_completed = iter + 1;
            break;
        }

        prev_energy = h_free_energy;
        stats.final_energy = h_free_energy;
        stats.iterations_completed = iter + 1;
    }
    std::cout << "[ResonanceEngine] Concluido run_resonance em " << stats.iterations_completed << " iteracoes. Energia: " << stats.final_energy << "\n";

    return stats;
}

bool ResonanceEngine::run_hebbian_learning(VRAMCache& vram_cache, 
                                           float learning_rate, 
                                           float weight_decay) {
    size_t nodes_count = vram_cache.get_active_nodes_count();
    uint32_t threads_per_block = 256;
    uint32_t blocks = (static_cast<uint32_t>(nodes_count) + threads_per_block - 1) / threads_per_block;

    rcm_hebbian_learning_kernel<<<blocks, threads_per_block>>>(
        vram_cache.get_d_mu(),
        vram_cache.get_d_edge_offsets(),
        vram_cache.get_d_edge_targets(),
        vram_cache.get_d_U(),
        vram_cache.get_d_V(),
        vram_cache.get_d_precisions(),
        vram_cache.get_d_phase_couplings(),
        learning_rate,
        weight_decay,
        static_cast<uint32_t>(nodes_count)
    );

    cudaError_t err = cudaDeviceSynchronize();
    return err == cudaSuccess;
}

std::vector<std::pair<uint64_t, uint64_t>> ResonanceEngine::find_correlated_chunks(VRAMCache& vram_cache, float threshold) {
    size_t nodes_count = vram_cache.get_active_nodes_count();
    if (nodes_count < 2) return {};

    float* d_similarity_matrix = nullptr;
    cudaError_t err = cudaMalloc(&d_similarity_matrix, nodes_count * nodes_count * sizeof(float));
    if (err != cudaSuccess) return {};

    dim3 threads_per_block(16, 16);
    dim3 blocks((static_cast<uint32_t>(nodes_count) + 15) / 16, (static_cast<uint32_t>(nodes_count) + 15) / 16);

    rcm_sleep_chunking_kernel<<<blocks, threads_per_block>>>(
        vram_cache.get_d_mu(),
        d_similarity_matrix,
        static_cast<uint32_t>(nodes_count)
    );

    std::vector<float> h_similarity_matrix(nodes_count * nodes_count);
    cudaMemcpy(h_similarity_matrix.data(), d_similarity_matrix, nodes_count * nodes_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_similarity_matrix);

    const auto& local_to_global = vram_cache.get_local_to_global();

    std::vector<std::pair<uint64_t, uint64_t>> correlated_pairs;
    for (size_t i = 0; i < nodes_count; ++i) {
        for (size_t j = i + 1; j < nodes_count; ++j) {
            float sim = h_similarity_matrix[i * nodes_count + j];
            if (sim >= threshold) {
                correlated_pairs.push_back({local_to_global[i], local_to_global[j]});
            }
        }
    }

    return correlated_pairs;
}

} // namespace rcm
