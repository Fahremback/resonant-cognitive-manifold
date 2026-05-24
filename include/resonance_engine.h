#pragma once
#include "common.hpp"
#include <cuda_runtime.h>

namespace rcm {

class ResonanceEngine {
public:
    ResonanceEngine(int num_nodes, int state_dim);
    ~ResonanceEngine();
    
    // Inicialização e Configuração
    void initialize_weights_orthogonal();
    void set_gamma(float gamma);
    
    // Kernels CUDA Principais
    void run_hebbian_learning(const float* activations, const float* errors, int batch_size);
    void compute_gradients_csr(const CSRGraph& graph, float* gradients);
    void sleep_chunking(int chunk_id, float phase_coupling_target);
    
    // Métricas e Telemetria
    float compute_stiefel_error();
    int get_effective_rank();
    void print_memory_stats();
    
    // Acesso a dados na GPU
    float* get_device_mu() { return d_mu; }
    float* get_device_U() { return d_U; }
    float* get_device_V() { return d_V; }
    
private:
    int num_nodes;
    int state_dim;
    float gamma;
    
    // Ponteiros para memória na GPU
    float* d_mu;
    float* d_sigma;
    float* d_U;
    float* d_V;
    float* d_grad;
    
    // Constantes na GPU
    static __constant__ float d_gamma_const;
    
    // Buffers temporários
    float* h_temp_buffer;
};

} // namespace rcm
