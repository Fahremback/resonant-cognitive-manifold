#pragma once
#include "common.hpp"
#include <vector>

namespace rcm {

class Projector {
public:
    Projector(int input_dim, int rank);
    ~Projector();
    
    // Projeção LoRA com regularização de Stiefel
    void project(float* input, float* output);
    void update_weights(const float* gradient, float learning_rate);
    
    // Regularização de Ortogonalidade (Stiefel Manifold)
    void orthogonalize_stiefel();
    float compute_orthogonality_error();
    
    // Acesso aos pesos U e V (baixo posto)
    float* get_U() { return U; }
    float* get_V() { return V; }
    
private:
    int input_dim;
    int rank;
    float* U; // Matriz input_dim x rank
    float* V; // Matriz rank x input_dim
    
    // Buffers temporários para cálculo de projeção
    float* temp_buffer;
};

} // namespace rcm
