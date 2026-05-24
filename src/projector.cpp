#include "projector.hpp"
#include <cstring>
#include <cmath>

namespace rcm {

Projector::Projector(int input_dim, int rank) 
    : input_dim(input_dim), rank(rank) {
    
    // Alocação de matrizes U e V
    U = new float[input_dim * rank];
    V = new float[rank * input_dim];
    temp_buffer = new float[rank * rank];
    
    // Inicialização ortogonal (QR simplificado)
    std::memset(U, 0, input_dim * rank * sizeof(float));
    std::memset(V, 0, rank * input_dim * sizeof(float));
    
    // Inicialização aleatória com normalização
    for (int i = 0; i < input_dim * rank; i++) {
        U[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f / sqrtf(input_dim);
    }
    
    for (int i = 0; i < rank * input_dim; i++) {
        V[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f / sqrtf(rank);
    }
    
    orthogonalize_stiefel();
}

Projector::~Projector() {
    delete[] U;
    delete[] V;
    delete[] temp_buffer;
}

void Projector::project(float* input, float* output) {
    // Projeção LoRA: output = V * (U^T * input)
    // Passo 1: projected = U^T * input (rank dims)
    float* projected = temp_buffer;
    for (int r = 0; r < rank; r++) {
        projected[r] = 0.0f;
        for (int d = 0; d < input_dim; d++) {
            projected[r] += U[d * rank + r] * input[d];
        }
    }
    
    // Passo 2: output = V * projected (input_dim dims)
    for (int d = 0; d < input_dim; d++) {
        output[d] = 0.0f;
        for (int r = 0; r < rank; r++) {
            output[d] += V[r * input_dim + d] * projected[r];
        }
    }
}

void Projector::update_weights(const float* gradient, float learning_rate) {
    // Update Hebbiano simples com regularização
    for (int i = 0; i < input_dim * rank; i++) {
        U[i] += learning_rate * gradient[i % input_dim];
        V[i] += learning_rate * gradient[i % input_dim];
    }
    
    // Aplicar regularização de Stiefel periodicamente
    orthogonalize_stiefel();
}

void Projector::orthogonalize_stiefel() {
    // Gram-Schmidt simplificado para manter ortogonalidade de U
    for (int r = 0; r < rank; r++) {
        // Normalizar coluna r
        float norm = 0.0f;
        for (int d = 0; d < input_dim; d++) {
            norm += U[d * rank + r] * U[d * rank + r];
        }
        norm = sqrtf(norm);
        
        if (norm > 1e-6f) {
            for (int d = 0; d < input_dim; d++) {
                U[d * rank + r] /= norm;
            }
        }
        
        // Ortogonalizar contra colunas anteriores
        for (int prev_r = 0; prev_r < r; prev_r++) {
            float dot = 0.0f;
            for (int d = 0; d < input_dim; d++) {
                dot += U[d * rank + r] * U[d * rank + prev_r];
            }
            
            for (int d = 0; d < input_dim; d++) {
                U[d * rank + r] -= dot * U[d * rank + prev_r];
            }
        }
    }
}

float Projector::compute_orthogonality_error() {
    // Calcula ||U^T * U - I||_F
    float error = 0.0f;
    
    for (int r1 = 0; r1 < rank; r1++) {
        for (int r2 = 0; r2 < rank; r2++) {
            float dot = 0.0f;
            for (int d = 0; d < input_dim; d++) {
                dot += U[d * rank + r1] * U[d * rank + r2];
            }
            
            float target = (r1 == r2) ? 1.0f : 0.0f;
            float diff = dot - target;
            error += diff * diff;
        }
    }
    
    return sqrtf(error);
}

} // namespace rcm
