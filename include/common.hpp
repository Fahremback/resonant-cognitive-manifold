#pragma once
#include <cmath>
#include <cstdint>

namespace rcm {

// Configurações Globais do RCM 4.0 Neuro-Flash
constexpr int STATE_DIM = 256;
constexpr int RANK = 48;
constexpr int HALF_DIM = STATE_DIM / 2;
constexpr float GAMMA_DEFAULT = 0.1f;
constexpr float STIEFEL_THRESHOLD = 3.5f;
constexpr float PHASE_COUPLING_ACTIVE = 1.0f;
constexpr float PHASE_COUPLING_SLEEP = 0.0f;

// Configurações de Memória Neuro-Flash
constexpr size_t MAX_VIRTUAL_PARAMS = 1000000000000ULL; // 1 Trilhão
constexpr size_t VRAM_CACHE_SIZE = 8 * 1024 * 1024 * 1024ULL; // 8GB
constexpr size_t SSD_BLOCK_SIZE = 4096;
constexpr float QUANTIZATION_BITS = 4.0f;

// Estruturas de Dados
struct ResonanceState {
    float mu[STATE_DIM];
    float sigma[STATE_DIM];
    float phase_coupling;
    uint64_t timestamp;
};

struct Edge {
    int src;
    int dst;
    float weight;
    float phase_offset;
};

struct CSRGraph {
    int* row_offsets;
    int* col_indices;
    float* values;
    int num_nodes;
    int num_edges;
};

// Funções Utilitárias
inline float leaky_relu(float x, float alpha = 0.01f) {
    return x > 0 ? x : alpha * x;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline float clamp(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, x));
}

} // namespace rcm
