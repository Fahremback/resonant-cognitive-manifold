#pragma once
#include "common.hpp"
#include "vram_cache.hpp"
#include <vector>
#include <utility>

namespace rcm {

class ResonanceEngine {
public:
    ResonanceEngine();
    ~ResonanceEngine();

    // Executa a otimização de minimização de energia livre variacional (F) na GPU em 8D
    ResonanceStats run_resonance(VRAMCache& vram_cache, 
                                 float alpha = 0.02f, 
                                 uint32_t max_iterations = 200, 
                                 float tolerance = 1e-5f);

    // Executa a calibragem dos pesos sinápticos via Hebbian Learning preditivo local na GPU em 8D
    bool run_hebbian_learning(VRAMCache& vram_cache, 
                               float learning_rate = 0.01f, 
                               float weight_decay = 0.001f);

    // Detecta pares de conceitos altamente correlacionados usando a GPU para consolidar em chunks no Sleep
    std::vector<std::pair<uint64_t, uint64_t>> find_correlated_chunks(VRAMCache& vram_cache, 
                                                                       float threshold = 0.9f);

private:
    float* d_grad_ = nullptr;             // Buffer de gradientes temporário na GPU (tamanho nodes_count * STATE_DIM)
    float* d_free_energy_ = nullptr;      // Buffer de acumulação de energia livre na GPU
    size_t allocated_nodes_ = 0;

    void ensure_buffers(size_t nodes_count);
    void free_buffers();
};

} // namespace rcm
