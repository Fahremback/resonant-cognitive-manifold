#pragma once
#include "common.hpp"
#include "ssd_storage.hpp"
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

namespace rcm {

class VRAMCache {
public:
    VRAMCache(size_t max_nodes = 8192, size_t max_edges = 65536);
    ~VRAMCache();

    // Constrói o subgrafo ativo na RAM com estados multidimensionais (STATE_DIM)
    bool load_subgraph(const std::vector<uint64_t>& seed_nodes, 
                       const std::vector<std::vector<float>>& seed_inputs,
                       const SSDStorage& storage,
                       bool pin_seeds = true,
                       uint32_t window_index = 0);

    // Recupera os valores dos estados mu multidimensionais do dispositivo para o host
    bool download_states(std::vector<float>& out_mu);

    // Recupera os pesos sinápticos multidimensionais atualizados na GPU
    bool download_weights_and_sync(SSDStorage& storage, uint32_t window_index = 0);

    // Ponteiros para memória de dispositivo (GPU)
    uint64_t* get_d_node_ids() { return d_node_ids_; }
    float* get_d_mu() { return d_mu_; }
    float* get_d_target_mu() { return d_target_mu_; }
    float* get_d_is_input() { return d_is_input_; }
    uint32_t* get_d_edge_offsets() { return d_edge_offsets_; }
    uint32_t* get_d_edge_targets() { return d_edge_targets_; }
    float* get_d_U() { return d_U_; }
    float* get_d_V() { return d_V_; }
    float* get_d_precisions() { return d_precisions_; }
    float* get_d_phase_couplings() { return d_phase_couplings_; }

    size_t get_active_nodes_count() const { return active_nodes_count_; }
    size_t get_active_edges_count() const { return active_edges_count_; }

    const std::unordered_map<uint64_t, uint32_t>& get_global_to_local() const { return global_to_local_; }
    const std::vector<uint64_t>& get_local_to_global() const { return local_to_global_; }

private:
    size_t max_nodes_;
    size_t max_edges_;

    size_t active_nodes_count_ = 0;
    size_t active_edges_count_ = 0;

    // Ponteiros na GPU (VRAM)
    uint64_t* d_node_ids_ = nullptr;
    float* d_mu_ = nullptr;
    float* d_target_mu_ = nullptr;
    float* d_is_input_ = nullptr;
    uint32_t* d_edge_offsets_ = nullptr;
    uint32_t* d_edge_targets_ = nullptr;
    float* d_U_ = nullptr;
    float* d_V_ = nullptr;
    float* d_precisions_ = nullptr;
    float* d_phase_couplings_ = nullptr;

    // Índices de mapeamento Host
    std::unordered_map<uint64_t, uint32_t> global_to_local_;
    std::vector<uint64_t> local_to_global_;

    // Estruturas de staging na RAM
    std::vector<uint64_t> h_node_ids_;
    std::vector<float> h_mu_;
    std::vector<float> h_target_mu_;
    std::vector<float> h_is_input_;
    std::vector<uint32_t> h_edge_offsets_;
    std::vector<uint32_t> h_edge_targets_;
    std::vector<float> h_U_;
    std::vector<float> h_V_;
    std::vector<float> h_precisions_;
    std::vector<float> h_phase_couplings_;
    std::vector<uint64_t> h_global_edge_indices_; // Índices globais correspondentes no SSD para sincronização

    void allocate_gpu_memory();
    void free_gpu_memory();
};

} // namespace rcm
