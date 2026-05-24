#pragma once
#include "common.hpp"
#include <unordered_map>
#include <vector>
#include <cuda_runtime.h>

namespace rcm {

class VRAMCache {
public:
    VRAMCache(size_t max_size_bytes = VRAM_CACHE_SIZE);
    ~VRAMCache();
    
    // Carregamento Sob Demanda (Flash-Loading)
    bool load_block_if_missing(uint64_t block_id, float* source_data, size_t count);
    void prefetch_blocks(const std::vector<uint64_t>& block_ids);
    
    // Descarte de Blocos Não Ressonantes
    void evict_least_resonant();
    void clear_cache();
    
    // Acesso Direto a Dados em VRAM
    float* get_device_ptr_for_block(uint64_t block_id);
    bool is_block_cached(uint64_t block_id) const;
    
    // Estatísticas de Cache
    size_t get_cache_hit_count() const { return cache_hits; }
    size_t get_cache_miss_count() const { return cache_misses; }
    float get_hit_rate() const;
    size_t get_current_usage_bytes() const { return current_usage_bytes; }
    size_t get_max_usage_bytes() const { return max_size_bytes; }
    
    // Métricas de Performance
    void print_stats();
    
private:
    size_t max_size_bytes;
    size_t current_usage_bytes;
    size_t cache_hits;
    size_t cache_misses;
    
    // Mapeamento bloco -> ponteiro VRAM
    std::unordered_map<uint64_t, float*> cached_blocks;
    
    // Fila LRU para evição
    std::vector<uint64_t> lru_queue;
    
    // Ponteiros CUDA para dados em VRAM
    std::unordered_map<uint64_t, cudaStream_t> block_streams;
};

} // namespace rcm
