#include "vram_cache.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

namespace rcm {

VRAMCache::VRAMCache(size_t max_bytes) 
    : max_size_bytes(max_bytes), current_usage_bytes(0), 
      cache_hits(0), cache_misses(0) {
}

VRAMCache::~VRAMCache() {
    clear_cache();
}

bool VRAMCache::load_block_if_missing(uint64_t block_id, float* source_data, size_t count) {
    if (is_block_cached(block_id)) {
        cache_hits++;
        return true;
    }
    
    cache_misses++;
    
    // Verificar se há espaço
    size_t block_bytes = count * sizeof(float);
    if (current_usage_bytes + block_bytes > max_size_bytes) {
        evict_least_resonant();
    }
    
    // Alocar na VRAM
    float* device_ptr;
    cudaMalloc(&device_ptr, block_bytes);
    cudaMemcpy(device_ptr, source_data, block_bytes, cudaMemcpyHostToDevice);
    
    cached_blocks[block_id] = device_ptr;
    lru_queue.push_back(block_id);
    current_usage_bytes += block_bytes;
    
    return false;
}

void VRAMCache::prefetch_blocks(const std::vector<uint64_t>& block_ids) {
    // Em produção: usar streams CUDA assíncronos
    for (auto bid : block_ids) {
        if (!is_block_cached(bid)) {
            // Simulação de prefetch
        }
    }
}

void VRAMCache::evict_least_resonant() {
    if (lru_queue.empty()) return;
    
    // Evitar o bloco menos recentemente usado
    uint64_t lru_block = lru_queue.front();
    lru_queue.erase(lru_queue.begin());
    
    auto it = cached_blocks.find(lru_block);
    if (it != cached_blocks.end()) {
        cudaFree(it->second);
        current_usage_bytes -= 256 * sizeof(float); // Tamanho estimado
        cached_blocks.erase(it);
    }
}

void VRAMCache::clear_cache() {
    for (auto& pair : cached_blocks) {
        cudaFree(pair.second);
    }
    cached_blocks.clear();
    lru_queue.clear();
    current_usage_bytes = 0;
}

float* VRAMCache::get_device_ptr_for_block(uint64_t block_id) {
    auto it = cached_blocks.find(block_id);
    if (it != cached_blocks.end()) {
        return it->second;
    }
    return nullptr;
}

bool VRAMCache::is_block_cached(uint64_t block_id) const {
    return cached_blocks.find(block_id) != cached_blocks.end();
}

float VRAMCache::get_hit_rate() const {
    size_t total = cache_hits + cache_misses;
    if (total == 0) return 0.0f;
    return (float)cache_hits / (float)total;
}

void VRAMCache::print_stats() {
    std::cout << "=== VRAM Cache Stats ===" << std::endl;
    std::cout << "Hits: " << cache_hits << std::endl;
    std::cout << "Misses: " << cache_misses << std::endl;
    std::cout << "Hit Rate: " << (get_hit_rate() * 100) << "%" << std::endl;
    std::cout << "Usage: " << (current_usage_bytes / (1024.0f * 1024.0f)) << "MB / " 
              << (max_size_bytes / (1024.0f * 1024.0f)) << "MB" << std::endl;
}

} // namespace rcm
