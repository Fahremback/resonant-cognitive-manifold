#include "ssd_storage.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <iostream>

#ifndef _WIN32
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#else
#define MAP_FAILED ((void*)-1)
#endif

namespace rcm {

SSDStorage::SSDStorage(const std::string& base_path) 
    : base_path(base_path), total_params(0), total_blocks(0), 
      bytes_read(0), bytes_written(0), fd(-1), mapped_ptr(nullptr), mapped_size(0) {
}

SSDStorage::~SSDStorage() {
#ifndef _WIN32
    if (mapped_ptr && mapped_ptr != MAP_FAILED) {
        munmap(mapped_ptr, mapped_size);
    }
    if (fd >= 0) {
        close(fd);
    }
#endif
}

void SSDStorage::initialize_virtual_params(size_t params) {
    total_params = params;
    size_t bytes_needed = params * sizeof(float);
    total_blocks = (bytes_needed + SSD_BLOCK_SIZE - 1) / SSD_BLOCK_SIZE;
    
    std::cout << "Inicializando " << total_params << " parâmetros virtuais (" 
              << (bytes_needed / (1024.0*1024.0*1024.0)) << "GB) em " 
              << total_blocks << " blocos SSD" << std::endl;
    
    // Em produção: criar arquivo mapeado em NVMe
    // Aqui: simulação em memória
    mapped_size = std::min(bytes_needed, size_t(1024 * 1024 * 1024)); // 1GB max simulação
}

void SSDStorage::read_block_async(uint64_t block_id, float* destination, size_t count) {
    if (block_id >= total_blocks) {
        std::cerr << "Erro: block_id " << block_id << " fora dos limites" << std::endl;
        return;
    }
    
    // Simulação de leitura assíncrona
    bytes_read += count * sizeof(float);
    
    // Em produção: usar CUDA GPUDirect Storage
    std::memset(destination, 0, count * sizeof(float));
}

void SSDStorage::wait_read_complete() {
    // Simulação: operação síncrona
}

void SSDStorage::write_block(uint64_t block_id, const float* source, size_t count) {
    if (block_id >= total_blocks) return;
    bytes_written += count * sizeof(float);
}

uint64_t SSDStorage::get_block_for_concept(int concept_id) {
    auto it = resonance_index.find(concept_id);
    if (it != resonance_index.end()) {
        return it->second;
    }
    // Hash simples para mapear conceito -> bloco
    return (uint64_t)(abs(concept_id) * 7919) % total_blocks;
}

void SSDStorage::update_resonance_index(int concept_id, uint64_t block_id) {
    resonance_index[concept_id] = block_id;
}

} // namespace rcm
