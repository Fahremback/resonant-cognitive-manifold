#pragma once
#include "common.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace rcm {

class SSDStorage {
public:
    SSDStorage(const std::string& base_path);
    ~SSDStorage();
    
    // Inicialização do armazenamento de pesos virtuais
    void initialize_virtual_params(size_t total_params);
    
    // Leitura Assíncrona de Blocos (Simulação GPUDirect)
    void read_block_async(uint64_t block_id, float* destination, size_t count);
    void wait_read_complete();
    
    // Escrita de Pesos Atualizados (Checkpoints)
    void write_block(uint64_t block_id, const float* source, size_t count);
    
    // Mapeamento de Índice de Ressonância
    uint64_t get_block_for_concept(int concept_id);
    void update_resonance_index(int concept_id, uint64_t block_id);
    
    // Estatísticas
    size_t get_total_blocks() const { return total_blocks; }
    size_t get_bytes_read() const { return bytes_read; }
    size_t get_bytes_written() const { return bytes_written; }
    
private:
    std::string base_path;
    size_t total_params;
    size_t total_blocks;
    size_t bytes_read;
    size_t bytes_written;
    
    // Arquivo mapeado em memória para alta performance
    int fd;
    void* mapped_ptr;
    size_t mapped_size;
    
    // Índice de ressonância (conceito -> bloco SSD)
    std::unordered_map<int, uint64_t> resonance_index;
};

} // namespace rcm
