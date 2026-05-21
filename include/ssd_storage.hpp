#pragma once
#include "common.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <windows.h>

namespace rcm {

class SSDStorage {
public:
    SSDStorage();
    ~SSDStorage();

    // Abre e mapeia os arquivos binários do hipergrafo (nós e arestas)
    bool open_db(const std::string& nodes_path, const std::string& edges_path);
    
    // Fecha as conexões e libera mapeamentos de memória
    void close_db();

    // Busca um nó individual no hipergrafo
    bool get_node(uint64_t node_id, DiskNode& out_node) const;

    // Busca as arestas associadas a um nó específico
    bool get_edges(const DiskNode& node, std::vector<DiskEdge>& out_edges) const;

    // Utilitário para popular ou reescrever o grafo binário no SSD
    bool write_graph(const std::vector<DiskNode>& nodes, const std::vector<DiskEdge>& edges);

    // Estatísticas e ponteiros de baixo nível para aceleração
    size_t get_node_count() const { return node_count_; }
    size_t get_edge_count() const { return edge_count_; }
    
    const DiskNode* get_nodes_ptr() const { return nodes_ptr_; }
    const DiskEdge* get_edges_ptr() const { return edges_ptr_; }

private:
    HANDLE nodes_file_ = INVALID_HANDLE_VALUE;
    HANDLE edges_file_ = INVALID_HANDLE_VALUE;
    HANDLE nodes_mapping_ = NULL;
    HANDLE edges_mapping_ = NULL;

    DiskNode* nodes_ptr_ = nullptr;
    DiskEdge* edges_ptr_ = nullptr;

    size_t nodes_size_ = 0;
    size_t edges_size_ = 0;
    size_t node_count_ = 0;
    size_t edge_count_ = 0;

    // Índice rápido na RAM para mapear ID -> Posição no array mapeado
    std::unordered_map<uint64_t, size_t> node_id_to_index_;

    void build_index();
    void cleanup();
};

} // namespace rcm
