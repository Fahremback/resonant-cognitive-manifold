#include "ssd_storage.hpp"
#include <iostream>
#include <fstream>

namespace rcm {

SSDStorage::SSDStorage() {}

SSDStorage::~SSDStorage() {
    cleanup();
}

void SSDStorage::close_db() {
    cleanup();
}

void SSDStorage::cleanup() {
    if (nodes_ptr_) { UnmapViewOfFile(nodes_ptr_); nodes_ptr_ = nullptr; }
    if (edges_ptr_) { UnmapViewOfFile(edges_ptr_); edges_ptr_ = nullptr; }
    if (nodes_mapping_) { CloseHandle(nodes_mapping_); nodes_mapping_ = NULL; }
    if (edges_mapping_) { CloseHandle(edges_mapping_); edges_mapping_ = NULL; }
    if (nodes_file_ != INVALID_HANDLE_VALUE) { CloseHandle(nodes_file_); nodes_file_ = INVALID_HANDLE_VALUE; }
    if (edges_file_ != INVALID_HANDLE_VALUE) { CloseHandle(edges_file_); edges_file_ = INVALID_HANDLE_VALUE; }
    node_id_to_index_.clear();
    nodes_size_ = 0;
    edges_size_ = 0;
    node_count_ = 0;
    edge_count_ = 0;
}

bool SSDStorage::open_db(const std::string& nodes_path, const std::string& edges_path) {
    cleanup();

    // 1. Abrir os arquivos de forma exclusiva/compartilhada para leitura
    nodes_file_ = CreateFileA(nodes_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (nodes_file_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[SSDStorage] Falha ao abrir arquivo de nos: " << nodes_path << " (Erro Win32: " << GetLastError() << ")\n";
        return false;
    }

    edges_file_ = CreateFileA(edges_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (edges_file_ == INVALID_HANDLE_VALUE) {
        std::cerr << "[SSDStorage] Falha ao abrir arquivo de arestas: " << edges_path << " (Erro Win32: " << GetLastError() << ")\n";
        cleanup();
        return false;
    }

    // 2. Obter tamanhos dos arquivos para determinar limites
    LARGE_INTEGER size;
    if (!GetFileSizeEx(nodes_file_, &size)) {
        std::cerr << "[SSDStorage] Falha ao ler tamanho de nos.\n";
        cleanup();
        return false;
    }
    nodes_size_ = size.QuadPart;
    node_count_ = nodes_size_ / sizeof(DiskNode);

    if (!GetFileSizeEx(edges_file_, &size)) {
        std::cerr << "[SSDStorage] Falha ao ler tamanho de arestas.\n";
        cleanup();
        return false;
    }
    edges_size_ = size.QuadPart;
    edge_count_ = edges_size_ / sizeof(DiskEdge);

    // 3. Mapear arquivos para memória virtual
    if (nodes_size_ > 0) {
        nodes_mapping_ = CreateFileMappingA(nodes_file_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!nodes_mapping_) {
            std::cerr << "[SSDStorage] Falha ao criar mapeamento para nos (Erro Win32: " << GetLastError() << ")\n";
            cleanup();
            return false;
        }
        nodes_ptr_ = static_cast<DiskNode*>(MapViewOfFile(nodes_mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!nodes_ptr_) {
            std::cerr << "[SSDStorage] Falha ao visualizar mapa de nos (Erro Win32: " << GetLastError() << ")\n";
            cleanup();
            return false;
        }
    }

    if (edges_size_ > 0) {
        edges_mapping_ = CreateFileMappingA(edges_file_, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!edges_mapping_) {
            std::cerr << "[SSDStorage] Falha ao criar mapeamento para arestas (Erro Win32: " << GetLastError() << ")\n";
            cleanup();
            return false;
        }
        edges_ptr_ = static_cast<DiskEdge*>(MapViewOfFile(edges_mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!edges_ptr_) {
            std::cerr << "[SSDStorage] Falha ao visualizar mapa de arestas (Erro Win32: " << GetLastError() << ")\n";
            cleanup();
            return false;
        }
    }

    // 4. Indexar IDs de nós para busca instantânea em RAM
    build_index();
    return true;
}

void SSDStorage::build_index() {
    node_id_to_index_.clear();
    node_id_to_index_.reserve(node_count_);
    for (size_t i = 0; i < node_count_; ++i) {
        node_id_to_index_[nodes_ptr_[i].id] = i;
    }
}

bool SSDStorage::get_node(uint64_t node_id, DiskNode& out_node) const {
    auto it = node_id_to_index_.find(node_id);
    if (it == node_id_to_index_.end()) {
        return false;
    }
    out_node = nodes_ptr_[it->second];
    return true;
}

bool SSDStorage::get_edges(const DiskNode& node, std::vector<DiskEdge>& out_edges) const {
    if (node.edge_count == 0) {
        out_edges.clear();
        return true;
    }

    uint64_t start_idx = node.edge_offset / sizeof(DiskEdge);
    if (start_idx + node.edge_count > edge_count_) {
        std::cerr << "[SSDStorage] Erro: Tentativa de leitura de arestas fora do limite do arquivo.\n";
        return false;
    }

    out_edges.resize(node.edge_count);
    std::memcpy(out_edges.data(), &edges_ptr_[start_idx], node.edge_count * sizeof(DiskEdge));
    return true;
}

bool SSDStorage::write_graph(const std::vector<DiskNode>& nodes, const std::vector<DiskEdge>& edges) {
    // Para garantir consistência ao reescrever, fechamos o mapeamento se estiver aberto
    cleanup();

    // Caminhos temporários locais. Faremos a criação via stream padrão para escrita sequencial robusta.
    // Usaremos caminhos padronizados "nodes.bin" e "edges.bin"
    std::ofstream n_file("nodes.bin", std::ios::binary | std::ios::trunc);
    if (!n_file.is_open()) {
        std::cerr << "[SSDStorage] Erro ao criar nodes.bin para escrita.\n";
        return false;
    }
    n_file.write(reinterpret_cast<const char*>(nodes.data()), nodes.size() * sizeof(DiskNode));
    n_file.close();

    std::ofstream e_file("edges.bin", std::ios::binary | std::ios::trunc);
    if (!e_file.is_open()) {
        std::cerr << "[SSDStorage] Erro ao criar edges.bin para escrita.\n";
        return false;
    }
    e_file.write(reinterpret_cast<const char*>(edges.data()), edges.size() * sizeof(DiskEdge));
    e_file.close();

    // Reabre e mapeia os novos arquivos gerados
    return open_db("nodes.bin", "edges.bin");
}

} // namespace rcm
