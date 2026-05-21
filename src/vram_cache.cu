#include "vram_cache.hpp"
#include <iostream>
#include <queue>
#include <unordered_set>

namespace rcm {

VRAMCache::VRAMCache(size_t max_nodes, size_t max_edges)
    : max_nodes_(max_nodes), max_edges_(max_edges) {
    allocate_gpu_memory();
}

VRAMCache::~VRAMCache() {
    free_gpu_memory();
}

void VRAMCache::allocate_gpu_memory() {
    cudaError_t err;

    err = cudaMalloc(&d_node_ids_, max_nodes_ * sizeof(uint64_t));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_node_ids_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_mu_, max_nodes_ * STATE_DIM * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_mu_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_target_mu_, max_nodes_ * STATE_DIM * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_target_mu_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_is_input_, max_nodes_ * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_is_input_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_edge_offsets_, (max_nodes_ + 1) * sizeof(uint32_t));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_edge_offsets_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_edge_targets_, max_edges_ * sizeof(uint32_t));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_edge_targets_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_U_, max_edges_ * STATE_DIM * RANK * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_U_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_V_, max_edges_ * RANK * STATE_DIM * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_V_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_precisions_, max_edges_ * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_precisions_: " << cudaGetErrorString(err) << "\n";

    err = cudaMalloc(&d_phase_couplings_, max_edges_ * sizeof(float));
    if (err != cudaSuccess) std::cerr << "[VRAMCache] Erro cudaMalloc d_phase_couplings_: " << cudaGetErrorString(err) << "\n";
}

void VRAMCache::free_gpu_memory() {
    if (d_node_ids_) cudaFree(d_node_ids_);
    if (d_mu_) cudaFree(d_mu_);
    if (d_target_mu_) cudaFree(d_target_mu_);
    if (d_is_input_) cudaFree(d_is_input_);
    if (d_edge_offsets_) cudaFree(d_edge_offsets_);
    if (d_edge_targets_) cudaFree(d_edge_targets_);
    if (d_U_) cudaFree(d_U_);
    if (d_V_) cudaFree(d_V_);
    if (d_precisions_) cudaFree(d_precisions_);
    if (d_phase_couplings_) cudaFree(d_phase_couplings_);
}

bool VRAMCache::load_subgraph(const std::vector<uint64_t>& seed_nodes, 
                               const std::vector<std::vector<float>>& seed_inputs,
                               const SSDStorage& storage,
                               bool pin_seeds,
                               uint32_t window_index) {
    if (seed_nodes.empty()) return false;

    global_to_local_.clear();
    local_to_global_.clear();

    // 1. Expansão de vizinhança (BFS de profundidade 3)
    std::unordered_set<uint64_t> active_set;
    std::queue<std::pair<uint64_t, int>> q;
    for (uint64_t seed : seed_nodes) {
        active_set.insert(seed);
        q.push({seed, 0});
    }

    while (!q.empty()) {
        auto item = q.front();
        q.pop();
        uint64_t curr_id = item.first;
        int depth = item.second;

        if (depth < 3) {
            DiskNode node;
            if (storage.get_node(curr_id, node)) {
                std::vector<DiskEdge> edges;
                if (storage.get_edges(node, edges)) {
                    for (const auto& edge : edges) {
                        if (active_set.find(edge.target_id) == active_set.end()) {
                            active_set.insert(edge.target_id);
                            q.push({edge.target_id, depth + 1});
                        }
                    }
                }
            }
        }
    }

    active_nodes_count_ = active_set.size();
    if (active_nodes_count_ > max_nodes_) {
        std::cerr << "[VRAMCache] Subgrafo ativo excede capacidade maxima de nos (" 
                  << active_nodes_count_ << " > " << max_nodes_ << "). Truncando.\n";
        active_nodes_count_ = max_nodes_;
    }

    // Mapeamento bidirecional ID local <-> global
    local_to_global_.resize(active_nodes_count_);
    size_t index = 0;
    
    // Prioriza os seed nodes
    for (uint64_t seed : seed_nodes) {
        if (active_set.count(seed)) {
            global_to_local_[seed] = static_cast<uint32_t>(index);
            local_to_global_[index] = seed;
            active_set.erase(seed);
            index++;
            if (index >= active_nodes_count_) break;
        }
    }

    // Insere vizinhos ativados
    for (uint64_t node_id : active_set) {
        if (index >= active_nodes_count_) break;
        global_to_local_[node_id] = static_cast<uint32_t>(index);
        local_to_global_[index] = node_id;
        index++;
    }

    // 2. Preparação de staging buffers na CPU
    h_node_ids_.resize(active_nodes_count_);
    h_mu_.resize(active_nodes_count_ * STATE_DIM, 0.0f);
    h_target_mu_.resize(active_nodes_count_ * STATE_DIM, 0.0f);
    h_is_input_.resize(active_nodes_count_, 0.0f);
    h_edge_offsets_.resize(active_nodes_count_ + 1, 0);
    h_edge_targets_.clear();
    h_U_.clear();
    h_V_.clear();
    h_precisions_.clear();
    h_phase_couplings_.clear();
    h_global_edge_indices_.clear();

    // Mapeia inputs dos seed nodes
    std::unordered_map<uint64_t, std::vector<float>> seed_to_val;
    for (size_t i = 0; i < seed_nodes.size(); ++i) {
        seed_to_val[seed_nodes[i]] = seed_inputs[i];
    }

    uint32_t current_edge_offset = 0;
    h_edge_offsets_[0] = 0;

    for (size_t i = 0; i < active_nodes_count_; ++i) {
        uint64_t g_id = local_to_global_[i];
        h_node_ids_[i] = g_id;

        DiskNode disk_node;
        if (storage.get_node(g_id, disk_node)) {
            // Inicializa mu e target_mu com STATE_DIM
            if (seed_to_val.count(g_id)) {
                const auto& input_vec = seed_to_val[g_id];
                for (uint32_t d = 0; d < STATE_DIM; ++d) {
                    float val = d < input_vec.size() ? input_vec[d] : 0.0f;
                    h_mu_[i * STATE_DIM + d] = val;
                    h_target_mu_[i * STATE_DIM + d] = val;
                }
                h_is_input_[i] = pin_seeds ? 1.0f : 0.0f; // Fixa valor do input ou deixa flutuar
            } else {
                for (uint32_t d = 0; d < STATE_DIM; ++d) {
                    h_mu_[i * STATE_DIM + d] = disk_node.default_mu[d];
                    h_target_mu_[i * STATE_DIM + d] = disk_node.default_mu[d];
                }
                h_is_input_[i] = 0.0f; // Flutua livremente
            }

            // Lê arestas associadas
            std::vector<DiskEdge> edges;
            if (storage.get_edges(disk_node, edges)) {
                uint64_t global_start_idx = disk_node.edge_offset / sizeof(DiskEdge);
                for (size_t e_idx = 0; e_idx < edges.size(); ++e_idx) {
                    const auto& edge = edges[e_idx];
                    auto it = global_to_local_.find(edge.target_id);
                    if (it != global_to_local_.end()) {
                        h_edge_targets_.push_back(it->second);
                        
                        // Rotaciona circularmente os pesos LoRA com base no offset temporal da janela
                        uint32_t W_offset = (window_index * (STATE_DIM / 2)) % STATE_DIM;
                        for (uint32_t d = 0; d < STATE_DIM; ++d) {
                            uint32_t global_d = (d + W_offset) % STATE_DIM;
                            for (uint32_t r = 0; r < RANK; ++r) {
                                h_U_.push_back(edge.U[global_d * RANK + r]);
                            }
                        }
                        for (uint32_t r = 0; r < RANK; ++r) {
                            for (uint32_t j = 0; j < STATE_DIM; ++j) {
                                uint32_t global_j = (j + W_offset) % STATE_DIM;
                                h_V_.push_back(edge.V[r * STATE_DIM + global_j]);
                            }
                        }
                        
                        h_precisions_.push_back(edge.precision);
                        h_phase_couplings_.push_back(edge.phase_coupling);
                        h_global_edge_indices_.push_back(global_start_idx + e_idx);
                        current_edge_offset++;
                    }
                }
            }
        }
        h_edge_offsets_[i + 1] = current_edge_offset;
    }

    active_edges_count_ = h_edge_targets_.size();
    std::cout << "[VRAMCache] Carregando subgrafo: " << active_nodes_count_ << " nos, " << active_edges_count_ << " arestas.\n";
    if (active_edges_count_ > max_edges_) {
        std::cerr << "[VRAMCache] Subgrafo ativo excede capacidade maxima de arestas ("
                  << active_edges_count_ << " > " << max_edges_ << ").\n";
        return false;
    }

    // 3. Transferência assíncrona de memória Host-to-Device (H2D)
    std::cout << "[VRAMCache] Iniciando cópia H2D...\n";
    cudaMemcpy(d_node_ids_, h_node_ids_.data(), active_nodes_count_ * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mu_, h_mu_.data(), active_nodes_count_ * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_mu_, h_target_mu_.data(), active_nodes_count_ * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_is_input_, h_is_input_.data(), active_nodes_count_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_offsets_, h_edge_offsets_.data(), (active_nodes_count_ + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_targets_, h_edge_targets_.data(), active_edges_count_ * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_, h_U_.data(), active_edges_count_ * STATE_DIM * RANK * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_, h_V_.data(), active_edges_count_ * RANK * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_precisions_, h_precisions_.data(), active_edges_count_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phase_couplings_, h_phase_couplings_.data(), active_edges_count_ * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "[VRAMCache] Sincronizando dispositivo...\n";
    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        std::cerr << "[VRAMCache] Erro na sincronização H2D: " << cudaGetErrorString(sync_err) << "\n";
        return false;
    }
    std::cout << "[VRAMCache] Subgrafo carregado com sucesso na GPU.\n";
    return true;
}

bool VRAMCache::download_states(std::vector<float>& out_mu) {
    out_mu.resize(active_nodes_count_ * STATE_DIM);
    cudaError_t err = cudaMemcpy(out_mu.data(), d_mu_, active_nodes_count_ * STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    return err == cudaSuccess;
}

bool VRAMCache::download_weights_and_sync(SSDStorage& storage, uint32_t window_index) {
    if (active_edges_count_ == 0) return true;

    // 1. Baixar os pesos U e V atualizados da GPU
    h_U_.resize(active_edges_count_ * STATE_DIM * RANK);
    h_V_.resize(active_edges_count_ * RANK * STATE_DIM);
    cudaError_t err = cudaMemcpy(h_U_.data(), d_U_, active_edges_count_ * STATE_DIM * RANK * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[VRAMCache] Erro ao baixar pesos U da GPU: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    err = cudaMemcpy(h_V_.data(), d_V_, active_edges_count_ * RANK * STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "[VRAMCache] Erro ao baixar pesos V da GPU: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // 2. Ler todos os nós e arestas atuais do storage
    std::vector<DiskNode> current_nodes;
    const DiskNode* nodes_ptr = storage.get_nodes_ptr();
    size_t node_count = storage.get_node_count();
    if (!nodes_ptr || node_count == 0) return false;
    current_nodes.assign(nodes_ptr, nodes_ptr + node_count);

    std::vector<DiskEdge> current_edges;
    const DiskEdge* edges_ptr = storage.get_edges_ptr();
    size_t edge_count = storage.get_edge_count();
    if (edge_count > 0 && edges_ptr) {
        current_edges.assign(edges_ptr, edges_ptr + edge_count);
    }

    // 3. Atualizar os pesos das arestas na lista global aplicando a rotacao inversa
    uint32_t W_offset = (window_index * (STATE_DIM / 2)) % STATE_DIM;
    for (size_t e = 0; e < active_edges_count_; ++e) {
        if (e < h_global_edge_indices_.size()) {
            uint64_t global_idx = h_global_edge_indices_[e];
            if (global_idx < current_edges.size()) {
                // Rotação inversa para U
                for (uint32_t d = 0; d < STATE_DIM; ++d) {
                    uint32_t global_d = (d + W_offset) % STATE_DIM;
                    for (uint32_t r = 0; r < RANK; ++r) {
                        current_edges[global_idx].U[global_d * RANK + r] = h_U_[(e * STATE_DIM + d) * RANK + r];
                    }
                }
                // Rotação inversa para V
                for (uint32_t r = 0; r < RANK; ++r) {
                    for (uint32_t j = 0; j < STATE_DIM; ++j) {
                        uint32_t global_j = (j + W_offset) % STATE_DIM;
                        current_edges[global_idx].V[r * STATE_DIM + global_j] = h_V_[(e * RANK + r) * STATE_DIM + j];
                    }
                }
            }
        }
    }

    // 4. Salvar novamente no SSD
    return storage.write_graph(current_nodes, current_edges);
}

} // namespace rcm
