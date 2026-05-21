#pragma once
#include "common.hpp"
#include "ssd_storage.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <sstream>

namespace rcm {

class Projector {
public:
    Projector();
    ~Projector();

    // Varre o banco de nós no SSD para construir a tabela de tradução de strings <-> IDs
    bool initialize(const SSDStorage& storage);

    // Converte a string de entrada do usuário em seeds de ressonância
    void text_to_seeds(const std::string& text, 
                       std::vector<uint64_t>& out_seeds, 
                       std::vector<std::vector<float>>& out_inputs) const;

    // Traduz o manifold de ativação da GPU em uma lista de termos ordenados por ressonância
    std::vector<std::pair<std::string, float>> project_states(
        const std::vector<float>& mus, 
        const std::vector<uint64_t>& local_to_global,
        const std::vector<uint64_t>& input_seeds
    ) const;

    // Constrói uma sentença contínua baseada no sequenciamento de fases e regras gramaticais
    std::string build_sentence(const std::vector<std::pair<std::string, float>>& projections, const SSDStorage& storage) const;

private:
    std::unordered_map<std::string, uint64_t> name_to_id_;
    std::unordered_map<uint64_t, std::string> id_to_name_;
    std::unordered_map<uint64_t, std::string> id_to_category_;

    std::string sanitize(const std::string& word) const;
};

} // namespace rcm
