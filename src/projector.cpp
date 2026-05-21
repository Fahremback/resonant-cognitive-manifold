#include "projector.hpp"
#include "code_tokenizer.hpp"
#include <iostream>
#include <cctype>
#include <cmath>

namespace rcm {

Projector::Projector() {}
Projector::~Projector() {}

std::string Projector::sanitize(const std::string& word) const {
    // Retorna a palavra/token inalterada para preservar a sintaxe de código original
    return word;
}

bool Projector::initialize(const SSDStorage& storage) {
    name_to_id_.clear();
    id_to_name_.clear();
    id_to_category_.clear();

    const DiskNode* nodes = storage.get_nodes_ptr();
    size_t count = storage.get_node_count();

    if (!nodes || count == 0) {
        std::cerr << "[Projector] Erro: Ponteiro de nos do SSD invalido ou banco vazio.\n";
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        std::string name_str(nodes[i].name);
        std::string clean_name = sanitize(name_str);
        if (!clean_name.empty()) {
            name_to_id_[clean_name] = nodes[i].id;
            id_to_name_[nodes[i].id] = name_str; 
            id_to_category_[nodes[i].id] = std::string(nodes[i].category);
        }
    }

    return true;
}

void Projector::text_to_seeds(const std::string& text, 
                               std::vector<uint64_t>& out_seeds, 
                               std::vector<std::vector<float>>& out_inputs) const {
    out_seeds.clear();
    out_inputs.clear();

    // Utiliza o CodeTokenizer para separar em tokens sintáticos válidos de Python
    std::vector<std::string> words = CodeTokenizer::tokenize(text);
    for (const auto& w : words) {
        std::string clean = sanitize(w);
        uint64_t target_id = 0;
        auto it = name_to_id_.find(clean);
        if (it != name_to_id_.end()) {
            target_id = it->second;
        } else {
            // Fallback para minúsculas (útil para tolerância a maiúsculas em prompts)
            std::string lower_clean = clean;
            std::transform(lower_clean.begin(), lower_clean.end(), lower_clean.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            auto it_lower = name_to_id_.find(lower_clean);
            if (it_lower != name_to_id_.end()) {
                target_id = it_lower->second;
            }
        }

        if (target_id != 0) {
            if (std::find(out_seeds.begin(), out_seeds.end(), target_id) == out_seeds.end()) {
                out_seeds.push_back(target_id);
                // Inicializa o vetor de excitação com 1.0 no primeiro elemento (t=0)
                std::vector<float> input_vec(STATE_DIM, 0.0f);
                input_vec[0] = 1.0f;
                out_inputs.push_back(input_vec);
            }
        }
    }
}

struct ProjectedConcept {
    std::string name;
    uint64_t global_id;
    float amplitude;
    uint32_t peak_phase;
};

std::vector<std::pair<std::string, float>> Projector::project_states(
    const std::vector<float>& mus, 
    const std::vector<uint64_t>& local_to_global,
    const std::vector<uint64_t>& input_seeds
) const {
    std::vector<std::pair<std::string, float>> projections;
    size_t active_nodes = local_to_global.size();

    // Para cada fase d, encontra qual nó (excluindo os de prompt) possui a maior ativação
    for (uint32_t d = 0; d < STATE_DIM; ++d) {
        float max_val = 0.0f;
        uint64_t best_g_id = 0;

        for (size_t i = 0; i < active_nodes; ++i) {
            uint64_t g_id = local_to_global[i];
            
            // Ignora nós de prompt no output gerado
            auto cat_it = id_to_category_.find(g_id);
            if (cat_it != id_to_category_.end() && cat_it->second == "prompt") {
                continue;
            }

            float val = mus[i * STATE_DIM + d];
            if (val > max_val) {
                max_val = val;
                best_g_id = g_id;
            }
        }

        // Se a ativação máxima nessa fase for significativa, projeta o token correspondente
        if (max_val > 0.05f && best_g_id != 0) {
            auto it = id_to_name_.find(best_g_id);
            if (it != id_to_name_.end()) {
                auto cat_it = id_to_category_.find(best_g_id);
                std::string cat_str = (cat_it != id_to_category_.end() && !cat_it->second.empty()) ? cat_it->second : "termo";
                std::string display_name = it->second + " (" + cat_str + ") [t=" + std::to_string(d) + "]";
                projections.push_back({display_name, max_val});
            }
        }
    }

    return projections;
}

std::string Projector::build_sentence(const std::vector<std::pair<std::string, float>>& projections, const SSDStorage& storage) const {
    if (projections.empty()) return "";

    std::vector<std::string> tokens;
    for (size_t i = 0; i < projections.size(); ++i) {
        std::string raw_name = projections[i].first;
        size_t paren = raw_name.find(" (");
        if (paren != std::string::npos) {
            raw_name = raw_name.substr(0, paren);
        }
        tokens.push_back(raw_name);
    }

    // Reconstrói o código Python usando o analisador léxico reverso
    return CodeTokenizer::detokenize(tokens);
}

} // namespace rcm
