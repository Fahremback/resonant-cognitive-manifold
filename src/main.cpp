#define NOMINMAX
#include "ssd_storage.hpp"
#include "vram_cache.hpp"
#include "resonance_engine.h"
#include "projector.hpp"
#include "code_tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <unordered_map>

// Função auxiliar para inicializar o banco de dados semântico estruturado RCM 4.0
bool setup_mock_database(rcm::SSDStorage& storage) {
    std::cout << "[RCM Config] Criando banco de dados semantico RCM 4.0 no SSD...\n";

    // Jogo da cobrinha compacto e funcional em Python usando Turtle
    std::string snake_code = 
        "import turtle\n"
        "import time\n"
        "import random\n"
        "\n"
        "delay = 0.1\n"
        "wn = turtle.Screen()\n"
        "wn.title(\"Snake\")\n"
        "wn.bgcolor(\"black\")\n"
        "wn.setup(width=600, height=600)\n"
        "wn.tracer(0)\n"
        "\n"
        "head = turtle.Turtle()\n"
        "head.shape(\"square\")\n"
        "head.color(\"white\")\n"
        "head.penup()\n"
        "head.direction = \"stop\"\n"
        "\n"
        "food = turtle.Turtle()\n"
        "food.shape(\"circle\")\n"
        "food.color(\"red\")\n"
        "food.penup()\n"
        "food.goto(0, 100)\n"
        "\n"
        "segments = []\n"
        "\n"
        "def go_up():\n"
        "    if head.direction != \"down\": head.direction = \"up\"\n"
        "def go_down():\n"
        "    if head.direction != \"up\": head.direction = \"down\"\n"
        "def go_left():\n"
        "    if head.direction != \"right\": head.direction = \"left\"\n"
        "def go_right():\n"
        "    if head.direction != \"left\": head.direction = \"right\"\n"
        "\n"
        "def move():\n"
        "    if head.direction == \"up\": head.sety(head.ycor() + 20)\n"
        "    if head.direction == \"down\": head.sety(head.ycor() - 20)\n"
        "    if head.direction == \"left\": head.setx(head.xcor() - 20)\n"
        "    if head.direction == \"right\": head.setx(head.xcor() + 20)\n"
        "\n"
        "wn.listen()\n"
        "wn.onkeypress(go_up, \"w\")\n"
        "wn.onkeypress(go_down, \"s\")\n"
        "wn.onkeypress(go_left, \"a\")\n"
        "wn.onkeypress(go_right, \"d\")\n"
        "\n"
        "for _ in range(50):\n"
        "    wn.update()\n"
        "    if head.xcor() > 290 or head.xcor() < -290 or head.ycor() > 290 or head.ycor() < -290:\n"
        "        time.sleep(1)\n"
        "        head.goto(0, 0)\n"
        "        head.direction = \"stop\"\n"
        "        for s in segments: s.goto(1000, 1000)\n"
        "        segments.clear()\n"
        "    if head.distance(food) < 20:\n"
        "        food.goto(random.randint(-280, 280), random.randint(-280, 280))\n"
        "        new_segment = turtle.Turtle()\n"
        "        new_segment.shape(\"square\")\n"
        "        new_segment.color(\"grey\")\n"
        "        new_segment.penup()\n"
        "        segments.append(new_segment)\n"
        "    for index in range(len(segments)-1, 0, -1):\n"
        "        segments[index].goto(segments[index-1].xcor(), segments[index-1].ycor())\n"
        "    if len(segments) > 0:\n"
        "        segments[0].goto(head.xcor(), head.ycor())\n"
        "    move()\n"
        "    for s in segments:\n"
        "        if s.distance(head) < 20:\n"
        "            time.sleep(1)\n"
        "            head.goto(0, 0)\n"
        "            head.direction = \"stop\"\n"
        "            for seg in segments: seg.goto(1000, 1000)\n"
        "            segments.clear()\n"
        "    time.sleep(delay)\n";

    std::vector<std::string> tokens = rcm::CodeTokenizer::tokenize(snake_code);
    std::cout << "[RCM Config] Tokenizado com sucesso! Total de tokens no codigo: " << tokens.size() << "\n";
    if (tokens.size() > rcm::STATE_DIM) {
        std::cerr << "[Warning] O codigo excede STATE_DIM (" << tokens.size() << " > " << rcm::STATE_DIM << ").\n";
    }

    // 1. Coletar tokens únicos e criar mapeamento para ID de nó
    std::vector<std::string> unique_tokens;
    std::unordered_map<std::string, uint64_t> token_to_id;
    for (const auto& tok : tokens) {
        if (token_to_id.find(tok) == token_to_id.end()) {
            uint64_t new_id = unique_tokens.size() + 1;
            token_to_id[tok] = new_id;
            unique_tokens.push_back(tok);
        }
    }

    // Adiciona palavras do prompt para mapear intenção do usuário
    std::vector<std::string> prompt_words = {
        "crie", "o", "jogo", "da", "cobrinha", "do", "snake", "game", "gerar"
    };
    for (const auto& p_word : prompt_words) {
        if (token_to_id.find(p_word) == token_to_id.end()) {
            uint64_t new_id = unique_tokens.size() + 1;
            token_to_id[p_word] = new_id;
            unique_tokens.push_back(p_word);
        }
    }

    // 2. Criar os nós de disco
    std::vector<rcm::DiskNode> nodes;
    for (size_t i = 0; i < unique_tokens.size(); ++i) {
        rcm::DiskNode node;
        node.id = i + 1;
        std::fill(std::begin(node.default_mu), std::end(node.default_mu), 0.0f);
        node.edge_count = 0;
        node.edge_offset = 0;
        
        // Categoria fictícia para metadados legíveis
        std::string cat = "termo";
        if (unique_tokens[i] == "import" || unique_tokens[i] == "def" || unique_tokens[i] == "if" || 
            unique_tokens[i] == "for" || unique_tokens[i] == "in" || unique_tokens[i] == "while") {
            cat = "keyword";
        } else if (unique_tokens[i] == "\n") {
            cat = "newline";
        } else if (unique_tokens[i] == "    " || unique_tokens[i] == "        ") {
            cat = "indent";
        } else if (std::find(prompt_words.begin(), prompt_words.end(), unique_tokens[i]) != prompt_words.end()) {
            cat = "prompt";
        }
        
        strncpy_s(node.name, sizeof(node.name), unique_tokens[i].c_str(), _TRUNCATE);
        strncpy_s(node.category, sizeof(node.category), cat.c_str(), _TRUNCATE);
        nodes.push_back(node);
    }

    // Configura o primeiro token da sequência como o seed com default_mu[0] = 0.1f
    if (!tokens.empty()) {
        uint64_t start_id = token_to_id[tokens[0]];
        nodes[start_id - 1].default_mu[0] = 0.1f;
    }

    // 3. Montar as arestas temporárias multiplexadas por fase
    struct TempEdgeKey {
        uint64_t source_id;
        uint64_t target_id;
        float phase_coupling;
        bool operator==(const TempEdgeKey& other) const {
            return source_id == other.source_id && target_id == other.target_id && phase_coupling == other.phase_coupling;
        }
    };

    struct TempEdgeKeyHash {
        std::size_t operator()(const TempEdgeKey& k) const {
            return std::hash<uint64_t>()(k.source_id) ^ (std::hash<uint64_t>()(k.target_id) << 1) ^ (std::hash<float>()(k.phase_coupling) << 2);
        }
    };

    struct TempEdgeValue {
        float U[rcm::STATE_DIM * rcm::RANK];
        float V[rcm::RANK * rcm::STATE_DIM];
        float precision;
    };

    std::unordered_map<TempEdgeKey, TempEdgeValue, TempEdgeKeyHash> temp_edges;

    for (size_t j = 0; j < tokens.size() - 1; ++j) {
        uint64_t src = token_to_id[tokens[j]];
        uint64_t tgt = token_to_id[tokens[j + 1]];

        // Aresta direta
        TempEdgeKey fk = {src, tgt, 1.0f};
        if (temp_edges.find(fk) == temp_edges.end()) {
            TempEdgeValue val;
            std::fill(std::begin(val.U), std::end(val.U), 0.0f);
            std::fill(std::begin(val.V), std::end(val.V), 0.0f);
            val.precision = 1.0f;
            temp_edges[fk] = val;
        }
        uint32_t dim_idx = (j + 1) % rcm::STATE_DIM;
        temp_edges[fk].U[dim_idx * rcm::RANK + 0] = 1.0f;
        temp_edges[fk].V[0 * rcm::STATE_DIM + dim_idx] = 1.0f;
    }

    // Adiciona conexões de acoplamento direto dos tokens de prompt para o token semente inicial 'import'
    if (!tokens.empty()) {
        uint64_t import_id = token_to_id[tokens[0]];
        for (const auto& p_word : prompt_words) {
            uint64_t src = token_to_id[p_word];
            if (src == import_id) continue;
            
            // Aresta direta de acoplamento instantâneo (phase_coupling = 0.0f)
            TempEdgeKey pk = {src, import_id, 0.0f};
            if (temp_edges.find(pk) == temp_edges.end()) {
                TempEdgeValue val;
                std::fill(std::begin(val.U), std::end(val.U), 0.0f);
                std::fill(std::begin(val.V), std::end(val.V), 0.0f);
                val.U[0 * rcm::RANK + 0] = 1.0f;
                val.V[0 * rcm::STATE_DIM + 0] = 1.0f;
                val.precision = 1.0f / static_cast<float>(prompt_words.size());
                temp_edges[pk] = val;
            }
        }
    }

    // 4. Montar a estrutura CSR
    std::vector<rcm::DiskEdge> final_edges;
    uint64_t current_offset_bytes = 0;

    for (auto& node : nodes) {
        node.edge_offset = current_offset_bytes;
        node.edge_count = 0;

        for (const auto& pair : temp_edges) {
            if (pair.first.source_id == node.id) {
                rcm::DiskEdge edge;
                edge.target_id = pair.first.target_id;
                edge.precision = pair.second.precision;
                edge.phase_coupling = pair.first.phase_coupling;
                for (uint32_t d = 0; d < rcm::STATE_DIM * rcm::RANK; ++d) {
                    edge.U[d] = pair.second.U[d];
                }
                for (uint32_t d = 0; d < rcm::RANK * rcm::STATE_DIM; ++d) {
                    edge.V[d] = pair.second.V[d];
                }
                final_edges.push_back(edge);
                node.edge_count++;
                current_offset_bytes += sizeof(rcm::DiskEdge);
            }
        }
    }

    return storage.write_graph(nodes, final_edges);
}

int main() {
    // Desativar buffering do stdout para garantir descarga imediata em pipes
    std::setvbuf(stdout, NULL, _IONBF, 0);
    std::cout << std::unitbuf;

    std::cout << "=========================================================\n";
    std::cout << "    Resonant Cognitive Manifold (RCM) 4.0 - C++/CUDA\n";
    std::cout << "=========================================================\n";

    rcm::SSDStorage storage;
    std::ifstream check_nodes("nodes.bin", std::ios::binary);
    std::ifstream check_edges("edges.bin", std::ios::binary);
    bool db_exists = check_nodes.good() && check_edges.good();
    check_nodes.close();
    check_edges.close();

    if (db_exists) {
        std::cout << "[SSDStorage] Detectados arquivos binarios existentes no disco. Carregando...\n";
        if (!storage.open_db("nodes.bin", "edges.bin")) {
            std::cerr << "[Fatal] Falha ao abrir banco de dados semantico existente.\n";
            return -1;
        }
    } else {
        if (!setup_mock_database(storage)) {
            std::cerr << "[Fatal] Falha ao configurar banco de dados semantico RCM 4.0.\n";
            return -1;
        }
    }

    std::cout << "[SSDStorage] Banco RCM 4.0 carregado com sucesso!\n";
    std::cout << "             Total de nos (conceitos): " << storage.get_node_count() << "\n";
    std::cout << "             Total de arestas (links): " << storage.get_edge_count() << "\n\n";

    rcm::Projector projector;
    if (!projector.initialize(storage)) {
        std::cerr << "[Fatal] Falha ao inicializar o projetor semantico.\n";
        return -1;
    }

    rcm::VRAMCache vram_cache(8192, 65536);
    rcm::ResonanceEngine engine;

    std::string user_input;
    std::cout << "Engine RCM 4.0 pronta. Digite comandos ou conceitos para ativar a IA.\n";
    std::cout << "Para gerar o Jogo da Cobrinha, digite:\n";
    std::cout << "  - 'crie o jogo da cobrinha' ou 'import' \n";
    std::cout << "Comandos de controle:\n";
    std::cout << "  - '/sleep'     (Consolida cliques altamente correlacionados em nos de chunking via GPU)\n";
    std::cout << "  - '/train'     (Aplica aprendizado Hebbiano no subgrafo ativo)\n";
    std::cout << "  - '/exit'      (Encerra a engine)\n\n";

    std::vector<uint64_t> last_seeds;

    while (true) {
        std::cout << "rcm> ";
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "/exit") break;

        if (user_input == "/sleep") {
            std::cout << "[Sleep Engine] Iniciando processo de consolidacao de chunks (Cognitive Chunking)...\n";
            auto correlated_pairs = engine.find_correlated_chunks(vram_cache, 0.30f);
            if (correlated_pairs.empty()) {
                std::cout << "[Sleep Engine] Nenhum par de conceitos com correlacao suficiente para fusao.\n";
                continue;
            }

            std::cout << "[Sleep Engine] Encontrados " << correlated_pairs.size() << " pares correlacionados para consolidacao.\n";

            std::vector<rcm::DiskNode> current_nodes;
            const rcm::DiskNode* nodes_ptr = storage.get_nodes_ptr();
            size_t node_count = storage.get_node_count();
            current_nodes.assign(nodes_ptr, nodes_ptr + node_count);

            std::unordered_map<uint64_t, std::vector<rcm::DiskEdge>> node_to_edges;
            for (size_t i = 0; i < node_count; ++i) {
                std::vector<rcm::DiskEdge> edges;
                storage.get_edges(current_nodes[i], edges);
                node_to_edges[current_nodes[i].id] = edges;
            }

            uint64_t next_id = current_nodes.size() + 1;
            for (const auto& pair : correlated_pairs) {
                rcm::DiskNode node_a, node_b;
                if (storage.get_node(pair.first, node_a) && storage.get_node(pair.second, node_b)) {
                    std::string new_name = std::string(node_a.name) + "_" + std::string(node_b.name);
                    std::cout << "               -> Consolidando [" << node_a.name << "] + [" << node_b.name 
                              << "] em novo no abstrato [" << new_name << "]\n";

                    rcm::DiskNode chunk_node;
                    chunk_node.id = next_id;
                    for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                        chunk_node.default_mu[d] = 0.1f;
                    }
                    chunk_node.edge_count = 0;
                    chunk_node.edge_offset = 0;
                    strncpy_s(chunk_node.name, sizeof(chunk_node.name), new_name.c_str(), _TRUNCATE);
                    strncpy_s(chunk_node.category, sizeof(chunk_node.category), "chunk", _TRUNCATE);

                    current_nodes.push_back(chunk_node);

                    // Arestas bidirecionais de alta precisão
                    rcm::DiskEdge e_a, e_b, e_c, e_d;
                    e_a.target_id = node_a.id;
                    std::fill(std::begin(e_a.U), std::end(e_a.U), 0.0f);
                    std::fill(std::begin(e_a.V), std::end(e_a.V), 0.0f);
                    for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                        e_a.U[d * rcm::RANK + (d / 64)] = 1.0f;
                        e_a.V[(d / 64) * rcm::STATE_DIM + d] = 1.0f;
                    }
                    e_a.precision = 2.0f;
                    e_a.phase_coupling = 0.0f;

                    e_b.target_id = node_b.id;
                    std::fill(std::begin(e_b.U), std::end(e_b.U), 0.0f);
                    std::fill(std::begin(e_b.V), std::end(e_b.V), 0.0f);
                    for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                        e_b.U[d * rcm::RANK + (d / 64)] = 1.0f;
                        e_b.V[(d / 64) * rcm::STATE_DIM + d] = 1.0f;
                    }
                    e_b.precision = 2.0f;
                    e_b.phase_coupling = 0.0f;

                    e_c.target_id = next_id;
                    std::fill(std::begin(e_c.U), std::end(e_c.U), 0.0f);
                    std::fill(std::begin(e_c.V), std::end(e_c.V), 0.0f);
                    for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                        e_c.U[d * rcm::RANK + (d / 64)] = 1.0f;
                        e_c.V[(d / 64) * rcm::STATE_DIM + d] = 1.0f;
                    }
                    e_c.precision = 2.0f;
                    e_c.phase_coupling = 0.0f;

                    e_d.target_id = next_id;
                    std::fill(std::begin(e_d.U), std::end(e_d.U), 0.0f);
                    std::fill(std::begin(e_d.V), std::end(e_d.V), 0.0f);
                    for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                        e_d.U[d * rcm::RANK + (d / 64)] = 1.0f;
                        e_d.V[(d / 64) * rcm::STATE_DIM + d] = 1.0f;
                    }
                    e_d.precision = 2.0f;
                    e_d.phase_coupling = 0.0f;

                    node_to_edges[next_id].push_back(e_a);
                    node_to_edges[next_id].push_back(e_b);
                    node_to_edges[node_a.id].push_back(e_c);
                    node_to_edges[node_b.id].push_back(e_d);
                    
                    next_id++;
                }
            }

            // Reconstruir CSR
            std::vector<rcm::DiskEdge> final_edges;
            uint64_t current_offset_bytes = 0;
            for (auto& node : current_nodes) {
                node.edge_offset = current_offset_bytes;
                node.edge_count = 0;

                const auto& edges = node_to_edges[node.id];
                for (const auto& edge : edges) {
                    final_edges.push_back(edge);
                    node.edge_count++;
                    current_offset_bytes += sizeof(rcm::DiskEdge);
                }
            }

            if (storage.write_graph(current_nodes, final_edges)) {
                std::cout << "[Sleep Engine] Consolidacao finalizada com sucesso no SSD!\n";
                std::cout << "               Novos nos totais no SSD: " << storage.get_node_count() << "\n";
                projector.initialize(storage);
            } else {
                std::cerr << "[Sleep Engine] Erro ao gravar grafo atualizado.\n";
            }
            continue;
        }

        if (user_input == "/train") {
            if (last_seeds.empty()) {
                std::cout << "[Treino] Nenhum estado ativo para treinar. Insira um input primeiro.\n";
                continue;
            }
            std::cout << "[Treino] Calibrando pesos via Hebbian Learning preditivo na GPU...\n";
            if (engine.run_hebbian_learning(vram_cache, 0.05f, 0.001f)) {
                std::cout << "[Treino] Sincronia concluida com sucesso. Pesos locais ajustados.\n";
                vram_cache.download_weights_and_sync(storage);
            } else {
                std::cerr << "[Treino] Erro na execucao do kernel Hebbian.\n";
            }
            continue;
        }

        if (user_input.empty()) continue;

        // 1. Converter texto do usuário em seeds de ativação
        std::vector<uint64_t> seeds;
        std::vector<std::vector<float>> inputs;
        projector.text_to_seeds(user_input, seeds, inputs);

        if (seeds.empty()) {
            std::cout << "[RCM] Nao entendi os conceitos inseridos. Tente outros termos do banco.\n";
            continue;
        }

        last_seeds = seeds;

        std::vector<std::pair<std::string, float>> all_projections;
        std::unordered_map<uint64_t, std::vector<float>> carry_over_states;

        bool has_more_generation = true;
        uint32_t window_index = 0;
        constexpr uint32_t MAX_WINDOWS = 8; // Limite ótimo para testes rápidos sem perda de coerência local
        constexpr uint32_t HALF_DIM = rcm::STATE_DIM / 2;

        while (has_more_generation && window_index < MAX_WINDOWS) {
            std::vector<uint64_t> current_seeds;
            std::vector<std::vector<float>> current_inputs;

            if (window_index == 0) {
                // Primeira janela é excitada diretamente pelo prompt
                current_seeds = seeds;
                current_inputs = inputs;
                std::cout << "\n[Window 0] Excitando prompt inicial...\n";
            } else {
                // Janelas subsequentes herdam o carry-over da janela anterior
                for (const auto& pair : carry_over_states) {
                    current_seeds.push_back(pair.first);
                    current_inputs.push_back(pair.second);
                }
                std::cout << "[Window " << window_index << "] Carregando carry-over da janela anterior (" 
                          << current_seeds.size() << " nos ativos no boundary)...\n";
            }

            // 2. Carregar o subgrafo ativo do SSD para a VRAM (passando window_index para rotacao de pesos)
            if (!vram_cache.load_subgraph(current_seeds, current_inputs, storage, true, window_index)) {
                std::cerr << "[VRAMCache] Erro ao carregar subgrafo para janela " << window_index << "\n";
                break;
            }

            // 3. Executar relaxamento (otimização de Energia Livre) na GPU
            rcm::ResonanceStats stats = engine.run_resonance(vram_cache, 0.5f, 150, -1.0f);

            // 4. Download dos estados convergidos da GPU
            std::vector<float> mus;
            if (!vram_cache.download_states(mus)) {
                std::cerr << "[VRAMCache] Falha ao fazer download de estados da GPU.\n";
                break;
            }

            std::cout << "[Debug] Window " << window_index << " completed in " << stats.iterations_completed << " iterations. Final Energy: " << stats.final_energy << "\n";
            std::cout << "[Debug] active_nodes_count: " << vram_cache.get_active_nodes_count() << "\n";
            int active_in_second_half_count = 0;
            for (size_t i = 0; i < vram_cache.get_active_nodes_count(); ++i) {
                uint64_t g_id = vram_cache.get_local_to_global()[i];
                rcm::DiskNode node;
                if (storage.get_node(g_id, node)) {
                    for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                        float val = mus[i * rcm::STATE_DIM + d];
                        if (val > 0.001f && d >= HALF_DIM) {
                            active_in_second_half_count++;
                            if (active_in_second_half_count <= 20) {
                                std::cout << "  [Debug] Node " << node.name << " (id: " << g_id 
                                          << ") active at d=" << d << " with val=" << val << "\n";
                            }
                        }
                    }
                }
            }
            std::cout << "[Debug] Total active in second half: " << active_in_second_half_count << "\n";

            // 5. Projetar estados desta janela
            auto window_projections = projector.project_states(mus, vram_cache.get_local_to_global(), current_seeds);

            // Filtra projeções da janela atual baseando-se no intervalo de fase
            std::vector<std::pair<std::string, float>> committed_projections;
            carry_over_states.clear();

            const auto& local_to_global = vram_cache.get_local_to_global();

            // Identificar estados para o carry-over (segunda metade) e comitados (primeira metade)
            for (size_t i = 0; i < local_to_global.size(); ++i) {
                uint64_t g_id = local_to_global[i];
                std::vector<float> next_input(rcm::STATE_DIM, 0.0f);
                bool has_any_active = false;

                for (uint32_t d = 0; d < rcm::STATE_DIM; ++d) {
                    float val = mus[i * rcm::STATE_DIM + d];
                    if (val > 0.001f) {
                        if (d >= HALF_DIM) {
                            next_input[d - HALF_DIM] = val; // Desloca para o início da nova janela
                            has_any_active = true;
                        }
                    }
                }

                if (has_any_active) {
                    carry_over_states[g_id] = next_input;
                }
            }

            // Determinar tokens válidos da janela
            uint32_t active_in_second_half = 0;
            for (const auto& proj : window_projections) {
                std::string raw_name = proj.first;
                // Extrai o índice do token "[t=index]"
                size_t t_pos = raw_name.find("[t=");
                if (t_pos != std::string::npos) {
                    uint32_t t_val = std::stoul(raw_name.substr(t_pos + 3));
                    if (t_val < HALF_DIM) {
                        // Ajusta o índice t para a exibição absoluta global
                        size_t end_bracket = raw_name.find("]", t_pos);
                        std::string adjusted_name = raw_name;
                        if (end_bracket != std::string::npos) {
                            uint32_t absolute_t = window_index * HALF_DIM + t_val;
                            adjusted_name.replace(t_pos + 3, end_bracket - (t_pos + 3), std::to_string(absolute_t));
                        }
                        committed_projections.push_back({adjusted_name, proj.second});
                    } else {
                        active_in_second_half++;
                    }
                }
            }

            // Adiciona as projeções da primeira metade ao log global
            all_projections.insert(all_projections.end(), committed_projections.begin(), committed_projections.end());

            // Critério de parada: se não há atividade significativa na segunda metade para continuar propagando
            if (active_in_second_half == 0 && carry_over_states.empty()) {
                has_more_generation = false;
            }

            window_index++;
        }

        // 6. Exibir o código Python gerado combinando todas as janelas
        if (all_projections.empty()) {
            std::cout << "                    Nenhuma ressonancia significativa detectada.\n";
        } else {
            // Imprime todos os tokens ativos
            for (size_t i = 0; i < all_projections.size(); ++i) {
                std::string display_name = all_projections[i].first;
                size_t nl_pos = display_name.find("\n");
                if (nl_pos != std::string::npos) {
                    display_name.replace(nl_pos, 1, "\\n");
                }
                std::cout << "                    - " << std::left << std::setw(40) 
                          << display_name << " [Ressonancia: " 
                          << std::fixed << std::setprecision(4) << all_projections[i].second << "]\n";
            }

            std::string output_sentence = projector.build_sentence(all_projections, storage);
            std::cout << "\n---------------------------------------------------------\n";
            std::cout << "IA Responde (Codigo Python Gerado com SWRE):\n" << output_sentence << "\n";
            std::cout << "---------------------------------------------------------\n\n";

            // Salva o código gerado em snake.py no diretório atual
            std::ofstream out_file("snake.py");
            if (out_file.is_open()) {
                out_file << output_sentence;
                out_file.close();
                std::cout << "[Salvar] Codigo Python gravado com sucesso em 'snake.py'!\n";
            } else {
                std::cerr << "[Salvar] Erro ao criar o arquivo 'snake.py'.\n";
            }
        }
    }

    std::cout << "[RCM Config] Encerrando engine e limpando cache...\n";
    storage.close_db();
    return 0;
}
