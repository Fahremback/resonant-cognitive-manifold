#pragma once
#include <cstdint>

namespace rcm {

constexpr uint32_t STATE_DIM = 1024; // Dimensão vetorial do manifold de estados



// Representação física de um Nó de Manifold de Fase no SSD
struct DiskNode {
    uint64_t id;                       // ID único global
    float default_mu[STATE_DIM];       // Vetor de estado padrão (dimensão 8)
    uint32_t edge_count;               // Quantidade de arestas saindo deste nó
    uint64_t edge_offset;              // Offset binário de bytes no arquivo de arestas
    char name[64];                     // String legível do conceito
    char category[16];                 // Categoria gramatical (sujeito, verbo, objeto, pontuacao, etc.)
};

constexpr uint32_t RANK = 16;

// Representação física de uma Aresta de Manifold de Fase no SSD (Fatorização LoRA U * V)
struct DiskEdge {
    uint64_t target_id;                // ID global do nó de destino
    float U[STATE_DIM * RANK];         // Matriz U (1024 x 16)
    float V[RANK * STATE_DIM];         // Matriz V (16 x 1024)
    float precision;                   // Termo de acoplamento/precisão pi_ji
    float phase_coupling;              // Acoplamento temporal (sintaxe de transição lógica)
};

// Estatísticas retornadas pela GPU pós-execução do kernel de ressonância
struct ResonanceStats {
    float initial_energy;
    float final_energy;
    uint32_t iterations_completed;
};

} // namespace rcm
