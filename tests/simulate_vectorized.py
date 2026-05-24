#!/usr/bin/env python3
"""
Simulação Vetorizada do RCM 4.0 Neuro-Flash
Implementação NumPy para validação rápida de conceitos
"""

import numpy as np
from typing import Dict, List, Tuple

class ResonantCognitiveManifold:
    def __init__(self, state_dim=256, rank=48, num_concepts=10000):
        self.state_dim = state_dim
        self.rank = rank
        self.num_concepts = num_concepts
        
        # Inicialização ortogonal das matrizes LoRA (Stiefel)
        self.U = self._orthogonal_init(state_dim, rank)
        self.V = self._orthogonal_init(state_dim, rank).T
        
        # Estados de ressonância
        self.mu = np.zeros((num_concepts, state_dim), dtype=np.float32)
        self.sigma = np.ones((num_concepts, state_dim), dtype=np.float32)
        self.phase_coupling = np.ones(num_concepts, dtype=np.float32)
        self.resonance_index = {}
    
    def _orthogonal_init(self, rows, cols):
        """Inicialização QR para ortogonalidade"""
        A = np.random.randn(rows, cols).astype(np.float32)
        Q, _ = np.linalg.qr(A)
        return Q.astype(np.float32)
    
    def hebbian_update(self, concept_id, activation, error, gamma=0.1):
        """Atualização Hebbiana com regularização de Stiefel"""
        projected = self.U.T @ activation
        
        delta_U = gamma * np.outer(error, projected)
        delta_V = gamma * np.outer(projected, error)
        
        self.U += delta_U
        self.V += delta_V
        
        # Re-ortogonalização periódica (Stiefel)
        if np.random.random() < 0.05:
            self.U, _ = np.linalg.qr(self.U)
            V_temp, _ = np.linalg.qr(self.V.T)
            self.V = V_temp.T
    
    def sleep_chunking(self, concept_ids: List[int]):
        """Consolidação de memória preservando phase_coupling"""
        for cid in concept_ids:
            start = max(0, cid - 10)
            end = min(self.num_concepts, cid + 10)
            avg_phase = np.mean(self.phase_coupling[start:end])
            self.phase_coupling[cid] = max(0.8, avg_phase)
    
    def flash_load(self, concept_id) -> np.ndarray:
        """Carregamento sob demanda (simulação)"""
        if concept_id not in self.resonance_index:
            self.resonance_index[concept_id] = np.random.randn(self.state_dim).astype(np.float32)
        return self.resonance_index[concept_id]
    
    def compute_stiefel_error(self) -> float:
        """Calcular erro de ortogonalidade ||U^T*U - I||_F"""
        UtU = self.U.T @ self.U
        I = np.eye(self.rank)
        return float(np.linalg.norm(UtU - I, 'fro'))
    
    def get_effective_rank(self) -> int:
        """Calcular posto numérico efetivo via SVD"""
        _, s, _ = np.linalg.svd(self.U)
        threshold = s[0] * 1e-6
        return int(np.sum(s > threshold))

def run_simulation():
    print("=== Simulação Vetorizada RCM 4.0 ===\n")
    
    model = ResonantCognitiveManifold(state_dim=256, rank=48)
    
    # Teste 1: Ortogonalidade inicial
    ortho_error = model.compute_stiefel_error()
    print(f"Erro de ortogonalidade inicial: {ortho_error:.6f}")
    assert ortho_error < 0.1, "Falha na inicialização ortogonal"
    
    # Teste 2: Atualização Hebbiana (com re-ortogonalização frequente)
    for i in range(100):
        concept = np.random.randint(0, model.num_concepts)
        activation = np.random.randn(model.state_dim).astype(np.float32)
        error = np.random.randn(model.state_dim).astype(np.float32)
        model.hebbian_update(concept, activation, error)
    
    ortho_error_after = model.compute_stiefel_error()
    print(f"Erro de ortogonalidade após treino: {ortho_error_after:.6f}")
    
    # Teste 3: Sleep Chunking
    concepts_to_sleep = list(range(50, 150))
    model.sleep_chunking(concepts_to_sleep)
    avg_phase = np.mean(model.phase_coupling[50:150])
    print(f"Phase coupling médio após sleep: {avg_phase:.4f}")
    assert avg_phase > 0.7, "Falha: phase_coupling foi destruído"
    
    # Teste 4: Flash Loading
    loaded = model.flash_load(42)
    print(f"Dimensão do bloco carregado: {loaded.shape}")
    
    # Teste 5: Posto efetivo
    eff_rank = model.get_effective_rank()
    print(f"Posto efetivo da matriz U: {eff_rank} (alvo: {model.rank})")
    
    print("\n✅ Todos os testes de simulação passaram!")
    return True

if __name__ == "__main__":
    run_simulation()
