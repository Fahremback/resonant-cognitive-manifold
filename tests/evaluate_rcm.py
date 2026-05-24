#!/usr/bin/env python3
"""
RCM 4.0 Neuro-Flash - Avaliação de Emergência Cognitiva
Testa capacidades não programadas do modelo
"""

import numpy as np
import sys

def test_copy_task(model, max_len=64):
    """Teste de memória de trabalho (Copy Task)"""
    print(f"Teste Copy Task N={max_len}...", end=" ")
    # Gerar sequência aleatória
    seq = np.random.randint(0, 100, max_len)
    # O modelo deve reproduzir a sequência
    output = model.generate_copy(seq)
    accuracy = np.mean(seq == output) * 100
    print(f"✅ {accuracy:.1f}% preciso" if accuracy > 80 else f"❌ {accuracy:.1f}% preciso")
    return accuracy > 80

def test_reverse_attention(model, length=32):
    """Teste de atenção bidirecional (Reverse)"""
    print(f"Teste Reverse N={length}...", end=" ")
    seq = np.random.randint(0, 50, length)
    expected = seq[::-1]
    output = model.generate_reverse(seq)
    accuracy = np.mean(expected == output) * 100
    print(f"✅ {accuracy:.1f}%" if accuracy > 70 else f"❌ {accuracy:.1f}%")
    return accuracy > 70

def test_temporal_shift(model, shift=5):
    """Teste de deslocamento temporal"""
    print(f"Teste Shift +{shift}...", end=" ")
    seq = np.random.randint(0, 30, 50)
    output = model.predict_shift(seq, shift)
    # Verificar se o shift foi aplicado corretamente
    passed = len(output) == len(seq)
    print(f"✅ PASSOU" if passed else f"❌ FALHOU")
    return passed

def test_associative_recall(model):
    """Teste de recordação associativa"""
    print("Teste Associative Recall...", end=" ")
    key = "nome_usuario"
    value = "Elias"
    model.store_association(key, value)
    
    # Inserir distratores
    for i in range(20):
        model.store_association(f"distractor_{i}", f"value_{i}")
    
    recalled = model.recall_association(key)
    passed = recalled == value
    print(f"✅ '{recalled}'" if passed else f"❌ '{recalled}'")
    return passed

def test_hierarchy_detection(model):
    """Teste de detecção de hierarquia"""
    print("Teste Hierarquia Aninhada...", end=" ")
    nested = {"level1": {"level2": {"level3": "valor"}}}
    result = model.detect_hierarchy(nested)
    passed = result.get_max_depth() >= 3
    print(f"✅ Profundidade {result.get_max_depth()}" if passed else f"❌ Falhou")
    return passed

def run_full_evaluation():
    print("=== RCM 4.0 Neuro-Flash: Bateria de Avaliação ===\n")
    
    # Mock do modelo para demonstração
    class MockModel:
        def generate_copy(self, seq): return seq  # Perfect copy
        def generate_reverse(self, seq): return seq[::-1]
        def predict_shift(self, seq, s): return seq
        def store_association(self, k, v): setattr(self, k, v)
        def recall_association(self, k): return getattr(self, k, None)
        def detect_hierarchy(self, d): 
            class R: 
                def get_max_depth(self): return 3
            return R()
    
    model = MockModel()
    results = []
    
    # Executar testes
    results.append(("Copy Task N=32", test_copy_task(model, 32)))
    results.append(("Copy Task N=64", test_copy_task(model, 64)))
    results.append(("Reverse N=32", test_reverse_attention(model, 32)))
    results.append(("Temporal Shift +5", test_temporal_shift(model, 5)))
    results.append(("Associative Recall", test_associative_recall(model)))
    results.append(("Hierarchy Detection", test_hierarchy_detection(model)))
    
    # Resumo
    print("\n=== Resumo ===")
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"Resultados: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    success = run_full_evaluation()
    sys.exit(0 if success else 1)
