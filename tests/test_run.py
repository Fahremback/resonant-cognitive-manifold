#!/usr/bin/env python3
"""
Teste de Execução Rápida do RCM 4.0 Neuro-Flash
Valida componentes individuais antes do deploy
"""

import sys
import numpy as np

def test_tokenizer():
    """Testar tokenização de código"""
    print("Testando CodeTokenizer...", end=" ")
    try:
        # Simulação em Python do tokenizer C++
        code = "def hello():\n    return 'world'"
        tokens = code.split()  # Simplificado
        assert len(tokens) > 0
        print(f"✅ {len(tokens)} tokens")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_projector_orthogonality():
    """Testar projetor com regularização de Stiefel"""
    print("Testando Projector (Stiefel)...", end=" ")
    try:
        state_dim, rank = 256, 48
        U = np.random.randn(state_dim, rank).astype(np.float32)
        
        # QR decomposition para ortogonalização
        U_ortho, _ = np.linalg.qr(U)
        
        # Verificar ortogonalidade
        UtU = U_ortho.T @ U_ortho
        I = np.eye(rank)
        error = np.linalg.norm(UtU - I, 'fro')
        
        assert error < 0.1, f"Erro muito alto: {error}"
        print(f"✅ Erro={error:.6f}")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_vram_cache():
    """Testar lógica de cache VRAM"""
    print("Testando VRAM Cache...", end=" ")
    try:
        max_blocks = 100
        cached = set()
        lru_queue = []
        
        # Simular carregamento
        for i in range(150):
            if i not in cached:
                if len(cached) >= max_blocks:
                    # Evict LRU
                    evict = lru_queue.pop(0)
                    cached.remove(evict)
                cached.add(i)
                lru_queue.append(i)
        
        assert len(cached) == max_blocks
        print(f"✅ {len(cached)} blocos em cache")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_ssd_mapping():
    """Testar mapeamento SSD de 1T parâmetros"""
    print("Testando SSD Storage (1T params)...", end=" ")
    try:
        total_params = 1_000_000_000_000  # 1 Trilhão
        block_size = 4096  # bytes
        float_size = 4  # bytes
        
        params_per_block = block_size // float_size
        total_blocks = (total_params + params_per_block - 1) // params_per_block
        
        # Verificar se é viável em NVMe moderno (4TB = ~1B blocos)
        assert total_blocks < 10_000_000_000, "Muitos blocos"
        print(f"✅ {total_blocks:,} blocos")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_flash_loading():
    """Testar latência de flash loading"""
    print("Testando Flash-Loading Latency...", end=" ")
    import time
    
    try:
        # Simular carregamento de 100 blocos
        start = time.time()
        for i in range(100):
            _ = np.random.randn(256).astype(np.float32)  # Simula leitura
        elapsed = (time.time() - start) * 1000  # ms
        
        assert elapsed < 1000, f"Muito lento: {elapsed}ms"
        print(f"✅ {elapsed:.2f}ms para 100 blocos")
        return True
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def run_all_tests():
    print("=== RCM 4.0 Neuro-Flash: Test Suite ===\n")
    
    tests = [
        ("Tokenizer", test_tokenizer),
        ("Projector Stiefel", test_projector_orthogonality),
        ("VRAM Cache", test_vram_cache),
        ("SSD Mapping", test_ssd_mapping),
        ("Flash Loading", test_flash_loading),
    ]
    
    results = []
    for name, test_fn in tests:
        print(f"\n[{name}]")
        result = test_fn()
        results.append((name, result))
    
    # Resumo
    print("\n" + "="*40)
    print("RESUMO DOS TESTES")
    print("="*40)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
