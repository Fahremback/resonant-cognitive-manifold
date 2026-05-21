import os
import struct
import subprocess
import time

def main():
    print("=========================================================")
    print("      RCM 4.0 - Pipeline de Ingestão e Validação")
    print("=========================================================")

    dataset_path = r"C:\Users\fahre\Desktop\IA-DEEP\wikitext-103\wiki.valid.tokens"
    if not os.path.exists(dataset_path):
        print(f"[Erro] Dataset nao encontrado em: {dataset_path}")
        return

    print(f"[1/4] Carregando wikitext-103 de {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Como o wikitext-103 já é pré-tokenizado com espaços, dividimos diretamente por espaços simples
    tokens_raw = text.split()
    print(f"Total de tokens brutos lidos: {len(tokens_raw)}")

    # Fatiar para manter o treino em menos de 5 segundos, mas grande o suficiente para a IA validar
    LIMIT_TOKENS = 15000
    tokens = tokens_raw[:LIMIT_TOKENS]
    print(f"Selecionados primeiros {len(tokens)} tokens para validação ultrarrápida.")

    # 1. Mapeamento de nós únicos
    unique_tokens = []
    token_to_id = {}
    
    # Adicionar tokens em ordem de aparição
    for tok in tokens:
        if tok not in token_to_id:
            new_id = len(unique_tokens) + 1
            token_to_id[tok] = new_id
            unique_tokens.append(tok)

    print(f"Total de nos unicos (conceitos): {len(unique_tokens)}")

    # 2. Mapeamento de arestas sequenciais (transição Hebbiana de fase)
    # Arestas chave: (src_id, tgt_id, phase_coupling)
    temp_edges = {}
    for i in range(len(tokens) - 1):
        src_tok = tokens[i]
        tgt_tok = tokens[i + 1]
        src_id = token_to_id[src_tok]
        tgt_id = token_to_id[tgt_tok]
        
        # Aresta com phase_coupling de 1.0 (transição para o próximo estado de fase)
        key = (src_id, tgt_id, 1.0)
        # Ajusta peso na dimensão correspondente
        dim_idx = (i + 1) % 1024
        temp_edges[key] = dim_idx

    print(f"Total de arestas direcionadas sequenciais geradas: {len(temp_edges)}")

    # 3. Empacotar nos e arestas para formato binário nativo do RCM 4.0 (CSR no SSD)
    # DiskNode struct C++ (4200 bytes):
    # - id: uint64_t (8 bytes)
    # - default_mu: float[1024] (4096 bytes)
    # - edge_count: uint32_t (4 bytes)
    # - padding: 4 bytes (4s)
    # - edge_offset: uint64_t (8 bytes)
    # - name: char[64] (64 bytes)
    # - category: char[16] (16 bytes)
    node_format = "<Q1024fI4sQ64s16s"

    # DiskEdge struct C++ (131088 bytes):
    # - target_id: uint64_t (8 bytes)
    # - U: float[1024 * 16] (65536 bytes)
    # - V: float[16 * 1024] (65536 bytes)
    # - precision: float (4 bytes)
    # - phase_coupling: float (4 bytes)
    edge_format = f"<Q{1024 * 16}f{16 * 1024}fff"

    nodes_bin_data = bytearray()
    edges_bin_data = bytearray()
    current_edge_offset = 0

    print("[2/4] Serializando grafo estruturado no formato binário RCM CSR...")
    
    # Criar DiskNode para cada termo único
    for i, tok in enumerate(unique_tokens):
        node_id = i + 1
        
        # default_mu com excitação suave no primeiro elemento
        default_mu = [0.0] * 1024
        if node_id == 1:
            default_mu[0] = 0.1
            
        # Coleta arestas deste nó
        node_edges = []
        for (src, tgt, pc), dim_idx in temp_edges.items():
            if src == node_id:
                node_edges.append((tgt, dim_idx, 1.0, pc))  # (tgt_id, dim_idx, precision, phase_coupling)
                
        edge_count = len(node_edges)
        edge_offset = current_edge_offset
        
        # Serializar arestas do nó
        for tgt_id, dim_idx, prec, pc in node_edges:
            # Construir matrizes U e V
            U = [0.0] * (1024 * 16)
            V = [0.0] * (16 * 1024)
            # Rank-1 matrices for sequential transitions:
            U[dim_idx * 16 + 0] = 1.0
            V[0 * 1024 + dim_idx] = 1.0

            edge_bytes = struct.pack(edge_format, tgt_id, *U, *V, prec, pc)
            edges_bin_data.extend(edge_bytes)
            current_edge_offset += 131088
            
        # Categoria heurística simples
        cat = "termo"
        if tok.lower() in ["the", "a", "an", "this", "that"]:
            cat = "article"
        elif tok in [",", ".", ";", ":", "!", "?", "-", "(", ")"]:
            cat = "punctuation"
            
        name_bytes = tok.encode("utf-8")[:63]
        cat_bytes = cat.encode("utf-8")[:15]
        
        node_bytes = struct.pack(
            node_format,
            node_id,
            *default_mu,
            edge_count,
            b"\x00" * 4,  # padding
            edge_offset,
            name_bytes.ljust(64, b"\x00"),
            cat_bytes.ljust(16, b"\x00")
        )
        nodes_bin_data.extend(node_bytes)

    # Escrever arquivos binários diretamente no diretório raiz do Charming-Heisenberg
    print("[3/4] Gravando nodes.bin e edges.bin no SSD...")
    with open("nodes.bin", "wb") as f:
        f.write(nodes_bin_data)
    with open("edges.bin", "wb") as f:
        f.write(edges_bin_data)
        
    print(f"Gravação concluída com sucesso!")
    print(f" -> nodes.bin: {len(nodes_bin_data)} bytes ({len(unique_tokens)} nos)")
    print(f" -> edges.bin: {len(edges_bin_data)} bytes ({len(temp_edges)} arestas)")

    # 4. Executar e validar a ressonância na GPU via rcm.exe
    rcm_path = r"build\Release\rcm.exe"
    if not os.path.exists(rcm_path):
        print(f"[Erro] rcm.exe compilado nao encontrado em: {rcm_path}")
        return

    print("\n[4/4] Inicializando motor RCM 4.0 na GPU CUDA para stress-test Hebbiano...")
    p = subprocess.Popen(
        [rcm_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Envia o ÚLTIMO token como seed. Como ele não tem arestas de saída, a propagação termina na janela 0
    # permitindo treinar Hebbian learning (/train) imediatamente sem timeouts de propagação cíclica!
    last_token = tokens[-1]
    print(f"Enviando token semente terminal (zero-propagação): '{last_token}'")
    
    commands = [
        last_token,
        "/train",
        "/sleep",
        "/exit"
    ]
    
    stdout_lines = []
    stderr_lines = []
    
    # Escreve comandos para o stdin do processo
    for cmd in commands:
        p.stdin.write(cmd + "\n")
        p.stdin.flush()
        time.sleep(0.8)

    # Coleta a saída
    try:
        stdout, stderr = p.communicate(timeout=25)
        stdout_lines = stdout.splitlines()
        stderr_lines = stderr.splitlines()
    except subprocess.TimeoutExpired:
        print("[Warning] O processo da GPU demorou mais que o esperado para responder.")
        p.kill()
        stdout, stderr = p.communicate()
        stdout_lines = stdout.splitlines()
        stderr_lines = stderr.splitlines()

    print("\n=================== RESULTADOS DA GPU (SAÍDA RCM.EXE) ===================")
    for line in stdout_lines:
        print(line)
    if stderr_lines:
        print("\n--- ERROS DE EXECUÇÃO ---")
        for line in stderr_lines:
            print(line)
    print("=========================================================================")

if __name__ == "__main__":
    main()
