import numpy as np

# Dimension of state vector
STATE_DIM = 1024

nodes = ["crie", "o", "jogo", "da", "cobrinha", "import", "turtle", "\\n", "time", "random"]
num_nodes = len(nodes)

# Initialize states
mu = np.zeros((num_nodes, STATE_DIM))
target_mu = np.zeros((num_nodes, STATE_DIM))
is_input = np.zeros(num_nodes)

# Seeds/Prompts are pinned to 1.0 at d=0
for i in range(5):
    mu[i, 0] = 1.0
    target_mu[i, 0] = 1.0
    is_input[i] = 1.0

# Setup edges: (src, tgt, phase_coupling, weight_dict, precision)
edges = []

# Prompt to import: phase_coupling = 0, weight[0] = 1.0, precision = 2.0
for p in range(5):
    edges.append((p, 5, 0.0, {0: 1.0}, 2.0))

# Sequence of snake game
edges.append((5, 6, 1.0, {1: 1.0}, 1.0))
edges.append((6, 7, 1.0, {2: 1.0}, 1.0))
edges.append((7, 5, 1.0, {3: 1.0, 6: 1.0}, 1.0))
edges.append((5, 8, 1.0, {4: 1.0}, 1.0))
edges.append((8, 7, 1.0, {5: 1.0}, 1.0))
edges.append((5, 9, 1.0, {7: 1.0}, 1.0))
edges.append((9, 7, 1.0, {8: 1.0}, 1.0))

# Run gradient descent iterations
alpha = 0.2
iterations = 5000
prev_energy = 999999.0
tolerance = 1e-5

converged_iter = -1
for iter in range(iterations):
    grad = np.zeros_like(mu)
    local_energy = 0.0
    
    # Compute prediction errors and gradients
    for src, tgt, pc, w_dict, prec in edges:
        shift = int(round(pc))
        for d, w in w_dict.items():
            idx_src = (d - shift) % STATE_DIM
            e = mu[tgt, d] - w * mu[src, idx_src]
            grad[tgt, d] += prec * e
            local_energy += 0.5 * prec * e * e

    energy_diff = abs(prev_energy - local_energy)
    if iter > 5 and energy_diff < tolerance:
        converged_iter = iter + 1
        break
        
    prev_energy = local_energy

    # Update states (vectorized)
    grad_clipped = np.clip(grad, -8.0, 8.0)
    mask = (is_input[:, None] > 0.5) & (target_mu > 0.0)
    mu = np.where(mask, target_mu, np.maximum(0.0, mu - alpha * grad_clipped))

print(f"Converged in {converged_iter} iterations. Final energy: {prev_energy:.6f}")
