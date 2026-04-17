import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

def extract_data(file_path):
    file = Path(file_path)
    G = nx.read_graphml(file)

    node_num_cols = ["dn_position_x", "dn_position_y", "dn_position_z", "dn_correspondence_id"]
    edge_num_cols = ["fiber_length_mean", "FA_mean", "number_of_fibers"]

    node_records = []
    for node_id, attrs in G.nodes(data=True):
        node_records.append({"file": file.name, "path": str(file), "node_id": node_id, **attrs})
    nodes_df = pd.DataFrame(node_records)
    for col in node_num_cols:
        if col in nodes_df.columns:
            nodes_df[col] = pd.to_numeric(nodes_df[col], errors="coerce")

    edge_records = []
    for u, v, attrs in G.edges(data=True):
        edge_records.append({"file": file.name, "path": str(file), "source": u, "target": v, **attrs})
    edges_df = pd.DataFrame(edge_records)
    for col in edge_num_cols:
        if col in edges_df.columns:
            edges_df[col] = pd.to_numeric(edges_df[col], errors="coerce")

    return G, nodes_df, edges_df

def setup_reservoir(
    file_path,
    J=0.9,
    lambda_reg=3e-6,
    t_thermalization=40,
    t_training=200,
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
    dt=0.01,
    input_scale=0.03,
    seed=42,
):
    G, nodes_df, edges_df = extract_data(file_path)

    N = G.number_of_nodes()
    node_ids = sorted(G.nodes())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    D = np.zeros((N, N))
    for u, v, attrs in G.edges(data=True):
        i, j = id_to_idx[u], id_to_idx[v]
        w = float(attrs.get("number_of_fibers", 1.0))
        D[i, j] = w
        D[j, i] = w

    rho_D = np.max(np.abs(np.linalg.eigvals(D)))
    if rho_D == 0:
        raise ValueError("connectivity matrix has zero spectral radius")
    W = (J / rho_D) * D

    washout_steps = round(t_thermalization / dt)
    train_steps   = round(t_training / dt)
    total_steps   = washout_steps + train_steps

    def lorenz_euler(n_steps):
        data = np.empty((n_steps, 3))
        state = np.array([1.0, 1.0, 1.0])
        for i in range(n_steps):
            x, y, z = state
            state = state + dt * np.array([
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z,
            ])
            data[i] = state
        return data

    lorenz_full = lorenz_euler(total_steps)
    u_tr = lorenz_full[washout_steps:]

    rng = np.random.default_rng(seed)
    W_in = input_scale * rng.normal(size=(N, 3))

    r = np.zeros(N)
    for t in range(washout_steps):
        r = np.tanh(W @ r + W_in @ lorenz_full[t])

    R = np.empty((train_steps, N))
    for t in range(train_steps):
        r = np.tanh(W @ r + W_in @ u_tr[t])
        R[t] = r

    RtR = R.T @ R
    RtU = R.T @ u_tr
    W_out = np.linalg.solve(RtR + lambda_reg * np.eye(N), RtU)

    return W, W_in, W_out, r, u_tr

def run_simulation(W, W_in, W_out, r_last, n_steps=100, dt=0.01, sigma=10.0, rho=10.0, beta=8 / 3):
    N = W.shape[0]
    traj = np.empty((n_steps, N))

    r = r_last.copy()

    for t in range(n_steps):
        u_hat = r @ W_out
        r = np.tanh(W @ r + W_in @ u_hat)
        traj[t] = r

    return traj
