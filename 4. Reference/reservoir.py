import numpy as np
import networkx as nx
import pandas as pd
import time
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

def lorenz_euler(n_steps, dt=0.01, sigma=10.0, rho=28.0, beta=8/3, state0=None):
    data = np.empty((n_steps, 3))
    state = np.array([1.0, 1.0, 1.0]) if state0 is None else np.array(state0, dtype=float)

    for i in range(n_steps):
        x, y, z = state
        state = state + dt * np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ])
        data[i] = state

    return data, state

def setup_reservoir(
    file_path,
    J=1.5,
    lambda_reg=3e-6,
    t_thermalization=30,
    t_training=2000,
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
    dt=0.01,
    input_scale=1.5,
    seed=42,
    alpha=0.3
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


    lorenz_full, lorenz_last_state = lorenz_euler(
        total_steps, dt=dt, sigma=sigma, rho=rho, beta=beta
    )

    u_mean = lorenz_full.mean(axis=0)
    u_std  = lorenz_full.std(axis=0)
    u_std[u_std == 0] = 1.0

    lorenz_full_norm = (lorenz_full - u_mean) / u_std
    u_tr = lorenz_full_norm[washout_steps:]

    rng = np.random.default_rng(seed)
    W_in = input_scale * rng.normal(size=(N, 3))

    r = np.zeros(N)
    for t in range(washout_steps):
        r = (1 - alpha) * r + alpha * np.tanh(W @ r + W_in @ lorenz_full_norm[t])

    R_train = np.empty((train_steps, N))
    for t in range(train_steps):
        r = (1 - alpha) * r + alpha * np.tanh(W @ r + W_in @ u_tr[t])
        R_train[t] = r

    RtR = R_train[:-1].T @ R_train[:-1]
    RtU = R_train[:-1].T @ u_tr[1:]
    W_out = np.linalg.solve(RtR + lambda_reg * np.eye(N), RtU)
    return W, W_in, W_out, r, u_tr, u_mean, u_std, lorenz_last_state

def run_simulation(W, W_in, W_out, r_last, u_mean, u_std, n_steps=200, alpha=0.3):
    N = W.shape[0]
    R = np.empty((n_steps, N))
    Y_norm = np.empty((n_steps, 3))
    Y = np.empty((n_steps, 3))

    r = r_last.copy()

    for t in range(n_steps):
        u_hat_norm = r @ W_out          # predicted normalized (x,y,z)
        u_hat = u_hat_norm * u_std + u_mean
        Y_norm[t] = u_hat_norm
        Y[t] = u_hat
        # feedback must be normalized
        r = (1 - alpha) * r + alpha * np.tanh(W @ r + W_in @ u_hat_norm)
        R[t] = r

    return R, Y_norm, Y

def run_reservoir_workflow(
    *,
    file_path,
    J,
    lambda_reg,
    t_thermalization,
    t_training,
    t_prediction,
    sigma,
    rho,
    beta,
    dt,
    input_scale,
    seed,
    alpha
):
    # -------------------------
    # derived step counts
    # -------------------------
    washout_steps = round(t_thermalization / dt)
    train_steps = round(t_training / dt)
    n_steps = round(t_prediction / dt)

    t0_total = time.perf_counter()

    # train reservoir
    t0 = time.perf_counter()
    W, W_in, W_out, r, u_tr, u_mean, u_std, lorenz_last_state = setup_reservoir(
        file_path=file_path,
        J=J,
        lambda_reg=lambda_reg,
        t_thermalization=t_thermalization,
        t_training=t_training,
        sigma=sigma,
        rho=rho,
        beta=beta,
        dt=dt,
        input_scale=input_scale,
        seed=seed,
        alpha=alpha
    )
    t1 = time.perf_counter()
    print(f"setup_reservoir finished in {t1 - t0:.3f} s")

    # autonomous prediction
    t0 = time.perf_counter()
    R_pred, Y_pred_norm, Y_pred = run_simulation(
        W=W,
        W_in=W_in,
        W_out=W_out,
        r_last=r,
        u_mean=u_mean,
        u_std=u_std,
        n_steps=n_steps,
        alpha=alpha
    )
    t1 = time.perf_counter()
    print(f"run_simulation finished in {t1 - t0:.3f} s")

    # ground truth
    t0 = time.perf_counter()
    Y_true, lorenz_states = lorenz_euler(
        n_steps=n_steps,
        dt=dt,
        sigma=sigma,
        rho=rho,
        beta=beta,
        state0=lorenz_last_state,
    )
    t1 = time.perf_counter()
    print(f"lorenz_euler finished in {t1 - t0:.3f} s")

    t1_total = time.perf_counter()
    total_runtime = t1_total - t0_total
    print(f"Total runtime: {total_runtime:.3f} s")

    return (
        washout_steps,
        train_steps,
        n_steps,
        W,
        W_in,
        W_out,
        r,
        u_tr,
        u_mean,
        u_std,
        lorenz_last_state,
        R_pred,
        Y_pred_norm,
        Y_pred,
        Y_true,
        lorenz_states,
        total_runtime,
        alpha
    )

