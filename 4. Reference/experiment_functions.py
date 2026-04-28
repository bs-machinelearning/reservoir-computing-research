import numpy as np 
import networkx as nx
import time
from pathlib import Path
from reservoir import lorenz_euler

def t_div(Y_pred, Y_true, epsilon=1.0, dt=0.01):

    distances = np.linalg.norm(Y_pred - Y_true, axis=1)
    exceeds = np.where(distances > epsilon)[0]
    
    if len(exceeds) == 0:
        return len(Y_pred) * dt
    
    t_div = exceeds[0] * dt
    # print(f"Divergence time: {float(t_div)}")
    return t_div

# ── Graph perturbations ───────────────────────────────────────────────────────

def perturb_graph(G, mode, p, rng):
    """
    Return a perturbed copy of G.

    Parameters
    ----------
    G : networkx.Graph
        Original connectome graph.
    mode : str
        One of 'drop_nodes', 'drop_edges', 'swap_edges', 'swap_nodes'.
    p : float
        Perturbation fraction in (0, 1).
    rng : numpy.random.Generator
        Random generator for reproducibility.

    Returns
    -------
    H : networkx.Graph
        Perturbed copy of G.
    """
    H = G.copy()

    if mode == "drop_nodes":
        nodes = list(H.nodes())
        k = max(0, round(p * len(nodes)))
        to_drop = rng.choice(nodes, size=k, replace=False).tolist()
        H.remove_nodes_from(to_drop)

    elif mode == "drop_edges":
        edges = list(H.edges())
        k = max(0, round(p * len(edges)))
        idxs = rng.choice(len(edges), size=k, replace=False)
        H.remove_edges_from([edges[i] for i in idxs])

    elif mode == "swap_edges":
        edges = list(H.edges())
        nodes = list(H.nodes())
        k = max(0, round(p * len(edges)))
        for i in rng.choice(len(edges), size=k, replace=False):
            u, v = edges[i]
            attrs = H[u][v].copy()
            H.remove_edge(u, v)
            new_v = rng.choice(nodes)
            if not H.has_edge(u, new_v) and u != new_v:
                H.add_edge(u, new_v, **attrs)

    elif mode == "swap_nodes":
        nodes = list(H.nodes())
        k = max(0, round(p * len(nodes)) // 2 * 2)  # must be even
        chosen = rng.choice(nodes, size=k, replace=False)
        mapping = dict(zip(chosen, np.roll(chosen, k // 2)))
        H = nx.relabel_nodes(H, mapping)

    else:
        raise ValueError(f"Unknown perturbation mode: {mode!r}")

    return H


# ── Single run on a (possibly perturbed) graph ────────────────────────────────

def run_one(G, seed, J, lambda_reg, t_thermalization, t_training, t_prediction,
            sigma, rho, beta, dt, input_scale):
    """
    Build W from graph G, train the reservoir, run autonomous prediction,
    and return predicted and true Lorenz trajectories.

    Parameters
    ----------
    G : networkx.Graph
        Connectome graph (possibly perturbed).
    seed : int
        Random seed for W_in.
    J, lambda_reg, t_thermalization, t_training, t_prediction,
    sigma, rho, beta, dt, input_scale :
        Reservoir hyperparameters (same meaning as in reservoir.py).

    Returns
    -------
    Y_pred : ndarray, shape (n_steps, 3), or None if graph is degenerate.
    Y_true : ndarray, shape (n_steps, 3), or None if graph is degenerate.
    """
    N = G.number_of_nodes()
    if N == 0:
        return None, None

    ws = round(t_thermalization / dt)
    ts = round(t_training      / dt)
    ns = round(t_prediction    / dt)

    # build W from graph
    node_ids  = sorted(G.nodes())
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    D = np.zeros((N, N))
    for u, v, attrs in G.edges(data=True):
        if u not in id_to_idx or v not in id_to_idx:
            continue
        i, j = id_to_idx[u], id_to_idx[v]
        w = float(attrs.get("number_of_fibers", 1.0))
        D[i, j] = w
        D[j, i] = w

    rho_D = np.max(np.abs(np.linalg.eigvals(D)))
    if rho_D == 0:
        return None, None
    W = (J / rho_D) * D

    # Lorenz driver (normalized)
    lf, ll = lorenz_euler(ws + ts, dt=dt, sigma=sigma, rho=rho, beta=beta)
    u_mean = lf.mean(axis=0)
    u_std  = lf.std(axis=0)
    u_std[u_std == 0] = 1.0
    ln   = (lf - u_mean) / u_std
    u_tr = ln[ws:]

    # input weights
    W_in = input_scale * np.random.default_rng(seed).normal(size=(N, 3))

    # washout
    r = np.zeros(N)
    for t in range(ws):
        r = np.tanh(W @ r + W_in @ ln[t])

    # training
    R = np.empty((ts, N))
    for t in range(ts):
        r = np.tanh(W @ r + W_in @ u_tr[t])
        R[t] = r

    W_out = np.linalg.solve(
        R[:-1].T @ R[:-1] + lambda_reg * np.eye(N),
        R[:-1].T @ u_tr[1:],
    )

    # autonomous prediction
    Y_pred = np.empty((ns, 3))
    for t in range(ns):
        u_hat_norm = r @ W_out
        Y_pred[t]  = u_hat_norm * u_std + u_mean
        r          = np.tanh(W @ r + W_in @ u_hat_norm)

    # ground-truth Lorenz continuing from where training left off
    Y_true, _ = lorenz_euler(ns, dt=dt, sigma=sigma, rho=rho, beta=beta,
                              state0=ll)

    return Y_pred, Y_true


# ── Full experiment ───────────────────────────────────────────────────────────

PERTURBATION_TYPES = ["drop_nodes", "drop_edges", "swap_edges", "swap_nodes"]


def run_experiment(
    file_paths,
    *,
    p_values,
    n_runs=3,
    epsilon=1.0,
    J=0.9,
    lambda_reg=3e-6,
    t_thermalization=30,
    t_training=200,
    t_prediction=50,
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
    dt=0.01,
    input_scale=0.5,
    seed_base=42,
    perturbation_types=None,
):
    """
    E_R [ E_runs [ t_div( f_p(R) ) ] ]

    Outer expectation : over the dataset of connectome files (file_paths).
    Inner expectation : over n_runs independent random seeds per reservoir.

    For every (perturbation_type, p) the function collects one t_div value
    per (file, run) pair; total sample size = len(file_paths) * n_runs.
    """
    if perturbation_types is None:
        perturbation_types = PERTURBATION_TYPES

    results = {
        mode: {p: [] for p in p_values}
        for mode in perturbation_types
    }

    n_files  = len(file_paths)
    n_combos = len(perturbation_types) * len(p_values) * n_files * n_runs
    done     = 0
    t_start  = time.perf_counter()

    for fi, fp in enumerate(file_paths):
        fp = Path(fp)
        G  = nx.read_graphml(fp)

        for mode in perturbation_types:
            for p in p_values:
                for run_idx in range(n_runs):

                    # W_in seed: varies across files and runs, fixed across p/mode
                    seed = seed_base + run_idx * 10_000 + fi

                    # perturbation seed: varies across everything
                    seed_p = (seed_base
                              + int(p * 1e6)
                              + hash(mode) % 10_000
                              + fi * 100
                              + run_idx)
                    rng_p = np.random.default_rng(seed_p)

                    H = perturb_graph(G, mode, p, rng_p)

                    t0 = time.perf_counter()
                    Y_pred, Y_true = run_one(
                        H, seed=seed,
                        J=J, lambda_reg=lambda_reg,
                        t_thermalization=t_thermalization,
                        t_training=t_training,
                        t_prediction=t_prediction,
                        sigma=sigma, rho=rho, beta=beta,
                        dt=dt, input_scale=input_scale,
                    )
                    elapsed = time.perf_counter() - t0

                    td = (0.0 if Y_pred is None
                          else t_div(Y_pred, Y_true, epsilon=epsilon, dt=dt))
                    results[mode][p].append(td)
                    done += 1

                    elapsed_total = time.perf_counter() - t_start
                    eta = elapsed_total / done * (n_combos - done)
                    #print(
                    #    f"[{done:5d}/{n_combos}]"
                    #    f"  file {fi+1:3d}/{n_files}  {fp.stem}"
                    #    f"  {mode:12s}  p={p:.2f}  run={run_idx}"
                    #    f"  t_div={td:.3f}s  job={elapsed:.1f}s"
                    #    f"  ETA={eta/60:.1f}min",
                    #    flush=True,
                    #)

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_experiment(results, p_values, epsilon, n_runs, n_files,
                    t_training, perturbation_types=None, confidence=0.95):
    
    import matplotlib.pyplot as plt
    from scipy import stats

    if perturbation_types is None:
        perturbation_types = PERTURBATION_TYPES

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)

    for ax, mode in zip(axes.flatten(), perturbation_types):
        vals  = [np.array(results[mode][p]) for p in p_values]
        means = np.array([v.mean() for v in vals])
        n_obs = np.array([len(v) for v in vals])

        # t-based confidence intervals
        t_crit = stats.t.ppf((1 + confidence) / 2, df=n_obs - 1)
        sems   = np.array([v.std(ddof=1) / np.sqrt(len(v)) for v in vals])
        ci_lo  = means - t_crit * sems
        ci_hi  = means + t_crit * sems

        # confidence band
        ax.fill_between(p_values, ci_lo, ci_hi,
                        alpha=0.20, color="steelblue",
                        label=f"{int(confidence*100)}% CI")

        # mean line
        ax.plot(p_values, means, marker="o", lw=2,
                color="steelblue", label="mean")

        # baseline
        # ax.axhline(means[0], color="gray", lw=1,
        #            linestyle="--", label="baseline (p=0)")

        ax.set_title(mode.replace("_", " "), fontsize=13, fontweight="bold")
        ax.set_xlabel("perturbation  p")
        ax.set_ylabel(r"$\mathbb{E}[t_{\mathrm{div}}]$ (s)")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        r"$\mathbb{E}_{R}\!\left[\mathbb{E}_{\mathrm{runs}}"
        r"\!\left[t_{\mathrm{div}}(f_p(R))\right]\right]$"
        f" vs perturbation $p$\n"
        f"(ε={epsilon},  n_runs={n_runs},  n_files={n_files},"
        f"  t_train={t_training} s,  {int(confidence*100)}% CI)",
        fontsize=12,
    )
    plt.tight_layout()
    return fig