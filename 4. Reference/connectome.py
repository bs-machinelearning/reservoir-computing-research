import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import networkx as nx


def data_matrix(file_path, folder="../data/repeated_10_scale_250/", key="number_of_fibers"):
    folder = Path(folder)
    G = nx.read_graphml(folder / file_path)
    ids = sorted(G.nodes())
    k = {u: i for i, u in enumerate(ids)}
    D = np.zeros((len(ids), len(ids)))
    for u, v, a in G.edges(data=True):
        w = float(a.get(key, 0.0))
        i, j = k[u], k[v]
        D[i, j] = D[j, i] = w
    return D


def _time_to_steps(duration, dt, name="duration"):
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if duration < 0:
        raise ValueError(f"{name} must be >= 0")
    return int(round(duration / dt))


def _time_axis(samples, t):
    T = np.asarray(samples).shape[0]
    if np.ndim(t) == 0:
        return np.linspace(0.0, float(t), T, endpoint=False)
    tt = np.asarray(t)
    if tt.shape[0] != T:
        raise ValueError(f"time vector has length {tt.shape[0]}, expected {T}")
    return tt


def lorenz_trajectory(
    duration=None,
    *,
    n_steps=None,
    dt=0.01,
    x0=(1.0, 1.0, 1.0),
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
):
    if n_steps is None:
        if duration is None:
            raise ValueError("provide either duration or n_steps")
        n_steps = _time_to_steps(duration, dt, "duration")

    x = np.array(x0, dtype=float)
    X = np.empty((n_steps, 3), dtype=float)

    for i in range(n_steps):
        X[i] = x
        x = x + dt * np.array([
            sigma * (x[1] - x[0]),
            x[0] * (rho - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ])

    return X, x

def setup_reservoir(
    file_path,
    J=0.9,
    lambda_reg=1e-5,
    washout_time=10.0,
    train_time=200.0,
    sigma=10.0,
    rho=28.0,
    beta=8 / 3,
    dt=0.01,
    seed=42,
    input_scale=0.03,
):
    D = data_matrix(file_path).astype(float)

    rho_D = np.max(np.abs(np.linalg.eigvals(D)))
    if rho_D == 0:
        raise ValueError("connectivity matrix has zero spectral radius")

    W = (J / rho_D) * D

    washout_steps = _time_to_steps(washout_time, dt, "washout_time")
    train_steps = _time_to_steps(train_time, dt, "train_time")

    total_steps = washout_steps + train_steps + 1
    U_raw, _ = lorenz_trajectory(
        n_steps=total_steps,
        dt=dt,
        x0=(1.0, 1.0, 1.0),
        sigma=sigma,
        rho=rho,
        beta=beta,
    )

    U_mean = U_raw.mean(axis=0)
    U_std = U_raw.std(axis=0)
    U_std[U_std == 0.0] = 1.0
    U = (U_raw - U_mean) / U_std

    rng = np.random.default_rng(seed)
    W_in = input_scale * rng.normal(size=(len(D), 3))

    r = np.zeros(len(D), dtype=float)

    for u in U[:washout_steps]:
        r = np.tanh(W @ r + W_in @ u)

    R = np.empty((train_steps, len(D)), dtype=float)
    for t in range(train_steps):
        u = U[washout_steps + t]
        r = np.tanh(W @ r + W_in @ u)
        R[t] = r

    Y = U[washout_steps + 1 : washout_steps + train_steps + 1]
    W_out = np.linalg.solve(
        R.T @ R + lambda_reg * np.eye(len(D)),
        R.T @ Y,
    )

    # This is the true Lorenz state the first autonomous prediction should match.
    lorenz_start = U_raw[washout_steps + train_steps].copy()
    lorenz_cfg = {
        "dt": dt,
        "sigma": sigma,
        "rho": rho,
        "beta": beta,
    }

    return W, W_in, W_out, r, U_mean, U_std, lorenz_start, lorenz_cfg

def run_reservoir(W, W_in, W_out, r, duration=20.0, dt=0.01):
    n_steps = _time_to_steps(duration, dt, "duration")

    R = np.empty((n_steps, len(r)), dtype=float)
    Y = np.empty((n_steps, W_out.shape[1]), dtype=float)

    for t in range(n_steps):
        u = r @ W_out
        r = np.tanh(W @ r + W_in @ u)
        R[t] = r
        Y[t] = u

    return R, Y


def plot_state_std(samples, t, ax=None):
    """
    Plot the standard deviation of reservoir activations over time.

    Parameters:
        samples (array-like): Reservoir states, shape (T, N).
        t (float or array-like): Total duration or explicit time vector.
        ax (matplotlib.axes.Axes, optional): Existing axes.

    Returns:
        tuple:
            - std (np.ndarray): Standard deviation at each time step, shape (T,)
            - ax (matplotlib.axes.Axes): Axes used
    """
    X = np.asarray(samples)
    tt = _time_axis(X, t)

    std = np.std(X, axis=1)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(tt, std)

    ymin = min(0.0, std.min())
    ymax = max(0.0, std.max())
    if ymin == ymax:
        ymax = ymin + 1e-12

    ax.set_ylim(ymin, ymax)
    ax.set(
        xlabel="t",
        ylabel="standard deviation",
        title="Reservoir activation standard deviation over time",
    )
    return std, ax

def plot_state_sand(samples, t, n_bins=50, ax=None):
    """
    Plot a sand/histogram image of reservoir activations over time.

    Parameters:
        samples (array-like): Reservoir states, shape (T, N).
        t (float or array-like): Total duration or explicit time vector.
        n_bins (int): Number of activation bins.
        ax (matplotlib.axes.Axes, optional): Existing axes.

    Returns:
        tuple:
            - H (np.ndarray): Histogram counts, shape (T, n_bins)
            - ax (matplotlib.axes.Axes): Axes used
    """
    X = np.asarray(samples)
    T = X.shape[0]
    tt = _time_axis(X, t)

    data_min = np.min(X)
    data_max = np.max(X)

    # avoid zero-width bins when all values are identical
    if data_min == data_max:
        eps = 1e-12 if data_min == 0 else abs(data_min) * 1e-12
        data_min -= eps
        data_max += eps

    edges = np.linspace(data_min, data_max, n_bins + 1)
    H = np.empty((T, n_bins), dtype=int)

    for i, s in enumerate(np.sort(X, axis=1)):
        H[i] = np.diff(np.searchsorted(s, edges))

    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[tt[0], tt[-1], data_min, data_max],
        interpolation="nearest",
    )
    ax.set(
        xlabel="t",
        ylabel="activation",
        title="Reservoir activation histogram over time",
    )
    plt.colorbar(im, ax=ax, label="count")
    return H, ax


def plot_sampled_nodes(samples, t, n=20, rng=None, ax=None):
    """
    Plot time series for a random subset of reservoir nodes.

    Parameters:
        samples (array-like): Reservoir states, shape (T, N).
        t (float or array-like): Total duration or explicit time vector.
        n (int): Number of nodes to sample.
        rng (int or np.random.Generator, optional): Random seed or generator.
        ax (matplotlib.axes.Axes, optional): Existing axes.

    Returns:
        tuple:
            - idx (np.ndarray): Chosen node indices
            - ax (matplotlib.axes.Axes): Axes used
    """
    X = np.asarray(samples)
    T, N = X.shape
    tt = _time_axis(X, t)

    if isinstance(rng, np.random.Generator):
        generator = rng
    else:
        generator = np.random.default_rng(rng)

    idx = generator.choice(N, size=min(n, N), replace=False)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(tt, X[:, idx], lw=1)
    ax.set(
        xlabel="t",
        ylabel="activation",
        title=f"{len(idx)} sampled reservoir nodes",
    )
    return idx, ax


def plot_generated_output(samples, W_out, t, labels=("x", "y", "z"), ax=None):
    """
    Plot the generated 3D output signal reconstructed from reservoir states.

    Parameters:
        samples (array-like): Reservoir states, shape (T, N).
        W_out (np.ndarray): Readout matrix, shape (N, 3).
        t (float or array-like): Total duration or explicit time vector.
        labels (tuple): Labels for output dimensions.
        ax (matplotlib.axes.Axes, optional): Existing axes.

    Returns:
        tuple:
            - Y (np.ndarray): Generated output, shape (T, 3)
            - ax (matplotlib.axes.Axes): Axes used
    """
    X = np.asarray(samples)
    tt = _time_axis(X, t)
    Y = X @ W_out

    if ax is None:
        _, ax = plt.subplots()

    for i in range(Y.shape[1]):
        label = labels[i] if i < len(labels) else f"dim {i}"
        ax.plot(tt, Y[:, i], lw=1, label=label)

    ax.set(xlabel="t", ylabel="output", title="Generated output signal")
    ax.legend()
    return Y, ax


def plot_lorenz_l1_error(y_pred, y_true, t, per_dimension=False, labels=("x", "y", "z"), ax=None):
    Y_pred = np.asarray(y_pred)
    Y_true = np.asarray(y_true)

    if Y_pred.shape != Y_true.shape:
        raise ValueError(f"shape mismatch: predicted {Y_pred.shape}, true {Y_true.shape}")

    tt = _time_axis(Y_pred, t)
    abs_err = np.abs(Y_true - Y_pred)
    l1 = np.sum(abs_err, axis=1)

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(tt, l1, lw=1.5, label="total L1 error")

    if per_dimension:
        for i in range(abs_err.shape[1]):
            label = labels[i] if i < len(labels) else f"dim {i}"
            ax.plot(tt, abs_err[:, i], lw=1.0, alpha=0.8, label=f"|{label} error|")
        ax.legend()

    ax.set(
        xlabel="t",
        ylabel="L1 error",
        title="L1 error: true Lorenz vs reservoir prediction",
    )
    return l1, ax