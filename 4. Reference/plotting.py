import numpy as np
import matplotlib.pyplot as plt

def plot_state_std(R, dt=0.01):
    t = np.arange(len(R)) * dt
    std = R.std(axis=1)
    fig, ax = plt.subplots()
    ax.plot(t, std)
    ax.set(xlabel="t", ylabel="std", title="Reservoir activation std over time")
    plt.show()


def plot_state_sand(R, dt=0.01, n_bins=50):
    t = np.arange(len(R)) * dt
    data_min, data_max = R.min(), R.max()
    edges = np.linspace(data_min, data_max, n_bins + 1)
    H = np.array([np.diff(np.searchsorted(np.sort(row), edges)) for row in R])
    fig, ax = plt.subplots()
    ax.imshow(H.T, origin="lower", aspect="auto",
              extent=[t[0], t[-1], data_min, data_max], interpolation="nearest")
    ax.set(xlabel="t", ylabel="activation", title="Reservoir activation histogram over time")
    plt.show()


def plot_sampled_nodes(R, dt=0.01, n=20, seed=42):
    t = np.arange(len(R)) * dt
    rng = np.random.default_rng(seed)
    idx = rng.choice(R.shape[1], size=min(n, R.shape[1]), replace=False)
    fig, ax = plt.subplots()
    ax.plot(t, R[:, idx], lw=1)
    ax.set(xlabel="t", ylabel="activation", title=f"{len(idx)} sampled reservoir nodes")
    plt.show()


def plot_generated_output(R, W_out, dt=0.01, labels=("x", "y", "z")):
    t = np.arange(len(R)) * dt
    Y = R @ W_out
    fig, ax = plt.subplots()
    for i in range(Y.shape[1]):
        ax.plot(t, Y[:, i], lw=1, label=labels[i])
    ax.set(xlabel="t", ylabel="output", title="Generated output signal")
    ax.legend()
    plt.show()


def plot_lorenz_l1_error(R, W_out, y_true, dt=0.01):
    t = np.arange(len(R)) * dt
    Y_pred = R @ W_out
    l1 = np.abs(y_true - Y_pred).sum(axis=1)
    fig, ax = plt.subplots()
    ax.plot(t, l1, lw=1.5)
    ax.set(xlabel="t", ylabel="L1 error", title="L1 error: true Lorenz vs reservoir prediction")
    plt.show()

def plot_lorenz_3d(R, W_out, y_true):
    Y_pred = R @ W_out
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*y_true.T, lw=0.8, color="black", label="true")
    ax.plot(*Y_pred.T, lw=1.3, color="tomato", linestyle="--", label="predicted")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz attractor: true vs predicted")
    ax.legend()
    plt.show()