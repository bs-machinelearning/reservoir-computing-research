import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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
    ax.plot(*y_true.T, lw=0.2, color="black", label="true")
    ax.plot(*Y_pred.T, lw=1.8, color="tomato", linestyle="--", label="pred")
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Lorenz attractor: true vs predicted")
    ax.legend()
    plt.show()
    
def plot_lorenz_components(Y_pred, Y_true, dt=0.01, labels=("x", "y", "z")):
    t = np.arange(len(Y_pred)) * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(t, Y_true[:, i], label=f"true {labels[i]}", lw=1.5)
        ax.plot(t, Y_pred[:, i], "--", label=f"pred {labels[i]}", lw=1.2)
        ax.set_ylabel(labels[i])
        ax.legend()

    axes[-1].set_xlabel("t")
    fig.suptitle("Lorenz components: true vs predicted")
    plt.tight_layout()
    plt.show()
    
def plot_lorenz_3d_pred_vs_true(Y_pred, Y_true):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*Y_true.T, lw=0.7, color="black", label="true")
    ax.plot(*Y_pred.T, lw=1.2, linestyle="--", color="tomato", label="pred")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz attractor: true vs predicted")
    ax.legend()
    plt.show()
    

def plot_lorenz_3d_interactive(Y_pred, Y_true):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=Y_true[:, 0],
        y=Y_true[:, 1],
        z=Y_true[:, 2],
        mode="lines",
        name="true",
        line=dict(width=3, color="black"),
    ))

    fig.add_trace(go.Scatter3d(
        x=Y_pred[:, 0],
        y=Y_pred[:, 1],
        z=Y_pred[:, 2],
        mode="lines",
        name="pred",
        line=dict(width=4, color="tomato"),   # solid + thinner
    ))

    fig.update_layout(
        title="Lorenz attractor: true vs predicted",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        width=900,
        height=700,
    )

    fig.show()