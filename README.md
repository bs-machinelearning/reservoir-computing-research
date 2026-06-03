# Reservoir Computing with Human Connectome Networks

This repository contains exploratory notebooks and report material for a group project on reservoir computing with structural human connectome graphs. The main idea is to replace the usual random recurrent reservoir with a graph derived from brain connectivity data, train only a linear readout, and test whether the resulting system can predict short-horizon Lorenz dynamics. The project also studies how prediction quality changes when the connectome is perturbed by deleting or rewiring nodes and edges.

## Repository structure

```text
1. Graph Networks/        Early graph and connectome exploration notebooks
2. Human Data/            Human connectome loading, cleaning, and feature analysis
3. Reservoir Model/       Initial reservoir-computing implementations
4. Reference/             Final/reference reservoir, perturbation, and comparison experiments
img/                      Figures already used by the report
resources/                Project guide and background PDFs
Report.typ                Main Typst report
Meeting Notes.typ         Meeting-notes document
pyproject.toml            Python project metadata and dependencies
uv.lock                   Locked dependency versions
```

## Setup

Install `uv`, then install the Python environment from the repository root:

```bash
uv sync
```

Open notebooks with the environment created by `uv`. In VS Code, select the corresponding Jupyter kernel before running cells.

The notebooks in `2. Human Data/` and `4. Reference/` expect connectome `.graphml` datasets in paths such as `../data/`, `../data/data_234/`, or `../data/repeated_10_scale_250/`. These data files are not bundled in the repository, so place the dataset in the expected location or update the `Path(...)` variables inside the notebooks.

## Main workflow

1. Inspect connectome graphs and their node/edge attributes.
2. Build a reservoir matrix from the weighted connectome adjacency matrix.
3. Drive the reservoir with a Lorenz time series during washout and training.
4. Train the readout using ridge regression.
5. Run autonomous prediction and compare the predicted trajectory with the true Lorenz continuation.
6. Repeat after graph perturbations to measure robustness using divergence time.

## Important files in `4. Reference/`

- `reservoir.py`: connectome-based reservoir implementation.
- `reservoir_with_leak_rate.py`: reservoir implementation with a leak-rate parameter.
- `experiment_functions.py`: perturbation utilities and batch experiment runner.
- `plotting.py`: plotting utilities for Lorenz prediction and reservoir states.
- `main.ipynb`: main connectome reservoir prediction experiment.
- `main_with_leak_rate.ipynb`: leaky reservoir experiment.
- `gaussian_reservoir_implementation.ipynb`: Gaussian random-reservoir baseline.
- `run_experiment.ipynb`: perturbation sweep over node/edge deletion and swapping.
- `experiment_results.png` and `results.npz`: saved edge-deletion experiment results.

## Report

The report is written in Typst:

```bash
typst compile Report.typ Report.pdf
```
