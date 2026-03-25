# LQR Research

This repository contains JAX implementations of several adaptive linear-quadratic regulator (LQR) algorithms together with a reusable benchmark suite.

Implemented algorithms:

- IR-LQR
- Thompson Sampling
- CEC
- CEC + PE
- LAGLQ
- OSLO

The benchmark suite includes:

- 7 classic physical systems
- 10 discretized integrator-chain systems

## Requirements

- Python 3.10+
- `uv`

The default installation uses the standard `jax` package from PyPI. If you need a platform-specific accelerator build of JAX, adjust the dependency resolution accordingly before syncing the project environment.

## Installation

Create the project environment with `uv`:

```bash
uv sync --python 3.11
```

This will:

- create `.venv`
- install the local project in editable mode
- JAX and NumPy for simulation and linear algebra
- Jaxtyping for type annotations
- Matplotlib for plotting
- CVXPY for the OSLO algorithm

If you do not already have a suitable Python installed, `uv` can install one for you:

```bash
uv python install 3.11
uv sync --python 3.11
```

## Usage

First, activate the virtual environment:

```bash
source .venv/bin/activate
```

### Run the small example experiment

```bash
python run_experiment.py
```

This runs a small 3-state example and opens a cumulative-regret plot.

### Run the full benchmark suite

Running and plotting are separate steps. First run the experiments to save results to disk, then plot from the saved data. This lets you re-plot with different settings without re-running experiments.

**Step 1: Run experiments** (saves `.npz` files to `results/`):

```bash
python run_benchmark.py
```

By default this runs all 17 benchmark systems with:

- `100` trials per system
- `1000` time steps per trial

Results are saved as `.npz` files in the `results/` directory (one file per system).

**Step 2: Plot results**:

```bash
# Plot all saved results interactively
python plot_benchmark.py

# Save figures to disk
python plot_benchmark.py --save-dir figures

# Plot a single system
python plot_benchmark.py --files results/Pendulum.npz

# Use linear y-axis
python plot_benchmark.py --no-log-y
```

### Common benchmark commands

Run only the physical systems:

```bash
python run_benchmark.py --suite physical
```

Run only integrator-chain systems:

```bash
python run_benchmark.py --suite integrator
```

Run selected systems by index:

```bash
python run_benchmark.py --systems 0 3 9
```

Change the number of trials and steps:

```bash
python run_benchmark.py --num-trials 50 --num-steps 500
```

Set the random seed:

```bash
python run_benchmark.py --seed 42
```

Save results to a custom directory:

```bash
python run_benchmark.py --output-dir results/my_experiment
python plot_benchmark.py --input-dir results/my_experiment
```

Tune selected algorithm hyperparameters:

```bash
python run_benchmark.py \
  --lam 0.1 \
  --oslo-mu 0.1 \
  --oslo-beta 1.0 \
  --laglq-delta 0.05
```

### Benchmark controller computation time

```bash
python run_timing.py
```

This measures the wall-clock time of each controller recomputation (only steps where K actually changes). Accepts the same `--suite`, `--systems`, and `--num-steps` flags as `run_benchmark.py`:

```bash
python run_timing.py --suite physical --num-steps 500
python run_timing.py --systems 0 3
```

## Using the Code as a Library

The code can also be imported directly from Python modules:

```python
import jax

from algorithms.irlqr import IRLQR
from simulation import simulate_many
from systems import get_physical_systems

key = jax.random.PRNGKey(0)
system = get_physical_systems(key)[0]
algo = IRLQR(lam=0.1, beta=0.05, A0=system.A0, B0=system.B0)
keys = jax.random.split(key, 5)

results = simulate_many(
    algo,
    system.A_star,
    system.B_star,
    system.Q,
    system.R,
    keys,
    num_steps=100,
    noise_sigma=system.noise_sigma,
    x0=system.x0,
)
```

`results["regrets"]` contains one regret trajectory per trial.

## Repository Layout

- `run_experiment.py`: small example experiment
- `run_benchmark.py`: run benchmark experiments and save results to disk
- `plot_benchmark.py`: load saved results and plot them
- `run_timing.py`: controller computation time benchmark
- `systems.py`: benchmark-system definitions
- `simulation.py`: simulation and plotting utilities
- `algorithms/`: algorithm implementations

## Notes

- All algorithms except OSLO are fully JIT-compiled with JAX. OSLO requires solving a semidefinite program (SDP) via CVXPY at each step, which makes it considerably slower than the other methods.
- LAGLQ additionally requires solving several discrete algebraic Riccati equations (DAREs) per controller update, so it is also considerably slower than the remaining JIT-compiled methods.
- `plot_benchmark.py` opens plots with Matplotlib via `plt.show()`, so run it in an environment with a display or use `--save-dir` to write figures to disk without a display.
