# LQR Research

This repository contains JAX implementations of several adaptive linear-quadratic regulator (LQR) algorithms together with a reusable benchmark suite.

Implemented algorithms:

- Intrinsic Reward LQR (IR-LQR, ours)
- Thompson Sampling (TS, https://arxiv.org/pdf/1703.08972)
- Certainty Equivalent Control with Persistent Exication (CEC+PE, https://arxiv.org/pdf/2001.09576)
- Optimistic LQR via Lagrangian Relaxation (LagLQ, https://arxiv.org/pdf/2007.06482)
- Optimistic Semidefinite Programming for LQ Control (OSLO, https://proceedings.mlr.press/v97/cohen19b/cohen19b.pdf)

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

Either run commands through `uv run`, as in the paper-reproduction commands
below, or activate the virtual environment once per shell session:

```bash
source .venv/bin/activate
```

## Reproducing the Paper Results

The repository includes the saved benchmark and controller-timing results used
by the combined aircraft-pitch and UAV-2D plot. To reproduce the paper figure
from those existing results, run:

```bash
uv run python plot_2_systems.py --no-show
```

`plot_2_systems.py` only loads the existing files under `results/`; it does not
run simulations or timing benchmarks. It writes the combined figure to
`cdc_2_systems.pdf`. Omit `--no-show` to also display the figure interactively.

To reproduce both the simulations and the figure from scratch, run:

```bash
uv run python run_2_systems.py --no-show
```

`run_2_systems.py` performs the benchmark simulations, measures controller
update times, saves the new results under `results/`, and then calls the same
plotting functionality as `plot_2_systems.py`. It uses:

- aircraft pitch: `lam=20`, `perturbation=0.01`
- UAV 2D: `lam=5`, `perturbation=0.1`
- 40 trials, 200 simulation steps, and 200 timing steps by default
- the existing per-system IR-LQR tunings in `results/irlqr_cv/`

Once `run_2_systems.py` has completed at least once, either command produces
the plot from the same saved inputs. Custom locations can be selected with
`--output-dir` and `--figure-path` for `run_2_systems.py`, or `--results-dir`
and `--figure-path` for `plot_2_systems.py`.

## Benchmark Suite and Other Usage

The benchmark suite includes:

- 7 classic physical systems
- 10 discretized integrator-chain systems

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

Tune the IR-LQR optimism parameters on the UAV 2-D system with a coarse,
five-fold trajectory cross-validation grid:

```bash
python tune_irlqr.py \
  --suite stabl --system uav_2d \
  --g1-values 0.00001 0.0001 0.001 0.01 \
  --g2-values 0.00001 0.0001 0.001 0.01 \
  --num-trials 40 --num-folds 5 --num-steps 200 \
  --perturbation 0.01
```

Every grid point uses the same system prior and process-noise trajectories.
Configurations are ranked first by total fallback-controller uses and then by
cross-validation regret, so a configuration that never falls back is preferred
whenever one exists.
The system can be selected by name or by its zero-based index within the
chosen suite, for example `--suite physical --system cart_pole` or
`--suite integrator --system 3`. The command prints the best pair and saves
each system's fold scores separately under `results/irlqr_cv/<system>.csv`.
Use `--output` to override that path. By default, the benchmark loads each
system's selected `g1` and `g2` from its CSV in `results/irlqr_cv/`. Explicit
`--irlqr-g1` and `--irlqr-g2` values override the tuned values. If a system has
no tuning CSV, the benchmark falls back to the hardcoded values `g1=0` and
`g2=1`.

After tuning, the command also plots the median cumulative regret (with a
20th--80th percentile band) for the three best configurations. Choose how many
ranked configurations to include with `--plot-top-k`, and add exact grid points
by repeating `--plot-config G1 G2`:

```bash
python tune_irlqr.py \
  --suite stabl --system uav_2d \
  --g1-values 0.00001 0.0001 0.001 0.01 \
  --g2-values 0.00001 0.0001 0.001 0.01 \
  --plot-top-k 1 --plot-config 0.00001 0.01 --plot-config 0.01 0.00001 \
  --plot-output figures/uav_2d_irlqr_tuning.pdf
```

Without `--plot-output`, the figure is shown interactively. Use `--no-plot` to
skip it. Every plotted curve uses the same trials already evaluated during
tuning, so plotting does not rerun the experiment.

Save results to a custom directory:

```bash
python run_benchmark.py --output-dir results/my_experiment
python plot_benchmark.py --input-dir results/my_experiment
```

Tune selected algorithm hyperparameters:

```bash
python run_benchmark.py \
  --lam 0.1 \
  --irlqr-g1 0.0 \
  --irlqr-g2 1.0 \
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
algo = IRLQR(lam=0.1, g1=0, g2=0.05, A0=system.A0, B0=system.B0)
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
- `run_2_systems.py`: run, time, and plot the aircraft and UAV examples
- `systems.py`: benchmark-system definitions
- `simulation.py`: simulation and plotting utilities
- `algorithms/`: algorithm implementations

## Hyperparameters

### Aircraft Pitch

Run command: `python run_benchmark.py --suite physical --systems 4 --num-trials 40 --num-steps 200 --lam 20 --perturbation 0.01 --solver sda`

| Algorithm | Key hyperparameters |
|-----------|---------------------|
| IR-LQR    | `g1=0.01`, `g2=0.001` (loaded from tuning CSV) |
| TS        | `beta=1e-3` |
| CEC+PE    | `init_act_std=0.1` |
| LagLQ     | `beta=1e-3`, `penalty_aux=1e4`, `solver="sda"` |
| OSLO      | `mu=1e-3`, `beta=1.0` |

All algorithms share `lam=20` and are initialized with `A0=system.A0`, `B0=system.B0`. OSLO additionally requires `sigma=system.noise_sigma`.

### UAV 2

Run command: `python run_benchmark.py --suite stabl --systems 2 --num-trials 40 --num-steps 200 --lam 5 --perturbation 0.1 --solver sda`

| Algorithm | Key hyperparameters |
|-----------|---------------------|
| IR-LQR    | `g1=0.01`, `g2=0.01` (loaded from tuning CSV) |
| TS        | `beta=1e-3` |
| CEC+PE    | `init_act_std=0.1` |
| LagLQ     | `beta=1e-3`, `penalty_aux=1e4`, `solver="sda"` |
| OSLO      | `mu=1e-3`, `beta=1.0` |

All algorithms share `lam=5` and are initialized with `A0=system.A0`, `B0=system.B0`. OSLO additionally requires `sigma=system.noise_sigma`.

## Notes

- All algorithms except OSLO are fully JIT-compiled with JAX. OSLO requires solving a semidefinite program (SDP) via CVXPY at each step, which makes it considerably slower than the other methods.
- LAGLQ additionally requires solving several discrete algebraic Riccati equations (DAREs) per controller update, so it is also considerably slower than the remaining JIT-compiled methods.
- `plot_benchmark.py` opens plots with Matplotlib via `plt.show()`, so run it in an environment with a display or use `--save-dir` to write figures to disk without a display.
