"""Tune IR-LQR's g1 and g2 using trajectory cross-validation.

Each grid point is evaluated on the same random trajectories. Configurations
are ranked lexicographically: first by the total number of fallback uses, then
by the mean fold-level median final cumulative regret. This makes avoiding the
fallback controller the primary tuning objective. The full table is saved as
CSV, and cumulative regret is plotted for the best configurations.

Example using every configurable argument:
    python tune_irlqr.py \
        --suite stabl \
        --system uav_2d \
        --g1-values 0.00001 0.0001 0.001 0.01 \
        --g2-values 0.00001 0.0001 0.001 0.01 \
        --num-trials 40 \
        --num-folds 5 \
        --num-steps 200 \
        --seed 0 \
        --lam 5 \
        --perturbation 0.01 \
        --solver sda \
        --plot-top-k 3 \
        --plot-config 0 1 \
        --plot-output figures/uav_2d_irlqr_tuning.pdf \
        --output results/irlqr_cv/uav_2d.csv
"""

import argparse
import csv
import math
import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from algorithms.irlqr import IRLQR
from simulation import simulate_many
from systems import (
    LQRSystem,
    get_benchmark_systems,
    get_integrator_systems,
    get_physical_systems,
    get_stabl_systems,
)


SUITES = {
    "all": get_benchmark_systems,
    "physical": get_physical_systems,
    "integrator": get_integrator_systems,
    "stabl": get_stabl_systems,
}


def select_system(
    suite: str, selector: str, key: jax.Array, perturbation: float
) -> LQRSystem:
    """Select a system by its name or zero-based index within a suite."""
    systems = SUITES[suite](key, perturbation=perturbation)
    try:
        index = int(selector)
    except ValueError:
        matches = [system for system in systems if system.name == selector]
        if not matches:
            available = ", ".join(system.name for system in systems)
            raise ValueError(
                f"unknown system {selector!r} in suite {suite!r}; "
                f"available systems: {available}"
            )
        return matches[0]

    if index < 0 or index >= len(systems):
        raise ValueError(
            f"system index {index} is outside the range 0..{len(systems) - 1} "
            f"for suite {suite!r}"
        )
    return systems[index]


def cross_validation_scores(
    cumulative_regrets: np.ndarray, num_folds: int
) -> tuple[np.ndarray, float]:
    """Return fold medians and their mean for one hyperparameter pair."""
    folds = np.array_split(cumulative_regrets, num_folds)
    fold_scores = np.asarray([np.median(fold) for fold in folds])
    return fold_scores, float(np.mean(fold_scores))


def tune(
    system: LQRSystem,
    g1_values: list[float],
    g2_values: list[float],
    num_trials: int,
    num_folds: int,
    num_steps: int,
    seed: int,
    lam: float,
    solver: str,
    keep_regrets: bool = True,
) -> tuple[list[dict], dict[tuple[float, float], np.ndarray]]:
    """Evaluate the grid and return sorted scores and per-step regrets."""
    if num_trials < num_folds:
        raise ValueError("num_trials must be at least num_folds")
    if not g1_values or not g2_values:
        raise ValueError("g1_values and g2_values must be non-empty")

    trial_key = jax.random.PRNGKey(seed)
    trial_keys = jax.random.split(trial_key, num_trials)

    rows = []
    regrets_by_config = {}
    total = len(g1_values) * len(g2_values)
    for index, (g1, g2) in enumerate(
        ((g1, g2) for g1 in g1_values for g2 in g2_values), start=1
    ):
        print(f"[{index:>3}/{total}] g1={g1:g}, g2={g2:g}", flush=True)
        algo = IRLQR(
            lam=lam,
            g1=g1,
            g2=g2,
            A0=system.A0,
            B0=system.B0,
            solver=solver,
        )
        results = simulate_many(
            algo,
            system.A_star,
            system.B_star,
            system.Q,
            system.R,
            trial_keys,
            num_steps,
            system.noise_sigma,
            system.x0,
            solver=solver,
        )
        cumulative_regrets = np.asarray(jnp.sum(results["regrets"], axis=1))
        fallback_uses = np.asarray(results["cum_fallback_uses"][:, -1])
        if keep_regrets:
            regrets_by_config[(g1, g2)] = np.asarray(results["regrets"])
        finite = np.isfinite(cumulative_regrets)
        if finite.all():
            fold_scores, cv_score = cross_validation_scores(
                cumulative_regrets, num_folds
            )
        else:
            fold_scores = np.full(num_folds, np.inf)
            cv_score = math.inf

        row = {
            "g1": g1,
            "g2": g2,
            "cv_score": cv_score,
            "median_regret": float(np.median(cumulative_regrets)),
            "mean_regret": float(np.mean(cumulative_regrets)),
            "num_finite": int(finite.sum()),
            "total_fallback_uses": int(np.sum(fallback_uses)),
            "trials_with_fallback": int(np.count_nonzero(fallback_uses)),
            "median_fallback_uses": float(np.median(fallback_uses)),
            "max_fallback_uses": int(np.max(fallback_uses)),
        }
        row.update(
            {f"fold_{fold + 1}": float(score) for fold, score in enumerate(fold_scores)}
        )
        rows.append(row)
        print(
            f"          fallback uses: {row['total_fallback_uses']} total "
            f"across {row['trials_with_fallback']}/{num_trials} trials; "
            f"CV score: {cv_score:.6g}",
            flush=True,
        )

    return sorted(
        rows,
        key=lambda row: (row["total_fallback_uses"], row["cv_score"]),
    ), regrets_by_config


def select_plot_configs(
    rows: list[dict],
    top_k: int,
    requested: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Select ranked and explicitly requested configurations without duplicates."""
    if top_k < 0:
        raise ValueError("plot_top_k must be non-negative")

    available = {(row["g1"], row["g2"]) for row in rows}
    missing = [config for config in requested if config not in available]
    if missing:
        formatted = ", ".join(f"(g1={g1:g}, g2={g2:g})" for g1, g2 in missing)
        raise ValueError(f"plot configurations are not in the tuning grid: {formatted}")

    selected = [(row["g1"], row["g2"]) for row in rows[:top_k]]
    for config in requested:
        if config not in selected:
            selected.append(config)
    return selected


def plot_cumulative_regret(
    rows: list[dict],
    regrets_by_config: dict[tuple[float, float], np.ndarray],
    configs: list[tuple[float, float]],
    title: str,
    output: str | None = None,
) -> None:
    """Plot median cumulative regret and its 20--80% trial interval."""
    if not configs:
        return

    rank = {(row["g1"], row["g2"]): i + 1 for i, row in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for config in configs:
        regrets = regrets_by_config[config]
        cumulative = np.cumsum(regrets, axis=1)
        q20, q50, q80 = np.quantile(cumulative, [0.2, 0.5, 0.8], axis=0)
        g1, g2 = config
        label = f"#{rank[config]}: g1={g1:g}, g2={g2:g}"
        (line,) = ax.plot(np.arange(1, cumulative.shape[1] + 1), q50, label=label)
        ax.fill_between(
            np.arange(1, cumulative.shape[1] + 1),
            q20,
            q80,
            color=line.get_color(),
            alpha=0.15,
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative regret")
    ax.set_title(title)
    ax.grid()
    ax.legend()
    fig.tight_layout()
    if output:
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def save_rows(rows: list[dict], output: str) -> None:
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite",
        choices=list(SUITES),
        default="stabl",
        help="Benchmark suite containing the system. Default: stabl.",
    )
    parser.add_argument(
        "--system",
        default="uav_2d",
        help="System name or zero-based index within the suite. Default: uav_2d.",
    )
    parser.add_argument(
        "--g1-values", type=float, nargs="+", default=[0.00001, 0.0001, 0.001, 0.01]
    )
    parser.add_argument(
        "--g2-values", type=float, nargs="+", default=[0.00001, 0.0001, 0.001, 0.01]
    )
    parser.add_argument("--num-trials", type=int, default=40)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lam", type=float, default=5.0)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument(
        "--solver", choices=["schur", "sda", "riccati"], default="sda"
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. By default, results are saved separately as "
            "results/irlqr_cv/<system>.csv."
        ),
    )
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=3,
        help="Plot the top K configurations by fallback uses, then CV score. Default: 3.",
    )
    parser.add_argument(
        "--plot-config",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("G1", "G2"),
        help=(
            "Also plot this exact (g1, g2) pair. May be repeated; the pair "
            "must be in the tuning grid."
        ),
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Save the regret plot here instead of showing it interactively.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not create a cumulative-regret plot.",
    )
    args = parser.parse_args()

    requested = [tuple(config) for config in args.plot_config]
    if args.plot_top_k < 0:
        parser.error("--plot-top-k must be non-negative")
    grid = {(g1, g2) for g1 in args.g1_values for g2 in args.g2_values}
    missing = [config for config in requested if config not in grid]
    if missing:
        formatted = ", ".join(f"(g1={g1:g}, g2={g2:g})" for g1, g2 in missing)
        parser.error(f"--plot-config values are not in the tuning grid: {formatted}")

    root_key = jax.random.PRNGKey(args.seed)
    system_key, _ = jax.random.split(root_key)
    system = select_system(
        args.suite, args.system, system_key, perturbation=args.perturbation
    )
    safe_name = system.name.replace(" ", "_").replace("/", "_")
    output = args.output or os.path.join("results", "irlqr_cv", f"{safe_name}.csv")

    print(f"Tuning IR-LQR on {system.name} (suite={args.suite})")
    rows, regrets_by_config = tune(
        system,
        args.g1_values,
        args.g2_values,
        args.num_trials,
        args.num_folds,
        args.num_steps,
        args.seed,
        args.lam,
        args.solver,
        keep_regrets=not args.no_plot,
    )
    save_rows(rows, output)

    best = rows[0]
    print("\nBest hyperparameters")
    print(f"  g1={best['g1']:g}")
    print(f"  g2={best['g2']:g}")
    print(f"  total fallback uses={best['total_fallback_uses']}")
    print(f"  trials with fallback={best['trials_with_fallback']}/{args.num_trials}")
    print(f"  CV score={best['cv_score']:.6g}")
    print(f"Full results saved to {output}")

    if not args.no_plot:
        configs = select_plot_configs(rows, args.plot_top_k, requested)
        plot_cumulative_regret(
            rows,
            regrets_by_config,
            configs,
            title=f"IR-LQR tuning: {system.name}",
            output=args.plot_output,
        )
        if args.plot_output and configs:
            print(f"Cumulative-regret plot saved to {args.plot_output}")


if __name__ == "__main__":
    main()
