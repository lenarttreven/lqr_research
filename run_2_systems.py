"""Run, time, and plot the aircraft-pitch and UAV-2D benchmarks.

The per-system experiment settings are fixed to the currently used values:

* aircraft_pitch: lam=20, perturbation=0.01
* uav_2d: lam=5, perturbation=0.1

IR-LQR's g1 and g2 are loaded from each system's tuning CSV by the same
resolver used by ``run_benchmark.py``. No tuning is performed here.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

import plot_2_systems
from run_benchmark import resolve_irlqr_parameters, run_on_system, save_data
from run_timing import time_on_system
from systems import get_physical_systems, get_stabl_systems


SYSTEM_SPECS = (
    {
        "name": "aircraft_pitch",
        "label": r"\textbf{Aircraft pitch}",
        "suite": "physical",
        "suite_fn": get_physical_systems,
        "lam": 20.0,
        "perturbation": 0.01,
    },
    {
        "name": "uav_2d",
        "label": r"\textbf{UAV 2D}",
        "suite": "stabl",
        "suite_fn": get_stabl_systems,
        "lam": 5.0,
        "perturbation": 0.1,
    },
)


def select_named_system(spec: dict, key: jax.Array):
    systems = spec["suite_fn"](key, perturbation=spec["perturbation"])
    return next(system for system in systems if system.name == spec["name"])


def save_timing_data(path: Path, args: argparse.Namespace, system_name: str, results):
    serializable = {
        system_name: {
            algo_name: {
                key: np.asarray(value) if hasattr(value, "shape") else value
                for key, value in algo_results.items()
            }
            for algo_name, algo_results in results.items()
        }
    }
    payload = {"args": vars(args), "results": serializable}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(payload, file)
    print(f"  Saved timing results to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--timing-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--solver", choices=["schur", "sda", "riccati"], default="sda")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-path", type=Path, default=Path("cdc_2_systems.pdf"))
    parser.add_argument(
        "--no-show", action="store_true", help="Save the combined plot without opening it."
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for spec in SYSTEM_SPECS:
        print(f"\n{'=' * 60}")
        print(
            f"{spec['name']}: lam={spec['lam']:g}, "
            f"perturbation={spec['perturbation']:g}"
        )
        print(f"{'=' * 60}")

        root_key = jax.random.PRNGKey(args.seed)
        run_key, system_key = jax.random.split(root_key)
        system = select_named_system(spec, system_key)
        run_key, benchmark_key = jax.random.split(run_key)
        g1, g2, source = resolve_irlqr_parameters(system.name, None, None)
        print(f"  IR-LQR parameters: g1={g1:g}, g2={g2:g} ({source})")

        benchmark_results = run_on_system(
            system,
            args.num_trials,
            args.num_steps,
            benchmark_key,
            lam=spec["lam"],
            irlqr_g1=g1,
            irlqr_g2=g2,
            oslo_mu=0.1,
            oslo_beta=1.0,
            laglq_beta=0.05,
            laglq_eps=1e-2,
            laglq_max_dual_iters=50,
            solver=args.solver,
        )
        save_data(benchmark_results, system.name, str(args.output_dir))

        timing_results = time_on_system(
            system,
            args.timing_steps,
            benchmark_key,
            lam=spec["lam"],
            irlqr_g1=g1,
            irlqr_g2=g2,
            oslo_mu=0.1,
            oslo_beta=1.0,
            laglq_beta=0.05,
            laglq_eps=1e-2,
            laglq_max_dual_iters=50,
        )
        timing_path = args.output_dir / f"timing_{spec['suite']}.pkl"
        save_timing_data(timing_path, args, system.name, timing_results)
    plot_2_systems.plot_saved_results(
        results_dir=args.output_dir,
        save_path=args.figure_path,
        show=not args.no_show,
    )
    print(f"\nCombined figure saved to {args.figure_path}")


if __name__ == "__main__":
    main()
