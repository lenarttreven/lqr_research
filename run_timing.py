"""Benchmark controller computation time for all LQR algorithms.

Runs each algorithm in a Python for-loop (no jax.lax.scan) and measures
the wall-clock time of each controller recomputation (only steps where K
actually changes).

Usage:
    # Time all algorithms on all systems (single trial per system)
    python run_timing.py

    # Only physical systems, 200 steps
    python run_timing.py --suite physical --num-steps 200

    # Specific system by index
    python run_timing.py --systems 0
"""

import argparse
import os
import pickle

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate_timed
from algorithms.irlqr import IRLQR
from algorithms.cec import CEC
from algorithms.cec_pe import CECPE
from algorithms.laglq import LAGLQ
from algorithms.oslo import OSLO
from algorithms.thompson_sampling import TS
from systems import (
    get_benchmark_systems,
    get_physical_systems,
    get_integrator_systems,
    LQRSystem,
)


def time_on_system(
    system: LQRSystem,
    num_steps: int,
    key: jax.Array,
    lam: float = 0.1,
    oslo_mu: float = 0.1,
    oslo_beta: float = 1.0,
    laglq_beta: float = 0.05,
    laglq_eps: float = 1e-2,
    laglq_max_dual_iters: int = 50,
) -> dict[str, dict]:
    """Run all algorithms on a single system, return results with timing."""

    algos = {
        "IR-LQR": IRLQR(lam=lam, beta=1, use_invsqrt=False, A0=system.A0, B0=system.B0),
        "TS": TS(lam=lam, beta=1e-3, A0=system.A0, B0=system.B0),
        "CEC": CEC(lam=lam, A0=system.A0, B0=system.B0),
        "CEC+PE": CECPE(lam=lam, init_act_std=1.0, A0=system.A0, B0=system.B0),
        "LagLQ": LAGLQ(
            lam=lam,
            beta=1e-3,
            eps=laglq_eps,
            max_dual_iters=laglq_max_dual_iters,
            A0=system.A0,
            B0=system.B0,
            penalty_aux=1e4,
            solver="sda",
        ),
        "OSLO": OSLO(
            mu=1e-3,
            lam=lam,
            beta=1.0,
            sigma=system.noise_sigma,
            A0=system.A0,
            B0=system.B0,
        ),
    }

    results = {}
    for name, algo in algos.items():
        print(f"    {name} ...", end=" ", flush=True)
        res = simulate_timed(
            algo,
            system.A_star,
            system.B_star,
            system.Q,
            system.R,
            key,
            num_steps,
            system.noise_sigma,
            system.x0,
        )
        times = res["update_times"]
        n = res["num_updates"]
        if n > 0:
            total = sum(times)
            mean_ms = total / n * 1e3
            print(f"{n} updates, total={total:.3f}s, mean={mean_ms:.3f}ms/update")
        else:
            print("0 updates")
        results[name] = res

    return results


def plot_timing(
    results_dict: dict[str, dict],
    title: str = "",
    save_path: str | None = None,
) -> None:
    """Bar chart with median, 0.2/0.8 quantile error bars, and update count."""
    names = list(results_dict.keys())
    medians_ms = []
    q20_ms = []
    q80_ms = []
    num_updates = []
    for n in names:
        times = np.array(results_dict[n]["update_times"])
        nu = results_dict[n]["num_updates"]
        num_updates.append(nu)
        if nu > 0:
            medians_ms.append(float(np.median(times)) * 1e3)
            q20_ms.append(float(np.quantile(times, 0.2)) * 1e3)
            q80_ms.append(float(np.quantile(times, 0.8)) * 1e3)
        else:
            medians_ms.append(0.0)
            q20_ms.append(0.0)
            q80_ms.append(0.0)

    medians_ms = np.array(medians_ms)
    q20_ms = np.array(q20_ms)
    q80_ms = np.array(q80_ms)

    fig, (ax_time, ax_count) = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(names))
    yerr = np.array([medians_ms - q20_ms, q80_ms - medians_ms])
    ax_time.bar(x, medians_ms, yerr=yerr, capsize=4)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(names, rotation=30, ha="right")
    ax_time.set_yscale("log")
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_title("Controller recomputation time (median, 20–80% quantile)")
    ax_time.grid(axis="y")

    ax_count.bar(x, num_updates)
    ax_count.set_xticks(x)
    ax_count.set_xticklabels(names, rotation=30, ha="right")
    ax_count.set_ylabel("Count")
    ax_count.set_title("Number of controller updates")
    ax_count.grid(axis="y")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="LQR controller timing benchmark")
    parser.add_argument("--num-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "physical", "integrator"],
    )
    parser.add_argument("--systems", type=int, nargs="*", default=None)
    parser.add_argument("--perturbation", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--oslo-mu", type=float, default=0.1)
    parser.add_argument("--oslo-beta", type=float, default=1.0)
    parser.add_argument("--laglq-beta", type=float, default=0.05)
    parser.add_argument("--laglq-eps", type=float, default=1e-2)
    parser.add_argument("--laglq-max-dual-iters", type=int, default=50)
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    key, sys_key = jax.random.split(key)
    suite_fn = {
        "all": get_benchmark_systems,
        "physical": get_physical_systems,
        "integrator": get_integrator_systems,
    }[args.suite]
    systems = suite_fn(sys_key, perturbation=args.perturbation)

    indices = args.systems if args.systems is not None else list(range(len(systems)))

    all_results = {}
    for idx in indices:
        system = systems[idx]
        d_x = system.A_star.shape[0]
        d_u = system.B_star.shape[1]
        print(f"\n{'='*60}")
        print(f"System {idx}: {system.name}  (d_x={d_x}, d_u={d_u})")
        print(f"{'='*60}")

        key, run_key = jax.random.split(key)
        results = time_on_system(
            system,
            args.num_steps,
            run_key,
            lam=args.lam,
            oslo_mu=args.oslo_mu,
            oslo_beta=args.oslo_beta,
            laglq_beta=args.laglq_beta,
            laglq_eps=args.laglq_eps,
            laglq_max_dual_iters=args.laglq_max_dual_iters,
        )
        all_results[system.name] = results

        fig_dir = "figures"
        os.makedirs(fig_dir, exist_ok=True)
        safe_name = system.name.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(fig_dir, f"timing_{safe_name}.pdf")
        plot_timing(results, title=f"{system.name}", save_path=save_path)

    # Save results to disk
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # Convert JAX arrays to numpy for serialization
    serializable = {}
    for sys_name, res in all_results.items():
        serializable[sys_name] = {}
        for algo_name, algo_res in res.items():
            serializable[sys_name][algo_name] = {
                k: (np.array(v) if hasattr(v, "shape") else v)
                for k, v in algo_res.items()
            }
    save_data = {"args": vars(args), "results": serializable}
    save_file = os.path.join(results_dir, f"timing_{args.suite}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {save_file}")


if __name__ == "__main__":
    main()
