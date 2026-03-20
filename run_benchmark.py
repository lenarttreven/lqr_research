"""Run LQR algorithms on the full benchmark suite.

Usage:
    # Run all systems (physical + integrator chains)
    python run_benchmark.py

    # Run only the physical systems
    python run_benchmark.py --suite physical

    # Run only integrator chains
    python run_benchmark.py --suite integrator

    # Run only specific systems by index
    python run_benchmark.py --systems 0 3 9

    # Customize number of trials and steps
    python run_benchmark.py --num-trials 50 --num-steps 500

    # Set random seed
    python run_benchmark.py --seed 42
"""

import argparse

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simulation import simulate_many, plot_regret
from algorithms.ofu import OFU
from algorithms.cec import CEC
from algorithms.cec_pe import CECPE
from algorithms.laglq import LAGLQ
from algorithms.oslo import OSLO, simulate_oslo_many
from algorithms.thompson_sampling import TS
from systems import (
    get_benchmark_systems,
    get_physical_systems,
    get_integrator_systems,
    LQRSystem,
)


def run_on_system(
    system: LQRSystem,
    num_trials: int,
    num_steps: int,
    key: jax.Array,
    lam: float = 0.1,
    oslo_mu: float = 0.1,
    oslo_beta: float = 1.0,
    laglq_beta: float = 0.05,
    laglq_eps: float = 1e-2,
    laglq_max_dual_iters: int = 50,
) -> dict[str, dict]:
    """Run all algorithms on a single system, return results dict."""

    scan_algos = {
        "OFU (V^{-1})": OFU(
            lam=lam, beta=0.05, use_invsqrt=False, A0=system.A0, B0=system.B0
        ),
        "Thompson Sampling": TS(lam=lam, beta=0.05, A0=system.A0, B0=system.B0),
        "CEC (doubling)": CEC(lam=lam, A0=system.A0, B0=system.B0),
        "CEC + PE (doubling)": CECPE(
            lam=lam, init_act_std=1.0, A0=system.A0, B0=system.B0
        ),
        "LAGLQ": LAGLQ(
            lam=lam,
            beta=laglq_beta,
            eps=laglq_eps,
            max_dual_iters=laglq_max_dual_iters,
            A0=system.A0,
            B0=system.B0,
            penalty_aux=1e2,
        ),
    }
    oslo_algos = {
        "OSLO": OSLO(
            mu=oslo_mu,
            lam=lam,
            beta=oslo_beta,
            sigma=system.noise_sigma,
            A0=system.A0,
            B0=system.B0,
        ),
    }

    keys = jax.random.split(key, num_trials)

    results = {}
    for name, algo in scan_algos.items():
        results[name] = simulate_many(
            algo,
            system.A_star,
            system.B_star,
            system.Q,
            system.R,
            keys,
            num_steps,
            system.noise_sigma,
            system.x0,
        )

    for name, algo in oslo_algos.items():
        results[name] = simulate_oslo_many(
            algo,
            system.A_star,
            system.B_star,
            system.Q,
            system.R,
            keys,
            num_steps,
            system.noise_sigma,
            system.x0,
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="LQR benchmark suite")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "physical", "integrator"],
        help="Which system suite to run. Default: all.",
    )
    parser.add_argument(
        "--systems",
        type=int,
        nargs="*",
        default=None,
        help="Indices of systems to run. Default: all in the chosen suite.",
    )
    parser.add_argument(
        "--perturbation",
        type=float,
        default=0.01,
        help="Magnitude of the initial prior perturbation on A0, B0. Default: 0.01.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
        help="Regularization parameter lambda for RLS. Default: 0.1.",
    )
    parser.add_argument(
        "--oslo-mu",
        type=float,
        default=0.1,
        help="OSLO optimism parameter mu. Default: 0.1.",
    )
    parser.add_argument(
        "--oslo-beta",
        type=float,
        default=1.0,
        help="OSLO scaling parameter beta. Default: 1.0.",
    )
    parser.add_argument(
        "--laglq-beta",
        type=float,
        default=0.05,
        help="LAGLQ confidence set radius beta. Default: 0.05.",
    )
    parser.add_argument(
        "--laglq-eps",
        type=float,
        default=1e-2,
        help="LAGLQ dual bisection tolerance. Default: 1e-2.",
    )
    parser.add_argument(
        "--laglq-max-dual-iters",
        type=int,
        default=50,
        help="Maximum dual-search iterations for LAGLQ. Default: 50.",
    )
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

    for idx in indices:
        system = systems[idx]
        d_x = system.A_star.shape[0]
        d_u = system.B_star.shape[1]
        print(f"\n{'='*60}")
        print(f"System {idx}: {system.name}  (d_x={d_x}, d_u={d_u})")
        print(f"{'='*60}")

        key, run_key = jax.random.split(key)
        results = run_on_system(
            system,
            args.num_trials,
            args.num_steps,
            run_key,
            lam=args.lam,
            oslo_mu=args.oslo_mu,
            oslo_beta=args.oslo_beta,
            laglq_beta=args.laglq_beta,
            laglq_eps=args.laglq_eps,
            laglq_max_dual_iters=args.laglq_max_dual_iters,
        )

        # optimal cost is the same across algorithms
        first_res = next(iter(results.values()))
        print(f"  Optimal avg stage cost: {float(first_res['optimal_cost'][0]):.4f}")

        for name, res in results.items():
            median_regret = jnp.median(jnp.sum(res["regrets"], axis=1))
            print(f"  {name}: median cum. regret = {median_regret:.2f}")

        plot_regret(results, title=f"Cumulative Regret — {system.name}", log_y=True)


if __name__ == "__main__":
    main()
