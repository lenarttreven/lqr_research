"""Run LQR learning experiments and compare algorithms."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simulation import simulate_many, plot_results
from algorithms.irlqr import IRLQR
from algorithms.cec_pe import CECPE
from algorithms.laglq import LAGLQ
from algorithms.oslo import OSLO, simulate_oslo_many
from algorithms.thompson_sampling import TS


def main() -> None:
    key = jax.random.PRNGKey(0)
    d_x, d_u = 3, 1
    dt = 0.1

    A_star = jnp.array(
        [
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    B_star = jnp.array(
        [
            [dt**3 / 6],
            [dt**2 / 2],
            [dt],
        ]
    )

    # Initial estimates: true system + small perturbation
    key, key_a, key_b = jax.random.split(key, 3)
    A0 = A_star + 0.05 * jax.random.normal(key_a, shape=A_star.shape)
    B0 = B_star + 0.05 * jax.random.normal(key_b, shape=B_star.shape)

    Q = jnp.diag(jnp.array([10.0, 1.0, 0.1]))
    R = jnp.array([[0.1]])

    num_trials = 20
    num_steps = 100
    noise_sigma = 0.1

    keys = jax.random.split(key, num_trials)
    x0 = jax.random.uniform(key, shape=(d_x,))

    # -- Define algorithms --
    scan_algos = {
        "IR-LQR": IRLQR(lam=1.0, beta=0.05, A0=A0, B0=B0),
        "TS": TS(lam=1.0, beta=0.05, A0=A0, B0=B0),
        "CEC+PE": CECPE(lam=1.0, init_act_std=1.0, A0=A0, B0=B0),
        "LagLQ": LAGLQ(
            lam=1.0,
            beta=1.0,
            eps=1e-4,
            max_dual_iters=30,
            A0=A0,
            B0=B0,
            mu_max_cap=0.1,
            penalty_aux=1e2,
        ),
    }
    oslo_algos = {
        "OSLO": OSLO(mu=1e-3, lam=1.0, beta=1.0, sigma=noise_sigma, A0=A0, B0=B0),
    }

    # -- Run scan-compatible algorithms --
    results = {}
    for name, algo in scan_algos.items():
        print(f"Running {name}...")
        results[name] = simulate_many(
            algo,
            A_star,
            B_star,
            Q,
            R,
            keys,
            num_steps,
            noise_sigma,
            x0,
        )
        print(
            f"  done. Median final cum. regret: "
            f"{jnp.median(jnp.sum(results[name]['regrets'], axis=1)):.2f}"
        )

    # -- Run OSLO (requires Python-loop simulation) --
    for name, algo in oslo_algos.items():
        print(f"Running {name}...")
        results[name] = simulate_oslo_many(
            algo,
            A_star,
            B_star,
            Q,
            R,
            keys,
            num_steps,
            noise_sigma,
            x0,
        )
        print(
            f"  done. Median final cum. regret: "
            f"{jnp.median(jnp.sum(results[name]['regrets'], axis=1)):.2f}"
        )

    # -- Plot --
    plot_results(results)


if __name__ == "__main__":
    main()
