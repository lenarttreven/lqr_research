"""Run LQR learning experiments and compare algorithms."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from simulation import simulate_many, plot_regret
from algorithms.ofu import OFU
from algorithms.cec_pe import CECPE
from algorithms.oslo import OSLO, simulate_oslo_many


if __name__ == "__main__":
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

    num_trials = 100
    num_steps = 1_000
    noise_sigma = 0.1

    keys = jax.random.split(key, num_trials)
    x0 = jax.random.uniform(key, shape=(d_x,))

    # -- Define algorithms --
    scan_algos = {
        "OFU (V^{-1})": OFU(lam=1.0, beta=0.05, use_invsqrt=False, A0=A0, B0=B0),
        "CEC + PE (doubling)": CECPE(lam=1.0, init_act_std=1.0, A0=A0, B0=B0),
    }
    oslo_algos = {
        "OSLO": OSLO(mu=0.1, lam=1.0, beta=1.0, sigma=noise_sigma, A0=A0, B0=B0),
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
    plot_regret(results)
