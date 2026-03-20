import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Float, Array

from algorithms.base import LQRAlgorithm
from lqr_utils import make_cost_matrix, dlqr_joint, lqr_avg_stage_cost


def simulate(
    algo: type[LQRAlgorithm],
    A: Float[Array, "dx dx"],
    B: Float[Array, "dx du"],
    Q: Float[Array, "dx dx"],
    R: Float[Array, "du du"],
    key: jax.Array,
    num_steps: int,
    noise_sigma: float,
    x0: Float[Array, "dx"] | None = None,
) -> dict:
    """Run a single-trajectory LQR learning experiment.

    The simulation loop:
        1. u, state = algo.get_action(x, state, key)
        2. x_next = A @ x + B @ u + w       (w ~ N(0, sigma^2 I))
        3. state = algo.update(x, u, x_next, state, t)

    Returns dict with:
        costs:        Float[Array, "T"]     per-step stage costs
        regrets:      Float[Array, "T"]     per-step regret vs oracle
        optimal_cost: Float[Array, ""]      oracle average stage cost
        k_star:       Float[Array, "du dx"] oracle optimal gain
    """
    dx, du = B.shape
    if x0 is None:
        x0 = jnp.zeros((dx,), dtype=A.dtype)

    H = make_cost_matrix(Q, R)  # (dx+du, dx+du)
    k_star, _ = dlqr_joint(A, B, H)  # (du, dx)
    optimal_cost = lqr_avg_stage_cost(A, B, Q, R, k_star, noise_sigma)  # scalar

    algo_state = algo.init_state(dx, du, Q, R)

    def one_step(carry, t):
        key, x, algo_state = carry

        # action
        key, sub_act = jax.random.split(key)
        u, algo_state = algo.get_action(x, algo_state, sub_act)  # (du,)

        # environment dynamics
        key, sub_w = jax.random.split(key)
        w = noise_sigma * jax.random.normal(sub_w, shape=(dx,), dtype=A.dtype)  # (dx,)
        x_next = A @ x + B @ u + w  # (dx,)

        # algorithm update
        algo_state = algo.update(x, u, x_next, algo_state, t)

        # stage cost and instantaneous regret
        cost = x.T @ Q @ x + u.T @ R @ u  # scalar
        regret = cost - optimal_cost  # scalar

        carry_next = (key, x_next, algo_state)
        return carry_next, (cost, regret)

    init_carry = (key, x0, algo_state)
    _, (costs, regrets) = jax.lax.scan(
        one_step, init_carry, xs=jnp.arange(num_steps, dtype=jnp.int32)
    )

    return {
        "costs": costs,          # (T,)
        "regrets": regrets,      # (T,)
        "optimal_cost": optimal_cost,
        "k_star": k_star,        # (du, dx)
    }


def simulate_many(
    algo: type[LQRAlgorithm],
    A: Float[Array, "dx dx"],
    B: Float[Array, "dx du"],
    Q: Float[Array, "dx dx"],
    R: Float[Array, "du du"],
    keys: jax.Array,
    num_steps: int,
    noise_sigma: float,
    x0: Float[Array, "dx"] | None = None,
) -> dict:
    """Vmap simulate over multiple random seeds.

    Returns dict with same keys as simulate, but with leading batch dimension:
        costs:   Float[Array, "num_trials T"]
        regrets: Float[Array, "num_trials T"]
    """
    return jax.vmap(
        lambda key: simulate(algo, A, B, Q, R, key, num_steps, noise_sigma, x0)
    )(keys)


def plot_regret(
    results_dict: dict[str, dict],
    title: str = "Cumulative Regret Comparison",
    log_y: bool = False,
) -> None:
    """Plot cumulative regret with 20-80% quantile bands.

    Args:
        results_dict: {algo_name: results} where results comes from simulate_many.
        log_y: If True, use logarithmic y-axis.
    """
    linestyles = ["-", "--", ":", "-."]

    for i, (name, results) in enumerate(results_dict.items()):
        cum = jnp.cumsum(results["regrets"], axis=1)  # (num_trials, T)
        q20, q50, q80 = jnp.quantile(
            cum, jnp.array([0.2, 0.5, 0.8]), axis=0
        )  # each (T,)
        t = jnp.arange(q50.shape[0])
        ls = linestyles[i % len(linestyles)]
        plt.plot(t, q50, label=name, linestyle=ls)
        plt.fill_between(t, q20, q80, alpha=0.15)

    if log_y:
        plt.yscale("log")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Regret")
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()
