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
    solver: str = "sda",
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
    k_star, _ = dlqr_joint(A, B, H, solver=solver)  # (du, dx)
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

        # current controller gain
        K = algo_state.K  # (du, dx)

        carry_next = (key, x_next, algo_state)
        return carry_next, (cost, regret, K)

    init_carry = (key, x0, algo_state)
    _, (costs, regrets, gains) = jax.lax.scan(
        one_step, init_carry, xs=jnp.arange(num_steps, dtype=jnp.int32)
    )

    # Expected per-step cost under each controller gain
    expected_costs = jax.vmap(
        lambda K: lqr_avg_stage_cost(A, B, Q, R, K, noise_sigma)
    )(gains)  # (T,)

    return {
        "costs": costs,  # (T,)
        "regrets": regrets,  # (T,)
        "expected_costs": expected_costs,  # (T,)
        "optimal_cost": optimal_cost,
        "k_star": k_star,  # (du, dx)
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
    solver: str = "sda",
) -> dict:
    """Vmap simulate over multiple random seeds.

    Returns dict with same keys as simulate, but with leading batch dimension:
        costs:   Float[Array, "num_trials T"]
        regrets: Float[Array, "num_trials T"]
    """
    return jax.vmap(
        lambda key: simulate(algo, A, B, Q, R, key, num_steps, noise_sigma, x0, solver=solver)
    )(keys)


def plot_results(
    results_dict: dict[str, dict],
    title: str = "",
    log_y: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot cumulative regret and per-step excess cost (2 subplots).

    Args:
        results_dict: {algo_name: results} where results comes from simulate_many.
        title: Overall figure title.
        log_y: If True, use logarithmic y-axis on both subplots.
        save_path: If provided, save the figure to this path.
    """
    linestyles = ["-", "--", ":", "-."]
    fig, (ax_regret, ax_cost) = plt.subplots(1, 2, figsize=(12, 4))

    for i, (name, results) in enumerate(results_dict.items()):
        ls = linestyles[i % len(linestyles)]
        t = jnp.arange(results["regrets"].shape[1])

        # Cumulative regret
        cum = jnp.cumsum(results["regrets"], axis=1)  # (num_trials, T)
        q20, q50, q80 = jnp.quantile(cum, jnp.array([0.2, 0.5, 0.8]), axis=0)
        ax_regret.plot(t, q50, label=name, linestyle=ls)
        ax_regret.fill_between(t, q20, q80, alpha=0.15)

        # Expected per-step cost minus optimal cost
        excess = results["expected_costs"] - results["optimal_cost"][:, None]  # (num_trials, T)
        q20, q50, q80 = jnp.quantile(excess, jnp.array([0.2, 0.5, 0.8]), axis=0)
        ax_cost.plot(t, q50, label=name, linestyle=ls)
        ax_cost.fill_between(t, q20, q80, alpha=0.15)

    if log_y:
        ax_regret.set_yscale("log")
        ax_cost.set_yscale("log")
    for ax in (ax_regret, ax_cost):
        ax.set_xlabel("Time step")
        ax.grid()

    ax_regret.set_ylabel("Cumulative Regret")
    ax_regret.set_title("Cumulative Regret")
    ax_cost.set_ylabel("Expected Cost − Optimal Cost")
    ax_cost.set_title("Expected Per-Step Excess Cost")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Single legend on top with 3 columns (below suptitle)
    handles, labels = ax_regret.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.93))

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
