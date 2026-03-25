import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
    expected_costs = jax.vmap(lambda K: lqr_avg_stage_cost(A, B, Q, R, K, noise_sigma))(
        gains
    )  # (T,)

    # Cumulative controller update count: detect when K changes between steps
    K_init = algo.init_state(dx, du, Q, R).K
    K_prev = jnp.concatenate([K_init[None], gains[:-1]], axis=0)  # (T, du, dx)
    k_changed = jnp.any(gains != K_prev, axis=(-2, -1))  # (T,)
    cum_updates = jnp.cumsum(k_changed)  # (T,)

    return {
        "costs": costs,  # (T,)
        "regrets": regrets,  # (T,)
        "expected_costs": expected_costs,  # (T,)
        "optimal_cost": optimal_cost,
        "k_star": k_star,  # (du, dx)
        "cum_updates": cum_updates,  # (T,)
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
        lambda key: simulate(
            algo, A, B, Q, R, key, num_steps, noise_sigma, x0, solver=solver
        )
    )(keys)


def plot_results(
    results_dict: dict[str, dict],
    title: str = "",
    log_y: bool = False,
    save_path: str | None = None,
    plot_cost: bool = False,
) -> None:
    """Plot cumulative regret and optionally per-step excess cost.

    Args:
        results_dict: {algo_name: results} where results comes from simulate_many.
        title: Overall figure title.
        log_y: If True, use logarithmic y-axis on subplots.
        save_path: If provided, save the figure to this path.
    """
    linestyles = ["-", "--", ":", "-."]

    # Adjust columns and figure size based on whether we plot the cost
    ncols = 2 if plot_cost else 1
    figsize = (12, 4) if plot_cost else (6.5, 4)

    # squeeze=False ensures `axes` is always a 2D array, avoiding unpacking errors
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    ax_regret = axes[0, 0]
    ax_cost = axes[0, 1] if plot_cost else None

    for i, (name, results) in enumerate(results_dict.items()):
        ls = linestyles[i % len(linestyles)]
        t = jnp.arange(results["regrets"].shape[1])

        # Cumulative regret (clip extreme values to keep plotting stable)
        cum = jnp.cumsum(results["regrets"], axis=1)  # (num_trials, T)
        cum = jnp.clip(cum, a_min=-1e30, a_max=1e30)
        q20, q50, q80 = jnp.quantile(cum, jnp.array([0.2, 0.5, 0.8]), axis=0)
        ax_regret.plot(t, q50, label=name, linestyle=ls)
        ax_regret.fill_between(t, q20, q80, alpha=0.15)

        # Expected per-step cost minus optimal cost
        if plot_cost:
            excess = (
                results["expected_costs"] - results["optimal_cost"][:, None]
            )  # (num_trials, T)
            q20, q50, q80 = jnp.quantile(excess, jnp.array([0.2, 0.5, 0.8]), axis=0)
            ax_cost.plot(t, q50, label=name, linestyle=ls)
            ax_cost.fill_between(t, q20, q80, alpha=0.15)

    # Formatting active axes
    active_axes = [ax_regret, ax_cost] if plot_cost else [ax_regret]

    if log_y:
        for ax in active_axes:
            ax.set_yscale("log")

    for ax in active_axes:
        ax.set_xlabel("Time step")
        ax.grid()

    ax_regret.set_ylabel("Cumulative Regret")
    ax_regret.set_title("Cumulative Regret")

    if plot_cost:
        ax_cost.set_ylabel("Expected Cost − Optimal Cost")
        ax_cost.set_title("Expected Per-Step Excess Cost")

    if title:
        fig.suptitle(title)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    # Single legend on top with 3 columns (below suptitle)
    handles, labels = ax_regret.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.93))

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def simulate_timed(
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
    """Run a single trajectory, timing only steps where K is recomputed.

    Uses a Python for-loop (slower than scan) so we can measure wall-clock
    time of each controller recomputation.  Only records a timing when the
    gain matrix K actually changes after ``algo.update``.

    Returns dict with the same keys as ``simulate`` plus:
        update_times:  list[float]  wall-clock seconds for each recomputation
        update_steps:  list[int]    time-step indices where K was recomputed
        num_updates:   int          total number of controller recomputations
    """
    dx, du = B.shape
    if x0 is None:
        x0 = jnp.zeros((dx,), dtype=A.dtype)

    H = make_cost_matrix(Q, R)
    k_star, _ = dlqr_joint(A, B, H, solver=solver)
    optimal_cost = lqr_avg_stage_cost(A, B, Q, R, k_star, noise_sigma)

    algo_state = algo.init_state(dx, du, Q, R)

    # JIT the update for JAX-traceable algorithms (not OSLO which uses CVXPY)
    from algorithms.oslo import OSLO

    can_jit = not isinstance(algo, OSLO)
    if can_jit:
        update_fn = jax.jit(algo.update)
        # warmup: compile once so JIT overhead is excluded from timing
        _dummy_state = update_fn(
            x0, jnp.zeros((du,), dtype=A.dtype), x0, algo_state, jnp.int32(0)
        )
        jax.block_until_ready(_dummy_state)
        del _dummy_state
    else:
        update_fn = algo.update

    costs = []
    regrets = []
    gains = []
    update_times: list[float] = []
    update_steps: list[int] = []
    x = x0

    for t in range(num_steps):
        key, sub_act, sub_w = jax.random.split(key, 3)

        # action
        u, algo_state = algo.get_action(x, algo_state, sub_act)

        # environment dynamics
        w = noise_sigma * jax.random.normal(sub_w, shape=(dx,), dtype=A.dtype)
        x_next = A @ x + B @ u + w

        # snapshot K before update
        K_before = algo_state.K

        # --- time the update call ---
        t0 = time.perf_counter()
        algo_state = update_fn(x, u, x_next, algo_state, jnp.int32(t))
        jax.block_until_ready(algo_state)
        elapsed = time.perf_counter() - t0
        # ----------------------------

        # record timing only when K actually changed
        if not jnp.array_equal(K_before, algo_state.K):
            update_times.append(elapsed)
            update_steps.append(t)

        cost = float(x.T @ Q @ x + u.T @ R @ u)
        regret = cost - float(optimal_cost)
        costs.append(cost)
        regrets.append(regret)
        gains.append(algo_state.K)

        x = x_next

    gains_arr = jnp.stack(gains)
    expected_costs = jax.vmap(lambda K: lqr_avg_stage_cost(A, B, Q, R, K, noise_sigma))(
        gains_arr
    )

    return {
        "costs": jnp.array(costs),
        "regrets": jnp.array(regrets),
        "expected_costs": expected_costs,
        "optimal_cost": optimal_cost,
        "k_star": k_star,
        "update_times": update_times,
        "update_steps": update_steps,
        "num_updates": len(update_times),
    }
