"""OSLO: Optimistic Semi-definite programming for LQ cOntrol.

Implements Algorithm 1 from Cohen, Koren, Mansour (2019):
"Learning Linear-Quadratic Regulators Efficiently with only sqrt(T) Regret."

The algorithm solves a relaxed SDP at each epoch to compute an optimistic
policy.  Because the SDP is solved via CVXPY (not JAX-traceable), this
algorithm cannot be used inside jax.lax.scan.  Use `simulate_oslo` from
this module instead.
"""

from typing import NamedTuple

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, Array

from algorithms.base import LQRAlgorithm
from lqr_utils import logdet, dlqr_joint, make_cost_matrix


class OSLOState(NamedTuple):
    """Algorithm state for OSLO.

    K:         Float[Array, "du dx"]     current controller gain
    V:         Float[Array, "dxu dxu"]   committed confidence matrix (at epoch start)
    V_cur:     Float[Array, "dxu dxu"]   running confidence matrix
    S:         Float[Array, "dx dxu"]    RLS cross-correlation (scaled by 1/beta)
    logdet_V:  Float[Array, ""]          log-det of committed V
    Q:         Float[Array, "dx dx"]     state cost matrix
    R:         Float[Array, "du du"]     input cost matrix
    mu:        Float[Array, ""]          optimism parameter
    lam:       Float[Array, ""]          regularization parameter (lambda)
    beta:      Float[Array, ""]          scaling parameter
    sigma:     Float[Array, ""]          noise standard deviation
    A0:        Float[Array, "dx dx"]     initial estimate of A
    B0:        Float[Array, "dx du"]     initial estimate of B
    A_hat:     Float[Array, "dx dx"]     current estimate of A
    B_hat:     Float[Array, "dx du"]     current estimate of B
    """

    K: jax.Array
    V: jax.Array
    V_cur: jax.Array
    S: jax.Array
    logdet_V: jax.Array
    Q: jax.Array
    R: jax.Array
    mu: jax.Array
    lam: jax.Array
    beta: jax.Array
    sigma: jax.Array
    A0: jax.Array
    B0: jax.Array
    A_hat: jax.Array
    B_hat: jax.Array


def _solve_oslo_sdp(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    V_inv: np.ndarray,
    W: np.ndarray,
    mu: float,
) -> np.ndarray | None:
    """Solve the relaxed SDP (line 8 of Algorithm 1) via CVXPY.

    min  Tr((Q 0; 0 R) @ Sigma)
    s.t. Sigma_xx >= (A B) Sigma (A B)^T + W - mu * Tr(V^{-1} Sigma) * I
         Sigma >= 0

    Returns the optimal Sigma or None if infeasible.
    """
    dx = A.shape[0]
    du = B.shape[1]
    n = dx + du

    Sigma = cp.Variable((n, n), symmetric=True)
    Sigma_xx = Sigma[:dx, :dx]

    AB = np.concatenate([A, B], axis=1)  # (dx, n)

    # Cost matrix block-diagonal
    H = np.block(
        [
            [Q, np.zeros((dx, du))],
            [np.zeros((du, dx)), R],
        ]
    )

    objective = cp.Minimize(cp.trace(H @ Sigma))

    # Relaxed constraint:
    # Sigma_xx >= AB @ Sigma @ AB^T + W - mu * tr(V^{-1} @ Sigma) * I
    optimism_term = mu * cp.trace(V_inv @ Sigma) * np.eye(dx)
    constraint_lhs = Sigma_xx - AB @ Sigma @ AB.T - W + optimism_term

    constraints = [
        Sigma >> 0,
        constraint_lhs >> 0,
    ]

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except (cp.SolverError, Exception):
        # Fallback to SCS if CLARABEL fails
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=20_000, eps=1e-9)
        except cp.SolverError:
            return None

    if problem.status in ("infeasible", "unbounded", None) or Sigma.value is None:
        return None

    return np.array(Sigma.value)


class OSLO(LQRAlgorithm):
    """OSLO algorithm (Cohen, Koren, Mansour 2019).

    Args:
        mu:    Optimism parameter (controls exploration bonus in the SDP).
        lam:   Regularization parameter (lambda) for the confidence matrix.
        beta:  Scaling parameter for the RLS updates.
        sigma: Process noise standard deviation.
        A0:    Initial estimate of the A matrix.
        B0:    Initial estimate of the B matrix.
    """

    def __init__(
        self,
        mu: float,
        lam: float,
        beta: float,
        sigma: float = 0.1,
        A0: Float[Array, "dx dx"] | None = None,
        B0: Float[Array, "dx du"] | None = None,
        solver: str = "sda",
    ):
        self.mu = mu
        self.lam = lam
        self.beta = beta
        self.sigma = sigma
        self.A0 = A0
        self.B0 = B0
        self.solver = solver

    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> OSLOState:
        dxu = dx + du
        H = make_cost_matrix(Q, R)  # (dxu, dxu)
        V = self.lam * jnp.eye(dxu, dtype=Q.dtype)  # (dxu, dxu)
        S = jnp.zeros((dx, dxu), dtype=Q.dtype)  # (dx, dxu)
        A0 = self.A0 if self.A0 is not None else jnp.zeros((dx, dx), dtype=Q.dtype)
        B0 = self.B0 if self.B0 is not None else jnp.zeros((dx, du), dtype=Q.dtype)

        # compute initial controller from (A0, B0)
        K, _ = dlqr_joint(A0, B0, H, solver=self.solver)
        return OSLOState(
            K=K,
            V=V,
            V_cur=V,
            S=S,
            logdet_V=logdet(V),
            Q=Q,
            R=R,
            mu=jnp.array(self.mu, dtype=Q.dtype),
            lam=jnp.array(self.lam, dtype=Q.dtype),
            beta=jnp.array(self.beta, dtype=Q.dtype),
            sigma=jnp.array(self.sigma, dtype=Q.dtype),
            A0=A0,
            B0=B0,
            A_hat=A0,
            B_hat=B0,
        )

    def get_action(
        self,
        x: Float[Array, "dx"],
        state: OSLOState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], OSLOState]:
        """Deterministic action u = K @ x."""
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))  # (du,)
        return u, state

    @staticmethod
    def _rls_estimate(state: OSLOState, dx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute (A_hat, B_hat) from the accumulated RLS statistics.

        The least-squares objective (line 7 of Algorithm 1):
            (1/beta) sum ||(A B) z_s - x_{s+1}||^2 + lambda ||(A B) - (A0 B0)||_F^2

        Solution: (A B) = (lambda * (A0 B0) + S) @ V^{-1}
        where V = lambda I + (1/beta) sum z_s z_s^T  and  S = (1/beta) sum x_{s+1} z_s^T.
        """
        du = state.V.shape[0] - dx
        AB0 = jnp.concatenate([state.A0, state.B0], axis=1)  # (dx, dxu)
        RHS = state.lam * AB0 + state.S  # (dx, dxu)
        AB_hat = jax.scipy.linalg.solve(
            state.V_cur.T, RHS.T, assume_a="sym"
        ).T  # (dx, dxu)
        A_hat = AB_hat[:, :dx]  # (dx, dx)
        B_hat = AB_hat[:, dx:]  # (dx, du)
        return A_hat, B_hat

    @staticmethod
    def _solve_sdp_and_extract_policy(
        A_hat: jnp.ndarray,
        B_hat: jnp.ndarray,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        V: jnp.ndarray,
        sigma: float,
        mu: float,
    ) -> jnp.ndarray | None:
        """Solve the relaxed SDP and extract policy K = Sigma_ux @ inv(Sigma_xx).

        Returns new K or None if SDP is infeasible.
        """
        dx = A_hat.shape[0]
        du = B_hat.shape[1]

        # Convert to numpy for CVXPY
        A_np = np.array(A_hat)
        B_np = np.array(B_hat)
        Q_np = np.array(Q)
        R_np = np.array(R)
        V_np = np.array(V)

        # V^{-1} for the SDP
        V_inv_np = np.linalg.solve(V_np, np.eye(V_np.shape[0]))
        V_inv_np = 0.5 * (V_inv_np + V_inv_np.T)

        # W = sigma^2 * I
        W_np = float(sigma) ** 2 * np.eye(dx)

        Sigma_opt = _solve_oslo_sdp(
            A_np,
            B_np,
            Q_np,
            R_np,
            V_inv_np,
            W_np,
            float(mu),
        )

        if Sigma_opt is None:
            return None

        Sigma_xx = Sigma_opt[:dx, :dx]
        Sigma_ux = Sigma_opt[dx:, :dx]

        # K = Sigma_ux @ inv(Sigma_xx)
        try:
            K_np = Sigma_ux @ np.linalg.solve(Sigma_xx, np.eye(dx))
        except np.linalg.LinAlgError:
            return None

        return jnp.array(K_np)

    def update(
        self,
        x: Float[Array, "dx"],
        u: Float[Array, "du"],
        x_next: Float[Array, "dx"],
        state: OSLOState,
        t: int,
    ) -> OSLOState:
        """Accumulate RLS stats; recompute policy via SDP when det doubles.

        NOTE: This method calls CVXPY and is NOT compatible with jax.lax.scan.
        Use `simulate_oslo` for simulation.
        """
        dx = x.shape[0]
        beta = float(state.beta)

        # Accumulate sufficient statistics (scaled by 1/beta)
        xu = jnp.concatenate([x, u], axis=0)  # (dxu,)
        V_cur = state.V_cur + (1.0 / beta) * jnp.outer(xu, xu)  # (dxu, dxu)
        S = state.S + (1.0 / beta) * jnp.outer(x_next, xu)  # (dx, dxu)

        state = state._replace(V_cur=V_cur, S=S)

        # Determinant-doubling trigger (line 5)
        logdet_V_cur = logdet(V_cur)
        do_update = bool(logdet_V_cur > state.logdet_V + jnp.log(2.0)) or (t == 0)

        if do_update:
            # Estimate parameters (line 7)
            A_hat, B_hat = OSLO._rls_estimate(state, dx)

            # Solve relaxed SDP and extract policy (lines 8-9)
            K_new = OSLO._solve_sdp_and_extract_policy(
                A_hat,
                B_hat,
                state.Q,
                state.R,
                V_cur,
                state.sigma,
                state.mu,
            )

            if K_new is not None:
                state = state._replace(
                    K=K_new,
                    V=V_cur,
                    logdet_V=logdet_V_cur,
                    A_hat=A_hat,
                    B_hat=B_hat,
                )
            else:
                # SDP infeasible: keep current policy, still commit V
                state = state._replace(
                    V=V_cur,
                    logdet_V=logdet_V_cur,
                    A_hat=A_hat,
                    B_hat=B_hat,
                )

        return state


def simulate_oslo(
    algo: OSLO,
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
    """Run OSLO on a single trajectory using a Python for-loop.

    Same return format as simulation.simulate.
    """
    from lqr_utils import make_cost_matrix, dlqr_joint, lqr_avg_stage_cost

    dx, du = B.shape
    if x0 is None:
        x0 = jnp.zeros((dx,), dtype=A.dtype)

    H = make_cost_matrix(Q, R)
    k_star, _ = dlqr_joint(A, B, H, solver=solver)
    optimal_cost = lqr_avg_stage_cost(A, B, Q, R, k_star, noise_sigma)

    algo_state = algo.init_state(dx, du, Q, R)

    costs = []
    regrets = []
    gains = []
    x = x0

    for t in range(num_steps):
        key, sub_act, sub_w = jax.random.split(key, 3)

        # Action
        u, algo_state = algo.get_action(x, algo_state, sub_act)

        # Record current gain
        gains.append(algo_state.K)

        # Environment dynamics
        w = noise_sigma * jax.random.normal(sub_w, shape=(dx,), dtype=A.dtype)
        x_next = A @ x + B @ u + w

        # Stage cost and regret
        cost = float(x.T @ Q @ x + u.T @ R @ u)
        regret = cost - float(optimal_cost)
        costs.append(cost)
        regrets.append(regret)

        # Algorithm update
        algo_state = algo.update(x, u, x_next, algo_state, t)

        x = x_next

    gains_arr = jnp.stack(gains)  # (T, du, dx)
    expected_costs = jax.vmap(
        lambda K: lqr_avg_stage_cost(A, B, Q, R, K, noise_sigma)
    )(gains_arr)  # (T,)

    return {
        "costs": jnp.array(costs),
        "regrets": jnp.array(regrets),
        "expected_costs": expected_costs,
        "optimal_cost": optimal_cost,
        "k_star": k_star,
    }


def simulate_oslo_many(
    algo: OSLO,
    A: Float[Array, "dx dx"],
    B: Float[Array, "dx du"],
    Q: Float[Array, "dx dx"],
    R: Float[Array, "du du"],
    keys: jax.Array,
    num_steps: int,
    noise_sigma: float,
    x0: Float[Array, "dx"] | None = None,
) -> dict:
    """Run OSLO over multiple seeds sequentially.

    Returns dict with same structure as simulate_many (leading batch dim).
    """
    all_results = []
    for i in range(keys.shape[0]):
        result = simulate_oslo(
            algo,
            A,
            B,
            Q,
            R,
            keys[i],
            num_steps,
            noise_sigma,
            x0,
        )
        all_results.append(result)

    return {
        "costs": jnp.stack([r["costs"] for r in all_results]),
        "regrets": jnp.stack([r["regrets"] for r in all_results]),
        "expected_costs": jnp.stack([r["expected_costs"] for r in all_results]),
        "optimal_cost": jnp.stack([r["optimal_cost"] for r in all_results]),
        "k_star": jnp.stack([r["k_star"] for r in all_results]),
    }
