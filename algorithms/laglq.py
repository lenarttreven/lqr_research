"""LAGLQ with JAX-friendly updates.

This implementation keeps the paper's main ingredients:
RLS estimation, determinant-doubling policy updates, an extended LQR
formulation, and a dichotomy search over the dual variable ``mu``.

The solver is written only with JAX operations so it can be used inside
`jax.lax.scan` and therefore with `simulation.simulate_many`.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from algorithms.base import LQRAlgorithm
from lqr_utils import dlqr_joint, logdet, make_cost_matrix, rls_estimate, steady_state_covariance


def _symmetrize(M: jax.Array) -> jax.Array:
    return 0.5 * (M + M.T)


def _spectral_radius(A: jax.Array) -> jax.Array:
    eigvals = jnp.linalg.eigvals(A)
    return jnp.max(jnp.abs(eigvals))


def _constraint_blocks(
    V: jax.Array,
    beta: jax.Array,
    dx: int,
    du: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return the paper's constraint blocks in state/control form."""
    dxu = dx + du
    I_dxu = jnp.eye(dxu, dtype=V.dtype)
    V_inv = jax.scipy.linalg.solve(V, I_dxu, assume_a="sym")
    V_inv = _symmetrize(V_inv)

    beta2 = beta**2
    V_xx = V_inv[:dx, :dx]
    V_ux = V_inv[dx:, :dx]
    V_uu = V_inv[dx:, dx:]

    Q_g = -beta2 * V_xx
    N_g = jnp.zeros((du + dx, dx), dtype=V.dtype)
    N_g = N_g.at[:du, :].set(-beta2 * V_ux)

    R_g = jnp.zeros((du + dx, du + dx), dtype=V.dtype)
    R_g = R_g.at[:du, :du].set(-beta2 * V_uu)
    R_g = R_g.at[du:, du:].set(jnp.eye(dx, dtype=V.dtype))

    top = jnp.concatenate([Q_g, N_g.T], axis=1)
    bottom = jnp.concatenate([N_g, R_g], axis=1)
    C_g = _symmetrize(jnp.concatenate([top, bottom], axis=0))
    return Q_g, N_g, R_g, C_g


def _lagrangian_cost_matrix(
    Q: jax.Array,
    R: jax.Array,
    V: jax.Array,
    beta: jax.Array,
    mu: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build the extended Lagrangian cost H_mu = C_dagger + mu C_g."""
    dx = Q.shape[0]
    du = R.shape[0]
    Q_g, N_g, R_g, C_g = _constraint_blocks(V, beta, dx, du)

    R_base = jnp.zeros((du + dx, du + dx), dtype=Q.dtype)
    R_base = R_base.at[:du, :du].set(R)

    Q_mu = _symmetrize(Q + mu * Q_g)
    N_mu = mu * N_g
    R_mu = _symmetrize(R_base + mu * R_g)

    top = jnp.concatenate([Q_mu, N_mu.T], axis=1)
    bottom = jnp.concatenate([N_mu, R_mu], axis=1)
    H_mu = _symmetrize(jnp.concatenate([top, bottom], axis=0))
    return H_mu, Q_g, N_g, R_g, C_g


def _extended_gain_from_u_gain(
    K_u: jax.Array,
    dx: int,
) -> jax.Array:
    K_w = jnp.zeros((dx, K_u.shape[1]), dtype=K_u.dtype)
    return jnp.concatenate([K_u, K_w], axis=0)


class _DualEval(NamedTuple):
    K_ext: jax.Array
    grad: jax.Array
    lambda_min_D: jax.Array
    valid: jax.Array


class _SearchState(NamedTuple):
    mu_l: jax.Array
    mu_r: jax.Array
    K_l: jax.Array
    grad_l: jax.Array
    lam_l: jax.Array
    done: jax.Array
    failed: jax.Array


class LAGLQState(NamedTuple):
    """Algorithm state for LAGLQ."""

    K: jax.Array
    K_ext: jax.Array
    V: jax.Array
    V_cur: jax.Array
    S: jax.Array
    logdet_V: jax.Array
    Q: jax.Array
    R: jax.Array
    lam: jax.Array
    A0: jax.Array
    B0: jax.Array
    A_hat: jax.Array
    B_hat: jax.Array
    beta: jax.Array
    mu: jax.Array
    eps: jax.Array
    D_upper: jax.Array
    horizon: jax.Array
    delta: jax.Array
    noise_sigma: jax.Array
    prior_radius: jax.Array
    eps_scale: jax.Array
    beta_scale: jax.Array
    mu_floor: jax.Array
    mu_max_cap: jax.Array
    max_dual_iters: jax.Array
    stability_tol: jax.Array


class LAGLQ(LQRAlgorithm):
    """JAX-friendly LAGLQ.

    The hyperparameters are stored in the state so `update()` remains a
    pure function suitable for `lax.scan`.
    """

    def __init__(
        self,
        lam: float = 1.0,
        horizon: int = 1_000,
        delta: float = 0.05,
        noise_sigma: float = 0.1,
        prior_radius: float = 0.05,
        eps_scale: float = 1.0,
        beta_scale: float = 1.0,
        mu_floor: float = 1e-6,
        mu_max_cap: float = 1e6,
        max_dual_iters: int = 30,
        cost_upper_bound: float | None = None,
        cost_upper_bound_scale: float = 2.0,
        stability_tol: float = 1e-6,
        A0: Float[Array, "dx dx"] | None = None,
        B0: Float[Array, "dx du"] | None = None,
    ):
        self.lam = lam
        self.horizon = horizon
        self.delta = delta
        self.noise_sigma = noise_sigma
        self.prior_radius = prior_radius
        self.eps_scale = eps_scale
        self.beta_scale = beta_scale
        self.mu_floor = mu_floor
        self.mu_max_cap = mu_max_cap
        self.max_dual_iters = max_dual_iters
        self.cost_upper_bound = cost_upper_bound
        self.cost_upper_bound_scale = cost_upper_bound_scale
        self.stability_tol = stability_tol
        self.A0 = A0
        self.B0 = B0

    @staticmethod
    def _ce_extended_gain(
        A: jax.Array,
        B: jax.Array,
        Q: jax.Array,
        R: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Standard certainty-equivalence gain, embedded in the extended space."""
        H = make_cost_matrix(Q, R)
        K_u, _ = dlqr_joint(A, B, H)
        K_ext = _extended_gain_from_u_gain(K_u, A.shape[0])
        rho = _spectral_radius(A + B @ K_u)
        valid = jnp.all(jnp.isfinite(K_ext)) & jnp.isfinite(rho) & (rho < 1.0 - 1e-6)
        return K_ext, valid

    @staticmethod
    def _fallback_gain(
        A: jax.Array,
        B: jax.Array,
        Q: jax.Array,
        R: jax.Array,
        previous: jax.Array,
    ) -> jax.Array:
        ce_ext, ce_valid = LAGLQ._ce_extended_gain(A, B, Q, R)
        prev_valid = jnp.all(jnp.isfinite(previous))
        zeros = jnp.zeros_like(previous)
        safe_prev = jnp.where(prev_valid, previous, zeros)
        return jnp.where(ce_valid, ce_ext, safe_prev)

    @staticmethod
    def _estimate_cost_upper_bound(
        A: jax.Array,
        B: jax.Array,
        Q: jax.Array,
        R: jax.Array,
        provided: float | None,
        scale: float,
    ) -> jax.Array:
        if provided is not None:
            return jnp.asarray(provided, dtype=Q.dtype)

        H = make_cost_matrix(Q, R)
        _, P = dlqr_joint(A, B, H)
        trace_P = jnp.trace(P)
        fallback = jnp.trace(Q)
        trace_safe = jnp.where(jnp.isfinite(trace_P) & (trace_P > 0), trace_P, fallback)
        return jnp.maximum(1.0, scale * trace_safe)

    @staticmethod
    def _confidence_radius(
        V: jax.Array,
        dx: int,
        du: int,
        lam: jax.Array,
        horizon: jax.Array,
        delta: jax.Array,
        noise_sigma: jax.Array,
        prior_radius: jax.Array,
        beta_scale: jax.Array,
        mu_floor: jax.Array,
    ) -> jax.Array:
        dxu = dx + du
        logdet_V = logdet(V)
        logdet_ref = dxu * jnp.log(jnp.maximum(lam, mu_floor))
        inside = jnp.maximum(
            logdet_V
            - logdet_ref
            + 2.0 * jnp.log(4.0 * jnp.maximum(horizon, 1.0) / jnp.maximum(delta, mu_floor)),
            0.0,
        )
        beta = dx * noise_sigma * jnp.sqrt(inside) + jnp.sqrt(jnp.maximum(lam, mu_floor)) * prior_radius
        return jnp.maximum(mu_floor, beta_scale * beta)

    @staticmethod
    def _paper_search_constants(
        A: jax.Array,
        B: jax.Array,
        Q: jax.Array,
        R: jax.Array,
        V: jax.Array,
        beta: jax.Array,
        D_upper: jax.Array,
        mu_floor: jax.Array,
        mu_max_cap: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dx = A.shape[0]
        C = make_cost_matrix(Q, R)
        lam_min_C = jnp.maximum(jnp.min(jnp.linalg.eigvalsh(C)), mu_floor)
        lam_max_C = jnp.max(jnp.linalg.eigvalsh(C))
        lam_max_V = jnp.max(jnp.linalg.eigvalsh(V))
        mu_max = jnp.minimum(mu_max_cap, lam_max_C * lam_max_V / jnp.maximum(beta**2, mu_floor))

        _, _, _, C_g = _constraint_blocks(V, beta, dx, B.shape[1])
        B_tilde = jnp.concatenate([B, jnp.eye(dx, dtype=A.dtype)], axis=1)

        kappa = jnp.maximum(1.0, D_upper / lam_min_C)
        cg_norm = jnp.linalg.norm(C_g, 2)
        b_norm = jnp.linalg.norm(B, 2)
        alpha = jnp.maximum(1.0, cg_norm / 2.0) * 8.0 * cg_norm * kappa**4
        alpha = alpha * ((2.0 + jnp.linalg.norm(A, 2) * b_norm) * (1.0 + b_norm)) ** 2

        sigma_btilde_sq = jnp.maximum(jnp.min(jnp.linalg.eigvalsh(B_tilde @ B_tilde.T)), mu_floor)
        c_mu_max = (lam_max_C + mu_max) * (1.0 + jnp.linalg.norm(B_tilde, 2) ** 2 * (1.0 + jnp.linalg.norm(A, 2) ** 2))
        inner = jnp.minimum(
            1.0,
            jnp.minimum(1.0, lam_min_C / (2.0 * kappa))
            * sigma_btilde_sq
            / jnp.maximum(2.0 * kappa**2 * c_mu_max, mu_floor),
        )
        raw_lambda0 = jnp.minimum(
            lam_min_C / jnp.maximum(2.0 * jnp.linalg.norm(B_tilde, 2) ** 2 * jnp.maximum(D_upper, 1.0), mu_floor),
            (8.0 ** (-(2 * dx + 1)) / jnp.maximum(kappa ** (2 * dx), mu_floor)) * inner,
        )
        lambda0 = jnp.maximum(raw_lambda0, mu_floor) ** 2
        return mu_max, alpha, lambda0

    @staticmethod
    def _dual_eval(
        A: jax.Array,
        B: jax.Array,
        Q: jax.Array,
        R: jax.Array,
        V: jax.Array,
        beta: jax.Array,
        mu: jax.Array,
        mu_floor: jax.Array,
        stability_tol: jax.Array,
    ) -> _DualEval:
        dx = A.shape[0]
        B_tilde = jnp.concatenate([B, jnp.eye(dx, dtype=A.dtype)], axis=1)

        H_mu, Q_g, N_g, R_g, _ = _lagrangian_cost_matrix(Q, R, V, beta, jnp.maximum(mu, 0.0))
        K_ext, P = dlqr_joint(A, B_tilde, H_mu)

        R_mu = H_mu[dx:, dx:]
        A_cl = A + B_tilde @ K_ext
        D = _symmetrize(R_mu + B_tilde.T @ P @ B_tilde)
        lambda_min_D = jnp.min(jnp.linalg.eigvalsh(D))

        M_g = _symmetrize(Q_g + N_g.T @ K_ext + K_ext.T @ N_g + K_ext.T @ R_g @ K_ext)
        G = steady_state_covariance(A_cl.T, M_g)
        grad = jnp.trace(G)

        rho = _spectral_radius(A_cl)
        finite = (
            jnp.all(jnp.isfinite(K_ext))
            & jnp.all(jnp.isfinite(P))
            & jnp.all(jnp.isfinite(D))
            & jnp.all(jnp.isfinite(G))
            & jnp.isfinite(grad)
            & jnp.isfinite(lambda_min_D)
            & jnp.isfinite(rho)
        )
        valid = finite & (lambda_min_D > mu_floor) & (rho < 1.0 - stability_tol)

        return _DualEval(
            K_ext=jnp.where(valid, K_ext, jnp.zeros_like(K_ext)),
            grad=jnp.where(valid, grad, -jnp.inf),
            lambda_min_D=jnp.where(valid, lambda_min_D, mu_floor),
            valid=valid,
        )

    @staticmethod
    def _solve_policy(
        A: jax.Array,
        B: jax.Array,
        Q: jax.Array,
        R: jax.Array,
        V: jax.Array,
        beta: jax.Array,
        eps: jax.Array,
        D_upper: jax.Array,
        previous: jax.Array,
        mu_floor: jax.Array,
        mu_max_cap: jax.Array,
        max_dual_iters: jax.Array,
        stability_tol: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        fallback = LAGLQ._fallback_gain(A, B, Q, R, previous)
        mu_max, alpha, lambda0 = LAGLQ._paper_search_constants(
            A, B, Q, R, V, beta, D_upper, mu_floor, mu_max_cap
        )

        sol0 = LAGLQ._dual_eval(A, B, Q, R, V, beta, jnp.array(0.0, dtype=A.dtype), mu_floor, stability_tol)

        def invalid_init(_):
            return fallback, jnp.array(0.0, dtype=A.dtype)

        def valid_init(_):
            init = _SearchState(
                mu_l=jnp.array(0.0, dtype=A.dtype),
                mu_r=mu_max,
                K_l=sol0.K_ext,
                grad_l=sol0.grad,
                lam_l=sol0.lambda_min_D,
                done=sol0.grad <= 0.0,
                failed=jnp.array(False),
            )

            def body_fun(_, carry: _SearchState) -> _SearchState:
                stop = alpha * (carry.mu_r - carry.mu_l) / jnp.maximum(carry.lam_l, mu_floor) <= eps
                fail = carry.lam_l <= lambda0 * eps**2
                done = carry.done | stop | fail | (carry.grad_l <= eps)

                def keep_state(_: None) -> _SearchState:
                    return _SearchState(
                        mu_l=carry.mu_l,
                        mu_r=carry.mu_r,
                        K_l=carry.K_l,
                        grad_l=carry.grad_l,
                        lam_l=carry.lam_l,
                        done=done,
                        failed=carry.failed | fail,
                    )

                def update_state(_: None) -> _SearchState:
                    mu_mid = 0.5 * (carry.mu_l + carry.mu_r)
                    sol_mid = LAGLQ._dual_eval(
                        A, B, Q, R, V, beta, mu_mid, mu_floor, stability_tol
                    )
                    go_left = sol_mid.valid & (sol_mid.grad > 0.0)
                    return _SearchState(
                        mu_l=jnp.where(go_left, mu_mid, carry.mu_l),
                        mu_r=jnp.where(go_left, carry.mu_r, mu_mid),
                        K_l=jnp.where(go_left, sol_mid.K_ext, carry.K_l),
                        grad_l=jnp.where(go_left, sol_mid.grad, carry.grad_l),
                        lam_l=jnp.where(go_left, sol_mid.lambda_min_D, carry.lam_l),
                        done=jnp.array(False),
                        failed=carry.failed,
                    )

                return jax.lax.cond(done, keep_state, update_state, operand=None)

            search = jax.lax.fori_loop(0, max_dual_iters, body_fun, init)
            good_gap = alpha * (search.mu_r - search.mu_l) / jnp.maximum(search.lam_l, mu_floor) <= eps
            accept = (~search.failed) & (search.grad_l <= eps) | ((~search.failed) & good_gap)
            K_out = jnp.where(accept, search.K_l, fallback)
            mu_out = jnp.where(accept, search.mu_l, jnp.array(0.0, dtype=A.dtype))
            return K_out, mu_out

        return jax.lax.cond(sol0.valid, valid_init, invalid_init, operand=None)

    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> LAGLQState:
        dxu = dx + du
        V = self.lam * jnp.eye(dxu, dtype=Q.dtype)
        S = jnp.zeros((dx, dxu), dtype=Q.dtype)

        A0 = self.A0 if self.A0 is not None else jnp.zeros((dx, dx), dtype=Q.dtype)
        B0 = self.B0 if self.B0 is not None else jnp.zeros((dx, du), dtype=Q.dtype)

        D_upper = self._estimate_cost_upper_bound(
            A0,
            B0,
            Q,
            R,
            self.cost_upper_bound,
            self.cost_upper_bound_scale,
        )
        beta0 = self._confidence_radius(
            V,
            dx,
            du,
            jnp.array(self.lam, dtype=Q.dtype),
            jnp.array(float(self.horizon), dtype=Q.dtype),
            jnp.array(self.delta, dtype=Q.dtype),
            jnp.array(self.noise_sigma, dtype=Q.dtype),
            jnp.array(self.prior_radius, dtype=Q.dtype),
            jnp.array(self.beta_scale, dtype=Q.dtype),
            jnp.array(self.mu_floor, dtype=Q.dtype),
        )

        K_prev = jnp.zeros((du + dx, dx), dtype=Q.dtype)
        K_ext0, mu0 = self._solve_policy(
            A0,
            B0,
            Q,
            R,
            V,
            beta0,
            jnp.array(self.eps_scale, dtype=Q.dtype),
            D_upper,
            K_prev,
            jnp.array(self.mu_floor, dtype=Q.dtype),
            jnp.array(self.mu_max_cap, dtype=Q.dtype),
            jnp.array(self.max_dual_iters, dtype=jnp.int32),
            jnp.array(self.stability_tol, dtype=Q.dtype),
        )

        return LAGLQState(
            K=K_ext0[:du, :],
            K_ext=K_ext0,
            V=V,
            V_cur=V,
            S=S,
            logdet_V=logdet(V),
            Q=Q,
            R=R,
            lam=jnp.array(self.lam, dtype=Q.dtype),
            A0=A0,
            B0=B0,
            A_hat=A0,
            B_hat=B0,
            beta=beta0,
            mu=mu0,
            eps=jnp.array(self.eps_scale, dtype=Q.dtype),
            D_upper=D_upper,
            horizon=jnp.array(float(self.horizon), dtype=Q.dtype),
            delta=jnp.array(self.delta, dtype=Q.dtype),
            noise_sigma=jnp.array(self.noise_sigma, dtype=Q.dtype),
            prior_radius=jnp.array(self.prior_radius, dtype=Q.dtype),
            eps_scale=jnp.array(self.eps_scale, dtype=Q.dtype),
            beta_scale=jnp.array(self.beta_scale, dtype=Q.dtype),
            mu_floor=jnp.array(self.mu_floor, dtype=Q.dtype),
            mu_max_cap=jnp.array(self.mu_max_cap, dtype=Q.dtype),
            max_dual_iters=jnp.array(self.max_dual_iters, dtype=jnp.int32),
            stability_tol=jnp.array(self.stability_tol, dtype=Q.dtype),
        )

    @staticmethod
    def get_action(
        x: Float[Array, "dx"],
        state: LAGLQState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], LAGLQState]:
        del key
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))
        return u, state

    @staticmethod
    def update(
        x: Float[Array, "dx"],
        u: Float[Array, "du"],
        x_next: Float[Array, "dx"],
        state: LAGLQState,
        t: int,
    ) -> LAGLQState:
        dx = x.shape[0]
        du = u.shape[0]

        xu = jnp.concatenate([x, u], axis=0)
        V_cur = state.V_cur + jnp.outer(xu, xu)
        S = state.S + jnp.outer(x_next, xu)
        logdet_V_cur = logdet(V_cur)
        do_update = logdet_V_cur > (state.logdet_V + jnp.log(2.0))

        def update_branch(_: None) -> LAGLQState:
            A_hat, B_hat = rls_estimate(V_cur, S, dx, state.A0, state.B0, state.lam)
            beta_t = LAGLQ._confidence_radius(
                V_cur,
                dx,
                du,
                state.lam,
                state.horizon,
                state.delta,
                state.noise_sigma,
                state.prior_radius,
                state.beta_scale,
                state.mu_floor,
            )
            eps_t = state.eps_scale / jnp.sqrt(t.astype(state.Q.dtype) + 1.0)
            K_ext, mu_t = LAGLQ._solve_policy(
                A_hat,
                B_hat,
                state.Q,
                state.R,
                V_cur,
                beta_t,
                eps_t,
                state.D_upper,
                state.K_ext,
                state.mu_floor,
                state.mu_max_cap,
                state.max_dual_iters,
                state.stability_tol,
            )
            return state._replace(
                K=K_ext[:du, :],
                K_ext=K_ext,
                V=V_cur,
                V_cur=V_cur,
                S=S,
                logdet_V=logdet_V_cur,
                A_hat=A_hat,
                B_hat=B_hat,
                beta=beta_t,
                mu=mu_t,
                eps=eps_t,
            )

        def no_update_branch(_: None) -> LAGLQState:
            return state._replace(V_cur=V_cur, S=S)

        return jax.lax.cond(do_update, update_branch, no_update_branch, operand=None)
