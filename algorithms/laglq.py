from typing import NamedTuple

import jax
import jax.numpy as jnp

from algorithms.base import LQRAlgorithm
from lqr_utils import (
    dlqr_joint,
    logdet,
    rls_estimate,
    steady_state_covariance,
)


def _symmetrize(M: jax.Array) -> jax.Array:
    return 0.5 * (M + M.T)


def _spectral_radius(A: jax.Array) -> jax.Array:
    eigvals = jnp.linalg.eigvals(A)
    return jnp.max(jnp.abs(eigvals))


def _constraint_blocks(V, beta, dx, du, penalty_aux=1.0):
    dxu = dx + du
    V_inv = jax.scipy.linalg.solve(V, jnp.eye(dxu), assume_a="sym")
    V_inv = _symmetrize(V_inv)

    beta2 = beta**2
    Q_g = -beta2 * V_inv[:dx, :dx]

    N_g = jnp.zeros((du + dx, dx))
    N_g = N_g.at[:du, :].set(-beta2 * V_inv[dx:, :dx])

    R_g = jnp.zeros((du + dx, du + dx))
    R_g = R_g.at[:du, :du].set(-beta2 * V_inv[dx:, dx:])
    R_g = R_g.at[du:, du:].set(penalty_aux * jnp.eye(dx))

    top = jnp.concatenate([Q_g, N_g.T], axis=1)
    bottom = jnp.concatenate([N_g, R_g], axis=1)
    return _symmetrize(jnp.concatenate([top, bottom], axis=0))


def _lagrangian_cost_matrix(Q, R, V, beta, mu, penalty_aux=1.0):
    dx, du = Q.shape[0], R.shape[0]
    C_g = _constraint_blocks(V, beta, dx, du, penalty_aux)

    R_base = jnp.zeros((du + dx, du + dx))
    R_base = R_base.at[:du, :du].set(R)

    Q_mu = Q + mu * C_g[:dx, :dx]
    N_mu = mu * C_g[dx:, :dx]
    R_mu = R_base + mu * C_g[dx:, dx:]

    top = jnp.concatenate([Q_mu, N_mu.T], axis=1)
    bottom = jnp.concatenate([N_mu, R_mu], axis=1)
    return _symmetrize(jnp.concatenate([top, bottom], axis=0)), C_g


class _DualEval(NamedTuple):
    K_ext: jax.Array
    grad: jax.Array
    valid: jax.Array


class LAGLQState(NamedTuple):
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
    beta: jax.Array  # now fixed!
    mu: jax.Array
    eps: jax.Array
    mu_floor: jax.Array
    mu_max_cap: jax.Array
    max_dual_iters: jax.Array
    stability_tol: jax.Array
    penalty_aux: jax.Array


class LAGLQ(LQRAlgorithm):

    def __init__(
        self,
        lam=1.0,
        beta=1.0,
        eps=1e-2,
        mu_floor=1e-6,
        mu_max_cap=1e6,
        max_dual_iters=30,
        stability_tol=1e-6,
        penalty_aux=1.0,
        A0=None,
        B0=None,
        solver: str = "sda",
    ):
        self.lam = lam
        self.beta = beta
        self.eps = eps
        self.mu_floor = mu_floor
        self.mu_max_cap = mu_max_cap
        self.max_dual_iters = max_dual_iters
        self.stability_tol = stability_tol
        self.penalty_aux = penalty_aux
        self.A0 = A0
        self.B0 = B0
        self.solver = solver

    @staticmethod
    def _dual_eval(A, B, Q, R, V, beta, mu, stability_tol, penalty_aux=1.0, solver="sda"):
        dx = A.shape[0]
        B_tilde = jnp.concatenate([B, jnp.eye(dx)], axis=1)

        H_mu, C_g = _lagrangian_cost_matrix(Q, R, V, beta, mu, penalty_aux)
        K_ext, _ = dlqr_joint(A, B_tilde, H_mu, solver=solver)

        A_cl = A + B_tilde @ K_ext

        M_g = C_g[:dx, :dx] + C_g[:dx, dx:] @ K_ext + K_ext.T @ C_g[dx:, :dx]
        M_g = _symmetrize(M_g + K_ext.T @ C_g[dx:, dx:] @ K_ext)

        G = steady_state_covariance(A_cl.T, M_g)
        grad = jnp.trace(G)

        rho = _spectral_radius(A_cl)
        valid = jnp.isfinite(grad) & (rho < 1.0 - stability_tol)

        return _DualEval(K_ext, grad, valid)

    @staticmethod
    def _solve_policy(A, B, Q, R, V, beta, eps, prev, mu_max, iters, stability_tol, penalty_aux=1.0, solver="sda"):

        sol0 = LAGLQ._dual_eval(A, B, Q, R, V, beta, 0.0, stability_tol, penalty_aux, solver=solver)

        # Bisection state: (i, mu_l, mu_r, K_r)
        init_state = (
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0),
            jnp.array(mu_max),
            prev,
        )

        def body_fn(carry):
            i, mu_l, mu_r, K_r = carry
            mu = 0.5 * (mu_l + mu_r)
            sol = LAGLQ._dual_eval(A, B, Q, R, V, beta, mu, stability_tol, penalty_aux, solver=solver)

            # If invalid: shrink upper bound
            # If valid and grad > 0: raise lower bound
            # If valid and grad <= 0: shrink upper bound, update K_r
            new_mu_l = jnp.where(sol.valid & (sol.grad > 0), mu, mu_l)
            new_mu_r = jnp.where(sol.valid & (sol.grad > 0), mu_r, mu)
            new_K_r = jnp.where(sol.valid & (sol.grad <= 0), sol.K_ext, K_r)

            return (i + 1, new_mu_l, new_mu_r, new_K_r)

        def cond_fn(carry):
            i, mu_l, mu_r, _ = carry
            return (mu_r - mu_l > eps) & (i < iters)

        _, mu_l, mu_r, K_r = jax.lax.while_loop(cond_fn, body_fn, init_state)

        # If unconstrained solution is already feasible, use it
        use_unconstrained = sol0.valid & (sol0.grad <= 0)
        K_out = jnp.where(use_unconstrained, sol0.K_ext, K_r)
        mu_out = jnp.where(use_unconstrained, 0.0, mu_r)

        return K_out, mu_out

    def init_state(self, dx, du, Q, R):
        beta = self.beta
        dxu = dx + du
        V = self.lam * jnp.eye(dxu)
        S = jnp.zeros((dx, dxu))

        A0 = self.A0 if self.A0 is not None else jnp.zeros((dx, dx))
        B0 = self.B0 if self.B0 is not None else jnp.zeros((dx, du))

        K_ext, mu = self._solve_policy(
            A0,
            B0,
            Q,
            R,
            V,
            beta,
            self.eps,
            jnp.zeros((du + dx, dx)),
            self.mu_max_cap,
            self.max_dual_iters,
            self.stability_tol,
            self.penalty_aux,
            solver=self.solver,
        )

        return LAGLQState(
            K=K_ext[:du],
            K_ext=K_ext,
            V=V,
            V_cur=V,
            S=S,
            logdet_V=logdet(V),
            Q=Q,
            R=R,
            lam=self.lam,
            A0=A0,
            B0=B0,
            A_hat=A0,
            B_hat=B0,
            beta=beta,
            mu=mu,
            eps=self.eps,
            mu_floor=self.mu_floor,
            mu_max_cap=self.mu_max_cap,
            max_dual_iters=self.max_dual_iters,
            stability_tol=self.stability_tol,
            penalty_aux=self.penalty_aux,
        )

    def get_action(self, x, state, key):
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))
        return u, state

    def update(self, x, u, x_next, state, t):
        xu = jnp.concatenate([x, u])
        V_cur = state.V_cur + jnp.outer(xu, xu)
        S = state.S + jnp.outer(x_next, xu)
        logdet_V_cur = logdet(V_cur)

        do_update = logdet_V_cur > state.logdet_V + jnp.log(2.0)

        def update_branch(_):
            A_hat, B_hat = rls_estimate(
                V_cur, S, x.shape[0], state.A0, state.B0, state.lam
            )

            K_ext, mu = self._solve_policy(
                A_hat,
                B_hat,
                state.Q,
                state.R,
                V_cur,
                state.beta,
                state.eps,
                state.K_ext,
                state.mu_max_cap,
                state.max_dual_iters,
                state.stability_tol,
                state.penalty_aux,
                solver=self.solver,
            )

            return state._replace(
                K=K_ext[: u.shape[0]],
                K_ext=K_ext,
                V=V_cur,
                V_cur=V_cur,
                S=S,
                logdet_V=logdet_V_cur,
                A_hat=A_hat,
                B_hat=B_hat,
                mu=mu,
            )

        return jax.lax.cond(
            do_update,
            update_branch,
            lambda _: state._replace(V_cur=V_cur, S=S),
            operand=None,
        )
