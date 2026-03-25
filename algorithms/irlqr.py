"""Optimism in the Face of Uncertainty (IRLQR) for LQR.

Uses RLS estimation with determinant-doubling trigger and an optimistic
cost matrix: H_optim = H - beta * V^{-1} (or V^{-1/2}).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from algorithms.base import LQRAlgorithm
from lqr_utils import (
    make_cost_matrix,
    dlqr_joint,
    rls_estimate,
    logdet,
    sym_invsqrt,
    sclip,
)


class IRLQRState(NamedTuple):
    """Algorithm state for IRLQR-LQR.

    K:           Float[Array, "du dx"]     current controller gain
    V:           Float[Array, "dxu dxu"]   committed RLS design matrix
    V_cur:       Float[Array, "dxu dxu"]   running RLS design matrix
    S:           Float[Array, "dx dxu"]    RLS cross-correlation
    logdet_V:    Float[Array, ""]          log-det of committed V
    H:           Float[Array, "dxu dxu"]   joint cost matrix (cached)
    beta:        Float[Array, ""]          optimism bonus scale
    lam:         Float[Array, ""]          regularization
    use_invsqrt: bool                      if True, use V^{-1/2} instead of ||V||^{1/2}*V^{-1}
    A0:          Float[Array, "dx dx"]     initial estimate of A
    B0:          Float[Array, "dx du"]     initial estimate of B
    c_lam:       Floar[Array, ""]          max eigenvalue for spectral clipping sclip
    """

    K: jax.Array
    V: jax.Array
    V_cur: jax.Array
    S: jax.Array
    logdet_V: jax.Array
    H: jax.Array
    beta: jax.Array
    lam: jax.Array
    use_invsqrt: jax.Array
    A0: jax.Array
    B0: jax.Array
    c_lam : jax.Array


class IRLQR(LQRAlgorithm):
    """IRLQR-LQR with determinant-doubling trigger.

    Constructor args stored as class attributes, used by init_state.
    """

    def __init__(
        self,
        lam: float = 1.0,
        beta: float = 0.05,
        use_invsqrt: bool = False,
        A0: Float[Array, "dx dx"] | None = None,
        B0: Float[Array, "dx du"] | None = None,
        solver: str = "sda",
        c_lam: float = 1,
    ):
        self.lam = lam
        self.beta = beta
        self.use_invsqrt = use_invsqrt
        self.A0 = A0
        self.B0 = B0
        self.solver = solver
        self.c_lam = c_lam

    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> IRLQRState:
        dxu = dx + du
        H = make_cost_matrix(Q, R)  # (dxu, dxu)
        V = self.lam * jnp.eye(dxu, dtype=Q.dtype)  # (dxu, dxu)
        S = jnp.zeros((dx, dxu), dtype=Q.dtype)  # (dx, dxu)
        A0 = self.A0 if self.A0 is not None else jnp.zeros((dx, dx), dtype=Q.dtype)
        B0 = self.B0 if self.B0 is not None else jnp.zeros((dx, du), dtype=Q.dtype)
        Heigs, _ = jnp.linalg.eigh(H)
        c_lam = Heigs[0]*0.95

        # compute initial controller from (A0, B0)
        K, _ = dlqr_joint(A0, B0, H, solver=self.solver)

        return IRLQRState(
            K=K,
            V=V,
            V_cur=V,
            S=S,
            logdet_V=logdet(V),
            H=H,
            beta=jnp.array(self.beta, dtype=Q.dtype),
            lam=jnp.array(self.lam, dtype=Q.dtype),
            use_invsqrt=jnp.array(self.use_invsqrt),
            A0=A0,
            B0=B0,
            c_lam=c_lam
        )

    def get_action(
        self,
        x: Float[Array, "dx"],
        state: IRLQRState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], IRLQRState]:
        """Deterministic action u = K @ x (no exploration noise in IRLQR)."""
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))  # (du,)
        return u, state

    def update(
        self,
        x: Float[Array, "dx"],
        u: Float[Array, "du"],
        x_next: Float[Array, "dx"],
        state: IRLQRState,
        t: int,
    ) -> IRLQRState:
        """Accumulate RLS stats; recompute controller when det doubles."""
        dx = x.shape[0]

        # accumulate sufficient statistics
        xu = jnp.concatenate([x, u], axis=0)  # (dxu,)
        V_cur = state.V_cur + jnp.outer(xu, xu)  # (dxu, dxu)
        S = state.S + jnp.outer(x_next, xu)  # (dx, dxu)

        # determinant-doubling trigger
        logdet_V_cur = logdet(V_cur)
        do_update = logdet_V_cur > (state.logdet_V + jnp.log(2.0))

        def update_branch(args):
            K, V, logdet_V = args

            A_hat, B_hat = rls_estimate(
                V_cur, S, dx, state.A0, state.B0, state.lam
            )  # (dx, dx), (dx, du)

            # optimism matrix: V^{-1} or V^{-1/2}
            dxu = V_cur.shape[0]
            O_inv = jax.scipy.linalg.solve(
                V_cur, jnp.eye(dxu, dtype=V_cur.dtype), assume_a="sym"
            )  # (dxu, dxu)
            O_inv = jnp.sqrt(jnp.linalg.norm(V_cur, ord=2)) * 0.5 * (O_inv + O_inv.T)
            O_invsqrt = sym_invsqrt(V_cur)  # (dxu, dxu)
            O = sclip(state.beta * jnp.where(state.use_invsqrt, O_invsqrt, O_inv), self.c_lam)  # (dxu, dxu)

            H_optim = 0.5 * (
                (state.H - O) + (state.H -  O).T
            )  # (dxu, dxu)
            K_new, _ = dlqr_joint(A_hat, B_hat, H_optim, solver=self.solver)  # (du, dx)
            return (K_new, V_cur, logdet_V_cur)

        def no_update_branch(args):
            return args

        K, V, logdet_V = jax.lax.cond(
            do_update,
            update_branch,
            no_update_branch,
            operand=(state.K, state.V, state.logdet_V),
        )

        return state._replace(K=K, V=V, V_cur=V_cur, S=S, logdet_V=logdet_V)
