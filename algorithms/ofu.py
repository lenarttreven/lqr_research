"""Optimism in the Face of Uncertainty (OFU) for LQR.

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
)


class OFUState(NamedTuple):
    """Algorithm state for OFU-LQR.

    K:        Float[Array, "du dx"]     current controller gain
    V:        Float[Array, "dxu dxu"]   committed RLS design matrix
    V_cur:    Float[Array, "dxu dxu"]   running RLS design matrix
    S:        Float[Array, "dx dxu"]    RLS cross-correlation
    logdet_V: Float[Array, ""]          log-det of committed V
    H:        Float[Array, "dxu dxu"]   joint cost matrix (cached)
    beta:     Float[Array, ""]          optimism bonus scale
    lam:      Float[Array, ""]          regularization
    use_invsqrt: bool                   if True, use V^{-1/2} instead of V^{-1}
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


class OFU(LQRAlgorithm):
    """OFU-LQR with determinant-doubling trigger.

    Constructor args stored as class attributes, used by init_state.
    """

    def __init__(
        self,
        lam: float = 1.0,
        beta: float = 0.05,
        use_invsqrt: bool = False,
    ):
        self.lam = lam
        self.beta = beta
        self.use_invsqrt = use_invsqrt

    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> OFUState:
        dxu = dx + du
        H = make_cost_matrix(Q, R)                          # (dxu, dxu)
        V = self.lam * jnp.eye(dxu, dtype=Q.dtype)          # (dxu, dxu)
        S = jnp.zeros((dx, dxu), dtype=Q.dtype)             # (dx, dxu)
        K = jnp.ones((du, dx), dtype=Q.dtype) * 0.01        # (du, dx)
        return OFUState(
            K=K,
            V=V,
            V_cur=V,
            S=S,
            logdet_V=logdet(V),
            H=H,
            beta=jnp.array(self.beta, dtype=Q.dtype),
            lam=jnp.array(self.lam, dtype=Q.dtype),
            use_invsqrt=jnp.array(self.use_invsqrt),
        )

    @staticmethod
    def get_action(
        x: Float[Array, "dx"],
        state: OFUState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], OFUState]:
        """Deterministic action u = K @ x (no exploration noise in OFU)."""
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))  # (du,)
        return u, state

    @staticmethod
    def update(
        x: Float[Array, "dx"],
        u: Float[Array, "du"],
        x_next: Float[Array, "dx"],
        state: OFUState,
        t: int,
    ) -> OFUState:
        """Accumulate RLS stats; recompute controller when det doubles."""
        dx = x.shape[0]

        # accumulate sufficient statistics
        xu = jnp.concatenate([x, u], axis=0)      # (dxu,)
        V_cur = state.V_cur + jnp.outer(xu, xu)    # (dxu, dxu)
        S = state.S + jnp.outer(x_next, xu)        # (dx, dxu)

        # determinant-doubling trigger
        logdet_V_cur = logdet(V_cur)
        do_update = logdet_V_cur > (state.logdet_V + jnp.log(2.0))

        def update_branch(args):
            K, V, logdet_V = args

            A_hat, B_hat = rls_estimate(V_cur, S, dx)  # (dx, dx), (dx, du)

            # optimism matrix: V^{-1} or V^{-1/2}
            dxu = V_cur.shape[0]
            O_inv = jax.scipy.linalg.solve(
                V_cur, jnp.eye(dxu, dtype=V_cur.dtype), assume_a="sym"
            )  # (dxu, dxu)
            O_inv = 0.5 * (O_inv + O_inv.T)
            O_invsqrt = sym_invsqrt(V_cur)  # (dxu, dxu)
            O = jnp.where(state.use_invsqrt, O_invsqrt, O_inv)  # (dxu, dxu)

            H_optim = 0.5 * (state.H - state.beta * O
                             + (state.H - state.beta * O).T)  # (dxu, dxu)
            K_new, _ = dlqr_joint(A_hat, B_hat, H_optim)       # (du, dx)
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
