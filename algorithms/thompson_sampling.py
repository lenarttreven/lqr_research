"""Thompson sampling for LQR.

Uses RLS estimation with determinant-doubling trigger.
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


class TSState(NamedTuple):
    """Algorithm state for Thompson sampling-LQR.

    K:           Float[Array, "du dx"]     current controller gain
    V:           Float[Array, "dxu dxu"]   committed RLS design matrix
    V_cur:       Float[Array, "dxu dxu"]   running RLS design matrix
    S:           Float[Array, "dx dxu"]    RLS cross-correlation
    logdet_V:    Float[Array, ""]          log-det of committed V
    H:           Float[Array, "dxu dxu"]   joint cost matrix (cached)
    beta:        Float[Array, ""]          optimism bonus scale
    lam:         Float[Array, ""]          regularization
    use_invsqrt: bool                      if True, use V^{-1/2} instead of V^{-1}
    A0:          Float[Array, "dx dx"]     initial estimate of A
    B0:          Float[Array, "dx du"]     initial estimate of B
    sample_key:  jax.Array                 PRNG key cached from get_action
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
    sample_key: jax.Array


class TS(LQRAlgorithm):
    """Thompson sampling-LQR with determinant-doubling trigger.

    Constructor args stored as class attributes, used by init_state.
    """

    def __init__(
        self,
        lam: float = 1.0,
        beta: float = 0.05,
        use_invsqrt: bool = False,
        A0: Float[Array, "dx dx"] | None = None,
        B0: Float[Array, "dx du"] | None = None,
    ):
        self.lam = lam
        self.beta = beta
        self.use_invsqrt = use_invsqrt
        self.A0 = A0
        self.B0 = B0

    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> TSState:
        dxu = dx + du
        H = make_cost_matrix(Q, R)  # (dxu, dxu)
        V = self.lam * jnp.eye(dxu, dtype=Q.dtype)  # (dxu, dxu)
        S = jnp.zeros((dx, dxu), dtype=Q.dtype)  # (dx, dxu)
        A0 = self.A0 if self.A0 is not None else jnp.zeros((dx, dx), dtype=Q.dtype)
        B0 = self.B0 if self.B0 is not None else jnp.zeros((dx, du), dtype=Q.dtype)

        # compute initial controller from (A0, B0)
        K, _ = dlqr_joint(A0, B0, H)

        return TSState(
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
            sample_key=jax.random.PRNGKey(0),
        )

    @staticmethod
    def get_action(
        x: Float[Array, "dx"],
        state: TSState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], TSState]:
        """Deterministic action u = K @ x (no exploration noise)."""
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))  # (du,)
        return u, state._replace(sample_key=key)

    @staticmethod
    def update(
        x: Float[Array, "dx"],
        u: Float[Array, "du"],
        x_next: Float[Array, "dx"],
        state: TSState,
        t: int,
    ) -> TSState:
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
            AB_hat = jnp.concatenate([A_hat, B_hat], axis=1)  # (dx, dxu)
            V_invsqrt = sym_invsqrt(V_cur)  # (dxu, dxu)
            eta = jax.random.normal(
                jax.random.fold_in(state.sample_key, t),
                shape=AB_hat.shape,
                dtype=AB_hat.dtype,
            )  # (dx, dxu)
            AB_ts = AB_hat + state.beta * (eta @ V_invsqrt)  # (dx, dxu)
            A_ts = AB_ts[:, :dx]  # (dx, dx)
            B_ts = AB_ts[:, dx:]  # (dx, du)

            K_new, _ = dlqr_joint(A_ts, B_ts, state.H)  # (du, dx)
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
