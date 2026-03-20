"""Certainty Equivalence Control with Persistent Excitation (CEC+PE).

Uses RLS estimation with a doubling-time update schedule (t = 1, 2, 4, 8, ...)
and decaying action noise for exploration.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

from algorithms.base import LQRAlgorithm
from lqr_utils import make_cost_matrix, dlqr_joint, rls_estimate


class CECPEState(NamedTuple):
    """Algorithm state for CEC+PE.

    K:           Float[Array, "du dx"]     current controller gain
    V:           Float[Array, "dxu dxu"]   committed RLS design matrix
    V_cur:       Float[Array, "dxu dxu"]   running RLS design matrix
    S:           Float[Array, "dx dxu"]    RLS cross-correlation
    H:           Float[Array, "dxu dxu"]   joint cost matrix (cached)
    next_update: int                        next time step to trigger update
    act_std:     Float[Array, ""]          current exploration noise std
    """
    K: jax.Array
    V: jax.Array
    V_cur: jax.Array
    S: jax.Array
    H: jax.Array
    next_update: jax.Array
    act_std: jax.Array


class CECPE(LQRAlgorithm):
    """CEC with persistent excitation and doubling-time schedule.

    Args:
        lam:          RLS regularization parameter.
        init_act_std: initial exploration noise standard deviation.
    """

    def __init__(self, lam: float = 1.0, init_act_std: float = 1.0):
        self.lam = lam
        self.init_act_std = init_act_std

    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> CECPEState:
        dxu = dx + du
        H = make_cost_matrix(Q, R)                          # (dxu, dxu)
        V = self.lam * jnp.eye(dxu, dtype=Q.dtype)          # (dxu, dxu)
        S = jnp.zeros((dx, dxu), dtype=Q.dtype)             # (dx, dxu)
        K = jnp.ones((du, dx), dtype=Q.dtype) * 0.01        # (du, dx)
        return CECPEState(
            K=K,
            V=V,
            V_cur=V,
            S=S,
            H=H,
            next_update=jnp.array(1, dtype=jnp.int32),
            act_std=jnp.array(self.init_act_std, dtype=Q.dtype),
        )

    @staticmethod
    def get_action(
        x: Float[Array, "dx"],
        state: CECPEState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], CECPEState]:
        """Action u = K @ x + exploration noise."""
        du = state.K.shape[0]
        eta = state.act_std * jax.random.normal(
            key, shape=(du,), dtype=x.dtype
        )  # (du,)
        u = (state.K @ x).reshape((du,)) + eta  # (du,)
        return u, state

    @staticmethod
    def update(
        x: Float[Array, "dx"],
        u: Float[Array, "du"],
        x_next: Float[Array, "dx"],
        state: CECPEState,
        t: int,
    ) -> CECPEState:
        """Accumulate RLS stats; recompute controller on doubling schedule."""
        dx = x.shape[0]

        # accumulate sufficient statistics
        xu = jnp.concatenate([x, u], axis=0)      # (dxu,)
        V_cur = state.V_cur + jnp.outer(xu, xu)    # (dxu, dxu)
        S = state.S + jnp.outer(x_next, xu)        # (dx, dxu)

        # doubling-time trigger: update at t = 1, 2, 4, 8, ...
        do_update = t.astype(jnp.int32) == state.next_update

        def update_branch(args):
            K, V, next_update, act_std = args

            A_hat, B_hat = rls_estimate(V_cur, S, dx)  # (dx, dx), (dx, du)
            K_new, _ = dlqr_joint(A_hat, B_hat, state.H)  # (du, dx)

            next_update_new = next_update * 2
            act_std_new = 1.0 / jnp.sqrt(t.astype(x.dtype) + 1.0)
            return (K_new, V_cur, next_update_new, act_std_new)

        def no_update_branch(args):
            return args

        K, V, next_update, act_std = jax.lax.cond(
            do_update,
            update_branch,
            no_update_branch,
            operand=(state.K, state.V, state.next_update, state.act_std),
        )

        return state._replace(
            K=K, V=V, V_cur=V_cur, S=S,
            next_update=next_update, act_std=act_std,
        )
