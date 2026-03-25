"""Certainty Equivalence Control (CEC) with doubling-time updates.

Matches the ``CECPE`` estimator and controller-refresh schedule, but does not
inject persistent excitation noise into the applied control.
"""

import jax
from jaxtyping import Array, Float

from algorithms.cec_pe import CECPE, CECPEState


class CEC(CECPE):
    """CEC with a doubling-time schedule and deterministic actions."""

    def __init__(
        self,
        lam: float = 1.0,
        A0: Float[Array, "dx dx"] | None = None,
        B0: Float[Array, "dx du"] | None = None,
        solver: str = "sda",
    ):
        super().__init__(lam=lam, init_act_std=0.0, A0=A0, B0=B0, solver=solver)

    @staticmethod
    def get_action(
        x: Float[Array, "dx"],
        state: CECPEState,
        key: jax.Array,
    ) -> tuple[Float[Array, "du"], CECPEState]:
        """Deterministic action u = K @ x."""
        del key
        du = state.K.shape[0]
        u = (state.K @ x).reshape((du,))
        return u, state
