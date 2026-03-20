import abc
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array

# Algorithm state is any JAX-compatible pytree (NamedTuple, dataclass, etc.)
AlgoState = Any


class LQRAlgorithm(abc.ABC):
    """Base class for single-trajectory LQR learning algorithms.

    Subclasses must implement three static methods that operate on a
    JAX-friendly pytree state, keeping everything jittable via jax.lax.scan.

    Convention:
        dx = state dimension
        du = input dimension
        dxu = dx + du
    """

    @abc.abstractmethod
    def init_state(
        self,
        dx: int,
        du: int,
        Q: Float[Array, "dx dx"],
        R: Float[Array, "du du"],
    ) -> AlgoState:
        """Create the initial algorithm state.

        Called once before the simulation loop (outside jax.lax.scan).
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def get_action(
        x: Float[Array, "dx"],       # current state
        state: AlgoState,             # algorithm state
        key: jax.Array,               # PRNG key for exploration
    ) -> tuple[Float[Array, "du"], AlgoState]:
        """Compute control action u given current state x.

        May update algorithm state (e.g. for stateful exploration).
        Returns (u, updated_state).
        """
        ...

    @staticmethod
    @abc.abstractmethod
    def update(
        x: Float[Array, "dx"],        # state before transition
        u: Float[Array, "du"],         # action taken
        x_next: Float[Array, "dx"],    # state after transition
        state: AlgoState,              # algorithm state
        t: int,                        # current time step
    ) -> AlgoState:
        """Update algorithm state after observing a transition (x, u) -> x_next.

        This is where RLS accumulation, trigger checking, and controller
        recomputation happen.
        """
        ...
