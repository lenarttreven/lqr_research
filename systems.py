"""Library of LQR benchmark systems.

Includes discretized integrator chains and classic physical systems
(cart-pole, DC motor, mass-spring-damper, Boeing 747 lateral, etc.)
commonly used in the adaptive LQR literature.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass(frozen=True)
class LQRSystem:
    """A discrete-time LQR problem instance with initial prior."""

    name: str
    A_star: Float[Array, "dx dx"]
    B_star: Float[Array, "dx du"]
    Q: Float[Array, "dx dx"]
    R: Float[Array, "du du"]
    A0: Float[Array, "dx dx"]
    B0: Float[Array, "dx du"]
    x0: Float[Array, "dx"]
    noise_sigma: float = 0.1


def _integrator_chain(d_x: int, d_u: int, dt: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build (A, B) for a discretized integrator chain.

    A is upper-triangular with A[i,j] = dt^(j-i) / (j-i)!  for j >= i.
    B[:, k] corresponds to input entering at state index (d_x - d_u + k).
    """
    A = jnp.zeros((d_x, d_x))
    for i in range(d_x):
        for j in range(i, d_x):
            order = j - i
            A = A.at[i, j].set(dt**order / _factorial(order))

    B = jnp.zeros((d_x, d_u))
    for k in range(d_u):
        entry_idx = d_x - d_u + k
        for i in range(d_x):
            order = entry_idx - i + 1
            if order > 0:
                B = B.at[i, k].set(dt**order / _factorial(order))
    return A, B


def _factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def _make_cost_matrices(d_x: int, d_u: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Cost matrices: Q has geometrically decreasing weights, R = 0.1 * I."""
    # Q weights: 10, ~3.16, 1, 0.316, 0.1, ... (geometric from 10 down)
    log_weights = jnp.linspace(jnp.log(10.0), jnp.log(0.1), d_x)
    Q = jnp.diag(jnp.exp(log_weights))
    R = 0.1 * jnp.eye(d_u)
    return Q, R


def make_system(
    d_x: int,
    d_u: int,
    dt: float,
    key: jax.Array,
    perturbation: float = 0.05,
    name: str | None = None,
) -> LQRSystem:
    """Create a single benchmark system."""
    A_star, B_star = _integrator_chain(d_x, d_u, dt)
    Q, R = _make_cost_matrices(d_x, d_u)

    key_a, key_b = jax.random.split(key)
    A0 = A_star + perturbation * jax.random.normal(key_a, shape=A_star.shape)
    B0 = B_star + perturbation * jax.random.normal(key_b, shape=B_star.shape)

    if name is None:
        name = f"integrator_dx{d_x}_du{d_u}"

    # Initial state: unit displacement in first state (position), rest zero.
    x0 = jnp.zeros(d_x).at[0].set(1.0)

    return LQRSystem(
        name=name,
        A_star=A_star,
        B_star=B_star,
        Q=Q,
        R=R,
        A0=A0,
        B0=B0,
        x0=x0,
    )


def _discretize(Ac: jnp.ndarray, Bc: jnp.ndarray, dt: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Zero-order hold discretization of a continuous-time system."""
    d_x = Ac.shape[0]
    # Matrix exponential method: build [[Ac, Bc], [0, 0]] * dt, exponentiate.
    d_u = Bc.shape[1]
    top = jnp.concatenate([Ac, Bc], axis=1)
    bottom = jnp.zeros((d_u, d_x + d_u))
    M = jnp.concatenate([top, bottom], axis=0) * dt
    eM = jax.scipy.linalg.expm(M)
    Ad = eM[:d_x, :d_x]
    Bd = eM[:d_x, d_x:]
    return Ad, Bd


def _perturb_prior(
    A_star: jnp.ndarray,
    B_star: jnp.ndarray,
    key: jax.Array,
    perturbation: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a noisy initial prior (A0, B0)."""
    key_a, key_b = jax.random.split(key)
    A0 = A_star + perturbation * jax.random.normal(key_a, shape=A_star.shape)
    B0 = B_star + perturbation * jax.random.normal(key_b, shape=B_star.shape)
    return A0, B0


# ---------------------------------------------------------------------------
# Classic physical systems
# ---------------------------------------------------------------------------


def make_cart_pole(key: jax.Array, dt: float = 0.02, perturbation: float = 0.05) -> LQRSystem:
    """Linearized cart-pole (inverted pendulum on cart).

    State: [x, x_dot, theta, theta_dot], Input: [force].
    Linearized about the upright equilibrium.  d_x=4, d_u=1.
    """
    g = 9.81
    m_c = 1.0   # cart mass
    m_p = 0.1   # pole mass
    l = 0.5     # pole half-length
    total = m_c + m_p

    Ac = jnp.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -m_p * g / total, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, total * g / (total * l), 0.0],
    ])
    Bc = jnp.array([
        [0.0],
        [1.0 / total],
        [0.0],
        [-1.0 / (total * l)],
    ])

    A_star, B_star = _discretize(Ac, Bc, dt)
    Q = jnp.diag(jnp.array([1.0, 0.1, 10.0, 0.1]))
    R = jnp.array([[0.01]])
    # Start displaced: cart at 0.5m, pole tilted 5 deg (~0.087 rad)
    x0 = jnp.array([0.5, 0.0, 0.087, 0.0])
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(name="cart_pole", A_star=A_star, B_star=B_star, Q=Q, R=R, A0=A0, B0=B0, x0=x0)


def make_dc_motor(key: jax.Array, dt: float = 0.01, perturbation: float = 0.05) -> LQRSystem:
    """DC motor with armature dynamics.

    State: [angle, angular_vel, current], Input: [voltage].  d_x=3, d_u=1.
    """
    J = 0.01    # rotor inertia
    b = 0.1     # viscous friction
    K = 0.01    # motor constant (= back-emf constant)
    R_a = 1.0   # armature resistance
    L = 0.5     # armature inductance

    Ac = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, -b / J, K / J],
        [0.0, -K / L, -R_a / L],
    ])
    Bc = jnp.array([
        [0.0],
        [0.0],
        [1.0 / L],
    ])

    A_star, B_star = _discretize(Ac, Bc, dt)
    Q = jnp.diag(jnp.array([10.0, 1.0, 0.1]))
    R_cost = jnp.array([[0.1]])
    # Start at 1 rad angle, zero velocity, zero current
    x0 = jnp.array([1.0, 0.0, 0.0])
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(name="dc_motor", A_star=A_star, B_star=B_star, Q=Q, R=R_cost, A0=A0, B0=B0, x0=x0)


def make_mass_spring_damper(
    n_masses: int,
    key: jax.Array,
    dt: float = 0.05,
    perturbation: float = 0.05,
) -> LQRSystem:
    """Coupled mass-spring-damper chain (Deán & Mania et al. style).

    n_masses masses coupled by springs and dampers in series, with the first
    mass actuated.  d_x = 2*n_masses, d_u = 1.
    """
    n = n_masses
    k_s = 1.0   # spring constant
    c = 0.1     # damping
    m = 1.0     # mass

    # Stiffness and damping matrices for the chain
    K_mat = jnp.zeros((n, n))
    C_mat = jnp.zeros((n, n))
    for i in range(n):
        K_mat = K_mat.at[i, i].set(2.0 * k_s if i > 0 and i < n - 1 else k_s)
        C_mat = C_mat.at[i, i].set(2.0 * c if i > 0 and i < n - 1 else c)
        if i > 0:
            K_mat = K_mat.at[i, i - 1].set(-k_s)
            C_mat = C_mat.at[i, i - 1].set(-c)
        if i < n - 1:
            K_mat = K_mat.at[i, i + 1].set(-k_s)
            C_mat = C_mat.at[i, i + 1].set(-c)

    # Continuous: x = [pos; vel], x_dot = [[0, I], [-K/m, -C/m]] x + [[0],[1/m e_1]] u
    Z = jnp.zeros((n, n))
    I_n = jnp.eye(n)
    Ac = jnp.block([
        [Z, I_n],
        [-K_mat / m, -C_mat / m],
    ])
    Bc_top = jnp.zeros((n, 1))
    Bc_bot = jnp.zeros((n, 1)).at[0, 0].set(1.0 / m)
    Bc = jnp.concatenate([Bc_top, Bc_bot], axis=0)

    A_star, B_star = _discretize(Ac, Bc, dt)
    d_x = 2 * n
    Q = jnp.eye(d_x)
    R_cost = jnp.array([[0.1]])
    # First mass displaced by 1.0, all others at rest
    x0 = jnp.zeros(d_x).at[0].set(1.0)
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(
        name=f"mass_spring_damper_{n}",
        A_star=A_star, B_star=B_star, Q=Q, R=R_cost, A0=A0, B0=B0, x0=x0,
    )


def make_boeing747_lateral(key: jax.Array, dt: float = 0.05, perturbation: float = 0.02) -> LQRSystem:
    """Boeing 747 lateral/directional dynamics (Skogestad & Postlethwaite).

    State: [sideslip, yaw_rate, roll_rate, roll_angle], Input: [rudder, aileron].
    d_x=4, d_u=2.
    """
    Ac = jnp.array([
        [-0.0558, -0.9968,  0.0802,  0.0415],
        [ 0.598,  -0.115,  -0.0318,  0.0],
        [-3.05,    0.388,  -0.4650,  0.0],
        [ 0.0,    0.0805,  1.0,      0.0],
    ])
    Bc = jnp.array([
        [ 0.00729,  0.0],
        [-0.475,    0.00775],
        [ 0.153,    0.143],
        [ 0.0,      0.0],
    ])

    A_star, B_star = _discretize(Ac, Bc, dt)
    Q = jnp.diag(jnp.array([1.0, 1.0, 1.0, 10.0]))
    R_cost = 0.1 * jnp.eye(2)
    # Small sideslip (2 deg) and roll angle (5 deg) perturbation
    x0 = jnp.array([0.035, 0.0, 0.0, 0.087])
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(
        name="boeing747_lateral", A_star=A_star, B_star=B_star,
        Q=Q, R=R_cost, A0=A0, B0=B0, x0=x0,
    )


def make_double_integrator(key: jax.Array, dt: float = 0.1, perturbation: float = 0.05) -> LQRSystem:
    """Classic double integrator (position + velocity, single force input).

    d_x=2, d_u=1.  The simplest nontrivial LQR benchmark.
    """
    Ac = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Bc = jnp.array([[0.0], [1.0]])
    A_star, B_star = _discretize(Ac, Bc, dt)
    Q = jnp.diag(jnp.array([1.0, 0.1]))
    R_cost = jnp.array([[0.01]])
    # Start at position 1.0, zero velocity
    x0 = jnp.array([1.0, 0.0])
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(
        name="double_integrator", A_star=A_star, B_star=B_star,
        Q=Q, R=R_cost, A0=A0, B0=B0, x0=x0,
    )


def make_aircraft_pitch(key: jax.Array, dt: float = 0.05, perturbation: float = 0.05) -> LQRSystem:
    """Longitudinal short-period aircraft pitch dynamics.

    State: [angle_of_attack, pitch_rate, pitch_angle], Input: [elevator].
    d_x=3, d_u=1.  From Stevens & Lewis.
    """
    Ac = jnp.array([
        [-0.313,  56.7, 0.0],
        [-0.0139, -0.426, 0.0],
        [ 0.0,    56.7, 0.0],
    ])
    Bc = jnp.array([
        [ 0.232],
        [ 0.0203],
        [ 0.0],
    ])

    A_star, B_star = _discretize(Ac, Bc, dt)
    Q = jnp.diag(jnp.array([1.0, 1.0, 10.0]))
    R_cost = jnp.array([[0.1]])
    # Small angle of attack (2 deg) and pitch angle (5 deg) perturbation
    x0 = jnp.array([0.035, 0.0, 0.087])
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(
        name="aircraft_pitch", A_star=A_star, B_star=B_star,
        Q=Q, R=R_cost, A0=A0, B0=B0, x0=x0,
    )


def make_magnetic_levitation(key: jax.Array, dt: float = 0.01, perturbation: float = 0.05) -> LQRSystem:
    """Linearized magnetic levitation (ball suspension).

    State: [position, velocity, current], Input: [voltage].  d_x=3, d_u=1.
    Linearized about the operating point where the ball hovers.
    """
    m = 0.05     # ball mass [kg]
    g = 9.81
    R_coil = 1.0  # coil resistance
    L = 0.01     # coil inductance
    C_mag = 1e-4  # magnetic force constant
    x0_eq = 0.01  # equilibrium gap
    i0_eq = jnp.sqrt(m * g * x0_eq**2 / C_mag)  # equilibrium current

    a21 = 2.0 * C_mag * i0_eq**2 / (m * x0_eq**3)
    a23 = -2.0 * C_mag * i0_eq / (m * x0_eq**2)

    Ac = jnp.array([
        [0.0, 1.0, 0.0],
        [a21, 0.0, a23],
        [0.0, 0.0, -R_coil / L],
    ])
    Bc = jnp.array([
        [0.0],
        [0.0],
        [1.0 / L],
    ])

    A_star, B_star = _discretize(Ac, Bc, dt)
    Q = jnp.diag(jnp.array([100.0, 1.0, 0.1]))
    R_cost = jnp.array([[0.01]])
    # Ball displaced 1mm from equilibrium gap, at rest
    x0 = jnp.array([0.001, 0.0, 0.0])
    A0, B0 = _perturb_prior(A_star, B_star, key, perturbation)
    return LQRSystem(
        name="magnetic_levitation", A_star=A_star, B_star=B_star,
        Q=Q, R=R_cost, A0=A0, B0=B0, x0=x0,
    )


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def get_integrator_systems(key: jax.Array, perturbation: float = 0.01) -> list[LQRSystem]:
    """Return 10 integrator-chain systems with state dims 2, 4, 6, ..., 20."""
    dt = 0.1
    dims = [
        (2, 1),
        (4, 1),
        (6, 1),
        (8, 2),
        (10, 2),
        (12, 2),
        (14, 3),
        (16, 3),
        (18, 4),
        (20, 4),
    ]

    keys = jax.random.split(key, len(dims))
    systems = []
    for k, (d_x, d_u) in zip(keys, dims):
        systems.append(make_system(d_x, d_u, dt, k, perturbation=perturbation))
    return systems


def get_physical_systems(key: jax.Array, perturbation: float = 0.05) -> list[LQRSystem]:
    """Return a suite of classic physical LQR benchmark systems."""
    keys = jax.random.split(key, 7)
    return [
        make_double_integrator(keys[0], perturbation=perturbation),
        make_dc_motor(keys[1], perturbation=perturbation),
        make_cart_pole(keys[2], perturbation=perturbation),
        make_aircraft_pitch(keys[3], perturbation=perturbation),
        make_boeing747_lateral(keys[4], perturbation=perturbation),
        make_mass_spring_damper(3, keys[5], perturbation=perturbation),
        make_magnetic_levitation(keys[6], perturbation=perturbation),
    ]


def get_benchmark_systems(key: jax.Array, perturbation: float = 0.05) -> list[LQRSystem]:
    """Return all benchmark systems (physical + integrator chains)."""
    key1, key2 = jax.random.split(key)
    return get_physical_systems(key1, perturbation) + get_integrator_systems(key2, perturbation)
