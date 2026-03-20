import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


# -- Type aliases --
# Matrices: Float[Array, "dx dx"], Float[Array, "dx du"], etc.
# Vectors:  Float[Array, "dx"], Float[Array, "du"]


@jax.jit
def steady_state_covariance(
    Acl: Float[Array, "dx dx"],
    W: Float[Array, "dx dx"],
    max_iters: int = 20000,
    tol: float = 1e-12,
) -> Float[Array, "dx dx"]:
    """Compute steady-state covariance Sigma = Acl @ Sigma @ Acl.T + W via iteration."""
    def cond_fun(state):
        i, Sigma, Sigma_next = state
        err = jnp.max(jnp.abs(Sigma_next - Sigma))
        return jnp.logical_and(i < max_iters, err > tol)

    def body_fun(state):
        i, Sigma, Sigma_next = state
        Sigma = Sigma_next
        Sigma_next = Acl @ Sigma @ Acl.T + W
        return (i + 1, Sigma, Sigma_next)

    Sigma0 = W
    Sigma1 = Acl @ Sigma0 @ Acl.T + W
    _, Sigma, _ = jax.lax.while_loop(cond_fun, body_fun, (0, Sigma0, Sigma1))
    return 0.5 * (Sigma + Sigma.T)


@jax.jit
def lqr_avg_stage_cost(
    A: Float[Array, "dx dx"],
    B: Float[Array, "dx du"],
    Q: Float[Array, "dx dx"],
    R: Float[Array, "du du"],
    K: Float[Array, "du dx"],
    sigma: float,
) -> Float[Array, ""]:
    """Average stage cost under gain K with process noise std sigma."""
    n = A.shape[0]
    W = (sigma**2) * jnp.eye(n, dtype=A.dtype)  # (dx, dx)
    Acl = A + B @ K  # (dx, dx)
    Sigma = steady_state_covariance(Acl, W)  # (dx, dx)
    return jnp.trace((Q + K.T @ R @ K) @ Sigma)


@jax.jit
def dlqr_joint(
    A: Float[Array, "dx dx"],
    B: Float[Array, "dx du"],
    H: Float[Array, "dxu dxu"],
    max_iters: int = 200,
    tol: float = 1e-8,
) -> tuple[Float[Array, "du dx"], Float[Array, "dx dx"]]:
    """Solve discrete LQR from joint cost matrix H = [[Q, S], [S^T, R]].

    Returns (K, P) where K is the optimal gain and P the value matrix.
    """
    n = A.shape[0]
    Q = 0.5 * (H[:n, :n] + H[:n, :n].T)  # (dx, dx)
    S = H[:n, n:]  # (dx, du)
    R = 0.5 * (H[n:, n:] + H[n:, n:].T)  # (du, du)

    def riccati_step(P):
        G = R + B.T @ P @ B  # (du, du)
        F = A.T @ P @ B + S  # (dx, du)
        Pn = Q + A.T @ P @ A - F @ jax.scipy.linalg.solve(G, F.T, assume_a="sym")
        return 0.5 * (Pn + Pn.T)  # (dx, dx)

    def cond_fun(state):
        i, P, Pn = state
        err = jnp.max(jnp.abs(Pn - P))
        return jnp.logical_and(i < max_iters, err > tol)

    def body_fun(state):
        i, P, Pn = state
        P = Pn
        Pn = riccati_step(P)
        return (i + 1, P, Pn)

    P0 = Q
    P1 = riccati_step(P0)
    _, P, _ = jax.lax.while_loop(cond_fun, body_fun, (0, P0, P1))

    G = R + B.T @ P @ B  # (du, du)
    K = -jax.scipy.linalg.solve(G, (B.T @ P @ A + S.T), assume_a="sym")  # (du, dx)
    return K, P


def make_cost_matrix(
    Q: Float[Array, "dx dx"],
    R: Float[Array, "du du"],
) -> Float[Array, "dxu dxu"]:
    """Build joint cost matrix H = [[Q, 0], [0, R]] of shape (dx+du, dx+du)."""
    dx = Q.shape[0]
    du = R.shape[0]
    zxu = jnp.zeros((dx, du), dtype=Q.dtype)  # (dx, du)
    zux = jnp.zeros((du, dx), dtype=Q.dtype)  # (du, dx)
    return jnp.block([[Q, zxu], [zux, R]])  # (dx+du, dx+du)


def sym_invsqrt(
    V: Float[Array, "n n"],
    eps: float = 1e-12,
) -> Float[Array, "n n"]:
    """Symmetric inverse square root of a PD matrix V."""
    evals, evecs = jnp.linalg.eigh(0.5 * (V + V.T))  # (n,), (n, n)
    evals = jnp.maximum(evals, eps)
    inv_sqrt = (evecs * (1.0 / jnp.sqrt(evals))) @ evecs.T  # (n, n)
    return 0.5 * (inv_sqrt + inv_sqrt.T)


def rls_estimate(
    V: Float[Array, "dxu dxu"],
    S: Float[Array, "dx dxu"],
    dx: int,
    A0: Float[Array, "dx dx"] | None = None,
    B0: Float[Array, "dx du"] | None = None,
    lam: float = 1.0,
) -> tuple[Float[Array, "dx dx"], Float[Array, "dx du"]]:
    """Compute (A_hat, B_hat) from RLS sufficient statistics V and S,
    centered around initial estimates (A0, B0) with regularization lam.

    AB_hat = (lam * AB0 + S) @ V^{-1}.
    """
    du = V.shape[0] - dx
    if A0 is None:
        A0 = jnp.zeros((dx, dx), dtype=V.dtype)  # (dx, dx)
    if B0 is None:
        B0 = jnp.zeros((dx, du), dtype=V.dtype)  # (dx, du)
    AB0 = jnp.concatenate([A0, B0], axis=1)  # (dx, dxu)
    RHS = lam * AB0 + S  # (dx, dxu)
    AB_hat = jax.scipy.linalg.solve(V.T, RHS.T, assume_a="sym").T  # (dx, dxu)
    A_hat = AB_hat[:, :dx]  # (dx, dx)
    B_hat = AB_hat[:, dx:]  # (dx, du)
    return A_hat, B_hat


def logdet(M: Float[Array, "n n"]) -> Float[Array, ""]:
    """Log-determinant of a matrix, returns -inf if non-positive."""
    sign, ld = jnp.linalg.slogdet(M)
    return jnp.where(sign > 0, ld, -jnp.inf)
