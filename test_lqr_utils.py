"""Tests for dlqr solvers in lqr_utils, focusing on the Schur solver."""

import jax
import jax.numpy as jnp

from lqr_utils import dlqr_joint, make_cost_matrix

jax.config.update("jax_enable_x64", True)


def dare_residual(A, B, H, K, P):
    """DARE residual: P should equal Q + A'PA - (A'PB+S)(R+B'PB)^{-1}(B'PA+S')."""
    n = A.shape[0]
    Q = 0.5 * (H[:n, :n] + H[:n, :n].T)
    S = H[:n, n:]
    R = 0.5 * (H[n:, n:] + H[n:, n:].T)
    G = R + B.T @ P @ B
    F = A.T @ P @ B + S
    P_rhs = Q + A.T @ P @ A - F @ jax.scipy.linalg.solve(G, F.T, assume_a="sym")
    return float(jnp.max(jnp.abs(P - P_rhs)))


def _double_integrator():
    A = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    B = jnp.array([[0.0], [1.0]])
    Q = jnp.eye(2)
    R = jnp.eye(1)
    H = make_cost_matrix(Q, R)
    return A, B, H


def _random_system(seed: int, dx: int = 4, du: int = 2):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    A = 0.9 * jax.random.normal(k1, (dx, dx))
    B = jax.random.normal(k2, (dx, du))
    Q_half = jax.random.normal(k3, (dx, dx))
    Q = Q_half.T @ Q_half + 0.1 * jnp.eye(dx)
    R_half = jax.random.normal(k4, (du, du))
    R = R_half.T @ R_half + 0.1 * jnp.eye(du)
    S = 0.1 * jax.random.normal(k5, (dx, du))
    H = jnp.block([[Q, S], [S.T, R]])
    return A, B, H


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_double_integrator():
    A, B, H = _double_integrator()
    K, P = dlqr_joint(A, B, H, solver="schur")

    res = dare_residual(A, B, H, K, P)
    check(res < 1e-10, f"DARE residual too large: {res}")

    Acl = A + B @ K
    eigs = jnp.abs(jnp.linalg.eigvals(Acl))
    check(bool(jnp.all(eigs < 1.0)), f"Closed-loop unstable, eigs: {eigs}")

    eigvals = jnp.linalg.eigvalsh(P)
    check(bool(jnp.all(eigvals > 0)), f"P not positive definite, eigvals: {eigvals}")

    print("  double_integrator: PASS")


def test_random_systems():
    for seed in range(5):
        A, B, H = _random_system(seed)
        K, P = dlqr_joint(A, B, H, solver="schur")

        res = dare_residual(A, B, H, K, P)
        check(res < 1e-8, f"seed={seed}: DARE residual too large: {res}")

        Acl = A + B @ K
        eigs = jnp.abs(jnp.linalg.eigvals(Acl))
        check(bool(jnp.all(eigs < 1.0)), f"seed={seed}: unstable, eigs: {eigs}")

        print(f"  random seed={seed}: PASS")


def test_agrees_with_sda():
    for seed in range(5):
        A, B, H = _random_system(seed)
        K_schur, P_schur = dlqr_joint(A, B, H, solver="schur")
        K_sda, P_sda = dlqr_joint(A, B, H, solver="sda")
        check(bool(jnp.allclose(P_schur, P_sda, atol=1e-8)),
              f"seed={seed}: P mismatch, max diff={float(jnp.max(jnp.abs(P_schur - P_sda)))}")
        check(bool(jnp.allclose(K_schur, K_sda, atol=1e-8)),
              f"seed={seed}: K mismatch, max diff={float(jnp.max(jnp.abs(K_schur - K_sda)))}")
        print(f"  agrees_with_sda seed={seed}: PASS")


def test_agrees_with_riccati():
    for seed in range(5):
        A, B, H = _random_system(seed)
        K_schur, P_schur = dlqr_joint(A, B, H, solver="schur")
        K_ric, P_ric = dlqr_joint(A, B, H, solver="riccati")
        check(bool(jnp.allclose(P_schur, P_ric, atol=1e-6)),
              f"seed={seed}: P mismatch vs riccati, max diff={float(jnp.max(jnp.abs(P_schur - P_ric)))}")
        check(bool(jnp.allclose(K_schur, K_ric, atol=1e-6)),
              f"seed={seed}: K mismatch vs riccati, max diff={float(jnp.max(jnp.abs(K_schur - K_ric)))}")
        print(f"  agrees_with_riccati seed={seed}: PASS")


def test_cross_term():
    A = jnp.array([[1.0, 0.1], [0.0, 1.0]])
    B = jnp.array([[0.0], [1.0]])
    Q = jnp.eye(2)
    R = jnp.eye(1)
    S = jnp.array([[0.1], [0.05]])
    H = jnp.block([[Q, S], [S.T, R]])

    K_schur, P_schur = dlqr_joint(A, B, H, solver="schur")
    K_sda, P_sda = dlqr_joint(A, B, H, solver="sda")

    res = dare_residual(A, B, H, K_schur, P_schur)
    check(res < 1e-10, f"DARE residual with cross term: {res}")
    check(bool(jnp.allclose(P_schur, P_sda, atol=1e-8)),
          f"P mismatch with cross term")
    check(bool(jnp.allclose(K_schur, K_sda, atol=1e-8)),
          f"K mismatch with cross term")
    print("  cross_term: PASS")


if __name__ == "__main__":
    tests = [test_double_integrator, test_random_systems, test_agrees_with_sda,
             test_agrees_with_riccati, test_cross_term]
    for t in tests:
        print(f"{t.__name__}:")
        t()
    print("\nAll tests passed!")
