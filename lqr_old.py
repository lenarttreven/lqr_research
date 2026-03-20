import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial


# ---------- Steady-state baseline (also jittable) ----------


@jax.jit
def steady_state_covariance(Acl, W, max_iters=20000, tol=1e-12):
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
def lqr_avg_stage_cost(A, B, Q, R, K, sigma):
    n = A.shape[0]
    W = (sigma**2) * jnp.eye(n, dtype=A.dtype)
    Acl = A + B @ K
    Sigma = steady_state_covariance(Acl, W)
    return jnp.trace((Q + K.T @ R @ K) @ Sigma)


# ---------- LQR solver (as you had) ----------


@jax.jit
def dlqr_joint(A, B, H, max_iters=200, tol=1e-8):
    n = A.shape[0]
    Q = 0.5 * (H[:n, :n] + H[:n, :n].T)
    S = H[:n, n:]
    R = 0.5 * (H[n:, n:] + H[n:, n:].T)

    def riccati_step(P):
        G = R + B.T @ P @ B
        F = A.T @ P @ B + S
        Pn = Q + A.T @ P @ A - F @ jax.scipy.linalg.solve(G, F.T, assume_a="sym")
        return 0.5 * (Pn + Pn.T)

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

    G = R + B.T @ P @ B
    K = -jax.scipy.linalg.solve(G, (B.T @ P @ A + S.T), assume_a="sym")
    return K, P


# ---------- Fully jittable learning loop ----------


def make_cost_matrix(Q, R):
    dx = Q.shape[0]
    du = R.shape[0]
    zxu = jnp.zeros((dx, du), dtype=Q.dtype)
    zux = jnp.zeros((du, dx), dtype=Q.dtype)
    return jnp.block([[Q, zxu], [zux, R]])


def sym_invsqrt(V, eps=1e-12):
    # V is symmetric PD (should be if lam>0 and updates add outer products)
    evals, evecs = jnp.linalg.eigh(0.5 * (V + V.T))
    evals = jnp.maximum(evals, eps)
    inv_sqrt = (evecs * (1.0 / jnp.sqrt(evals))) @ evecs.T
    return 0.5 * (inv_sqrt + inv_sqrt.T)


@partial(jax.jit, static_argnames=("num_steps", "optimism"))
def learn_system_jit(
    a_star,
    b_star,
    q,
    r,
    key,
    lam,
    beta,
    num_steps,
    noise_sigma,
    x0=None,
    optimism="inv",
):
    dx, du = b_star.shape
    if x0 is None:
        x0 = jnp.zeros((dx,), dtype=a_star.dtype)

    H = make_cost_matrix(q, r)

    # Oracle baseline for regret
    k_star, _ = dlqr_joint(a_star, b_star, H)
    optimal_cost = lqr_avg_stage_cost(a_star, b_star, q, r, k_star, noise_sigma)

    # RLS-ish stats
    v_t = lam * jnp.eye(dx + du, dtype=a_star.dtype)
    v_t_cur = v_t
    s_t = jnp.zeros((dx, dx + du), dtype=a_star.dtype)

    # init controller (your stabilizing seed)
    k_hat = jnp.ones(shape=(du, dx)) * 0.1

    # precompute logdet thresholding (use slogdet for stability)
    def logdet(M):
        sign, ld = jnp.linalg.slogdet(M)
        # if sign<=0, treat as -inf to avoid updates; should not happen if v_t stays PD
        return jnp.where(sign > 0, ld, -jnp.inf)

    logdet_v = logdet(v_t)

    def one_step(carry, t):
        key, x, k_hat, v_t, v_t_cur, s_t, logdet_v = carry

        # noise
        key, sub = jax.random.split(key)
        w = noise_sigma * jax.random.normal(sub, shape=(dx,), dtype=a_star.dtype)

        # control and transition
        u = (k_hat @ x).reshape((du,))
        x_next = a_star @ x + b_star @ u + w

        xu = jnp.concatenate([x, u], axis=0)
        v_t_cur = v_t_cur + jnp.outer(xu, xu)
        s_t = s_t + jnp.outer(x_next, xu)

        # update condition: det(V_cur) > 2 det(V)  <=> logdet(V_cur) > logdet(V) + log 2
        logdet_v_cur = logdet(v_t_cur)
        do_update = logdet_v_cur > (logdet_v + jnp.log(2.0))

        def update_branch(args):
            k_hat, v_t, logdet_v = args
            v_t_new = v_t_cur
            logdet_v_new = logdet_v_cur

            # AB_hat = S * inv(V)  -> use solve: X V = S^T  => X = (solve(V^T, S^T))^T
            Vinv_S_T = jax.scipy.linalg.solve(
                v_t_new.T, s_t.T, assume_a="sym"
            )  # (dx+du, dx)
            a_b_hat = Vinv_S_T.T  # (dx, dx+du)
            a_hat = a_b_hat[:, :dx]
            b_hat = a_b_hat[:, dx:]

            # optimism term (your original): H - beta * inv(V)
            # compute inv(V) via solve against I (small matrices here)
            I = jnp.eye(dx + du, dtype=a_star.dtype)
            Vinv = jax.scipy.linalg.solve(v_t_new, I, assume_a="sym")

            # --- choose optimism matrix ---
            if optimism == "inv":
                I = jnp.eye(dx + du, dtype=a_star.dtype)
                Vinv = jax.scipy.linalg.solve(v_t_new, I, assume_a="sym")
                O = 0.5 * (Vinv + Vinv.T)
            elif optimism == "invsqrt":
                O = sym_invsqrt(v_t_new)
            else:
                raise ValueError("optimism must be 'inv' or 'invsqrt'")

            H_optim = 0.5 * (H - beta * O + (H - beta * O).T)
            k_new, _ = dlqr_joint(a_hat, b_hat, H_optim)
            return (k_new, v_t_new, logdet_v_new)

        def no_update_branch(args):
            return args

        k_hat, v_t, logdet_v = jax.lax.cond(
            do_update,
            update_branch,
            no_update_branch,
            operand=(k_hat, v_t, logdet_v),
        )

        # stage cost and regret
        cost = x.T @ q @ x + u.T @ r @ u
        regret = cost - optimal_cost

        carry_next = (key, x_next, k_hat, v_t, v_t_cur, s_t, logdet_v)
        outs = (cost, regret, do_update)
        return carry_next, outs

    init_carry = (key, x0, k_hat, v_t, v_t_cur, s_t, logdet_v)
    carry_final, (costs, regrets, updates) = jax.lax.scan(
        one_step,
        init_carry,
        xs=jnp.arange(num_steps),
    )

    return {
        "costs": costs,
        "regrets": regrets,
        "updates": updates,
        "optimal_cost": optimal_cost,
        "k_star": k_star,
        "k_hat_final": carry_final[2],
    }


@partial(jax.jit, static_argnames=("num_steps", "optimism"))
def run_many(
    a_star, b_star, q, r, keys, lam, beta, num_steps, noise_sigma, optimism="inv"
):
    return jax.vmap(
        lambda key: learn_system_jit(
            a_star=a_star,
            b_star=b_star,
            q=q,
            r=r,
            key=key,
            lam=lam,
            beta=beta,
            num_steps=num_steps,
            noise_sigma=noise_sigma,
            optimism=optimism,
        )
    )(keys)


@partial(jax.jit, static_argnames=("num_steps",))
def learn_system_doubling_jit(
    a_star,
    b_star,
    q,
    r,
    key,
    lam,
    num_steps,
    noise_sigma,
    x0=None,
):
    dx, du = b_star.shape
    if x0 is None:
        x0 = jnp.zeros((dx,), dtype=a_star.dtype)

    H = make_cost_matrix(q, r)

    # Oracle baseline for regret
    k_star, _ = dlqr_joint(a_star, b_star, H)
    optimal_cost = lqr_avg_stage_cost(a_star, b_star, q, r, k_star, noise_sigma)

    # RLS-ish stats
    v_t = lam * jnp.eye(dx + du, dtype=a_star.dtype)
    v_t_cur = v_t
    s_t = jnp.zeros((dx, dx + du), dtype=a_star.dtype)

    # init controller (your stabilizing seed)
    k_hat = jnp.ones(shape=(du, dx), dtype=a_star.dtype) * 0.1

    # doubling-trick schedule: update at t=1,2,4,8,... (scan index starts at 0)
    next_update = jnp.array(1, dtype=jnp.int32)

    def one_step(carry, t):
        key, x, k_hat, v_t, v_t_cur, s_t, next_update = carry

        # --- exploration noise in action: N(0, 1/sqrt(t+1)) ---
        key, sub_u = jax.random.split(key)
        # scalar std; safe for t=0
        act_std = 1.0 / jnp.sqrt(t.astype(a_star.dtype) + 1.0)
        eta = act_std * jax.random.normal(sub_u, shape=(du,), dtype=a_star.dtype)

        # environment/process noise
        key, sub_w = jax.random.split(key)
        w = noise_sigma * jax.random.normal(sub_w, shape=(dx,), dtype=a_star.dtype)

        # control and transition
        u = (k_hat @ x).reshape((du,)) + eta
        x_next = a_star @ x + b_star @ u + w

        # accumulate regression stats
        xu = jnp.concatenate([x, u], axis=0)
        v_t_cur = v_t_cur + jnp.outer(xu, xu)
        s_t = s_t + jnp.outer(x_next, xu)

        # scheduled update?
        do_update = t.astype(jnp.int32) == next_update

        def update_branch(args):
            k_hat, v_t, next_update = args

            v_t_new = v_t_cur  # "commit" the accumulated info

            # AB_hat = S * inv(V)  -> solve(V^T, S^T)^T
            Vinv_S_T = jax.scipy.linalg.solve(
                v_t_new.T, s_t.T, assume_a="sym"
            )  # (dx+du, dx)
            a_b_hat = Vinv_S_T.T  # (dx, dx+du)
            a_hat = a_b_hat[:, :dx]
            b_hat = a_b_hat[:, dx:]

            # no optimism: use plain H
            k_new, _ = dlqr_joint(a_hat, b_hat, H)

            # next update doubles: 1->2->4->...
            next_update_new = next_update * 2
            return (k_new, v_t_new, next_update_new)

        def no_update_branch(args):
            return args

        k_hat, v_t, next_update = jax.lax.cond(
            do_update,
            update_branch,
            no_update_branch,
            operand=(k_hat, v_t, next_update),
        )

        # stage cost + regret
        cost = x.T @ q @ x + u.T @ r @ u
        regret = cost - optimal_cost

        carry_next = (key, x_next, k_hat, v_t, v_t_cur, s_t, next_update)
        outs = (cost, regret, do_update)
        return carry_next, outs

    init_carry = (key, x0, k_hat, v_t, v_t_cur, s_t, next_update)
    carry_final, (costs, regrets, updates) = jax.lax.scan(
        one_step,
        init_carry,
        xs=jnp.arange(num_steps, dtype=jnp.int32),
    )

    return {
        "costs": costs,
        "regrets": regrets,
        "updates": updates,
        "optimal_cost": optimal_cost,
        "k_star": k_star,
        "k_hat_final": carry_final[2],
    }


@partial(jax.jit, static_argnames=("num_steps",))
def run_many_doubling(a_star, b_star, q, r, keys, lam, num_steps, noise_sigma):
    return jax.vmap(
        lambda key: learn_system_doubling_jit(
            a_star=a_star,
            b_star=b_star,
            q=q,
            r=r,
            key=key,
            lam=lam,
            num_steps=num_steps,
            noise_sigma=noise_sigma,
        )
    )(keys)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    d_x, d_u = 2, 1
    dt = 0.1

    # k = 2.0  # spring
    # c = 0.5  # damping

    # a_star = jnp.array([[1.0, dt], [-k * dt, 1.0 - c * dt]])
    # b_star = jnp.array([[0.0], [dt]])

    # a_star = jnp.array([[1., dt],
    #             [0., 1.]])
    # b_star = jnp.array([[0.5*dt**2],
    #             [dt]])
    q = jnp.eye(d_x)
    r = jnp.eye(d_u)

    d_x, d_u = 3, 1
    a_star = jnp.array(
        [
            [0.95, dt, 0.5 * dt**2],
            [0.0, 0.95, dt],
            [0.0, 0.0, 0.95],
        ]
    )
    b_star = jnp.array(
        [
            [dt**3 / 6],
            [dt**2 / 2],
            [dt],
        ]
    )
    q = jnp.diag(jnp.array([10.0, 1.0, 0.1]))
    r = jnp.array([[0.1]])

    # a_star = jnp.array([[1.2, 0.3], [0.1, 1.05]])
    # b_star = jnp.array([[0.5], [1.0]])

    # q = jnp.eye(2)
    # r = jnp.array([[0.2]])

    N = 100
    keys = jax.random.split(key, N)

    results_inv = run_many(
        a_star=a_star,
        b_star=b_star,
        q=q,
        r=r,
        keys=keys,
        lam=1.0,
        beta=0.1,
        num_steps=1_000,
        noise_sigma=0.1,
        optimism="inv",
    )

    # results_invsqrt = run_many(
    #     a_star=a_star,
    #     b_star=b_star,
    #     q=q,
    #     r=r,
    #     keys=keys,
    #     lam=1.0,
    #     beta=0.1,
    #     num_steps=1_000,
    #     noise_sigma=0.1,
    #     optimism="invsqrt",
    # )

    results_doubling = run_many_doubling(
        a_star=a_star,
        b_star=b_star,
        q=q,
        r=r,
        keys=keys,
        lam=1.0,
        num_steps=1_000,
        noise_sigma=0.1,
    )

    def quantile_band(regrets):
        cum = jnp.cumsum(regrets, axis=1)
        q20, q50, q80 = jnp.quantile(cum, jnp.array([0.2, 0.5, 0.8]), axis=0)
        return cum, q20, q50, q80

    _, q20_a, q50_a, q80_a = quantile_band(results_inv["regrets"])
    # _, q20_b, q50_b, q80_b = quantile_band(results_invsqrt["regrets"])
    _, q20_c, q50_c, q80_c = quantile_band(results_doubling["regrets"])

    t = jnp.arange(q50_a.shape[0])

    plt.plot(t, q50_a, label="inv: H - β V^{-1}")
    plt.fill_between(t, q20_a, q80_a, alpha=0.2)

    # plt.plot(t, q50_b, label="invsqrt: H - β V^{-1/2}", linestyle="--")
    # plt.fill_between(t, q20_b, q80_b, alpha=0.2)

    plt.plot(t, q50_c, label="doubling + action noise", linestyle=":")
    plt.fill_between(t, q20_c, q80_c, alpha=0.2)

    plt.xlabel("Time step")
    plt.ylabel("Cumulative Regret")
    plt.title("Cumulative Regret Comparison (median with 20–80% band)")
    plt.grid()
    plt.legend()
    plt.show()
