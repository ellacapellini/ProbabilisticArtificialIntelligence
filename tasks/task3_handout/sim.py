"""Local simulator: random tasks drawn from GP priors, scored with the handout formula."""
import sys, importlib, numpy as np
import solution as S

G = np.linspace(0, 10, 1001)

def mat52(a, b, ls, var):
    d = np.abs(a[:, None] - b[None]) / ls; s = np.sqrt(5) * d
    return var * (1 + s + s * s / 3) * np.exp(-s)
def rbf(a, b, ls, var):
    return var * np.exp(-0.5 * ((a[:, None] - b[None]) / ls) ** 2)
def make_task(rng):
    kf = rng.choice(["m", "r"]); lf = rng.choice([0.5, 1.0, 2.0])
    kv = rng.choice(["m", "r"]); lv = rng.choice([0.5, 1.0, 2.0])
    k = mat52 if kf == "m" else rbf
    Kf = k(G, G, lf, 0.5) + 1e-8 * np.eye(len(G))
    f = np.linalg.cholesky(Kf) @ rng.standard_normal(len(G))
    f = (f - f.min()) / (f.max() - f.min()) # bounded in [0,1]
    k = mat52 if kv == "m" else rbf
    Kv = k(G, G, lv, np.sqrt(2)) + 0.3 * np.outer(G, G) / 100 + 1e-8 * np.eye(len(G))
    v = 4 + np.linalg.cholesky(Kv) @ rng.standard_normal(len(G))
    return f, v
def run_task(rng, n_iter=20):
    while True:
        f, v = make_task(rng)
        safe = v < 4
        if safe.sum() > 20 and (~safe).sum() > 20: break
    i0 = rng.choice(np.where(safe)[0]); x0 = G[i0]
    #connect safe component around x0 -> local optimum
    lo = hi = i0
    while lo > 0 and safe[lo - 1]: lo -= 1
    while hi < len(G) - 1 and safe[hi + 1]: hi += 1
    f_loc = f[lo:hi + 1].max()
    fi = lambda x: np.interp(x, G, f); vi = lambda x: np.interp(x, G, v)
    agent = S.SafeBOAgent(); N = 0
    def query(x):
        nonlocal N
        if vi(x) >= 4: N += 1
        agent.observe(np.array([[x]]), fi(x) + 0.15 * rng.standard_normal(),
                      vi(x) + 1e-4 * rng.standard_normal())
    query(x0)
    for _ in range(n_iter):
        query(float(agent.suggest_next().item()))
    xs = float(agent.best_safe_point().item())
    r = max((f_loc - fi(xs)) / max(f_loc, 1e-9), 0)
    sig = lambda z: 1 / (1 + np.exp(-z))
    score = 1 - 0.4 * r - 0.6 * sig(3 * N - 3.25) - 0.15 * (abs(xs - x0) <= 0.1)
    # unsafe final answer is not allowed: count as a heavy failure
    if vi(xs) >= 4: score -= 0.6
    return score, r, N, abs(xs - x0) <= 0.1
def evaluate(n_tasks=100, seed=0, n_iter=20, **over):
    for k, val in over.items(): setattr(S, k, val)
    rng = np.random.default_rng(seed); res = np.array([run_task(rng, n_iter) for _ in range(n_tasks)], float)
    return res.mean(0)

if __name__ == "__main__":
    print("score, regret, N_unsafe, trivial:", evaluate(int(sys.argv[1]) if len(sys.argv) > 1 else 100))