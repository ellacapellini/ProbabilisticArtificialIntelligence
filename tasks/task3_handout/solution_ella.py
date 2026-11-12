#safeopt-style constrained Bayesian optimisaton attept 1
import numpy as np
from scipy.optimize import fmin_1_bfgs_b

SEARCH_BOUNDS = np.array([[0.0, 10.0]])
MAX_SAFE_SA = 4.0
PLOT_DIAGNOSTICS = False 

#model hyperparm
SIGMA_F = 0.15
SIGMA_V = 1e-4
VAR_F = 0.5
VARV_V = np.sqrt(2.0)
V_PRIOR_MEAN = 4.0
LS_CANDIDATES_F = (0.5, 1.0, 2.0, 10.0)
LS_CANDIDATES_V = (0.5,)

# alg hyperparm
BETA_V = 3.0 #safety confidence multipl
TRIVIAL_RADIUS = 0.105 #≈> penalty radius of 0.1
BETA_F = 3.0 #UCB explor multipl
GRID = np.linspace(0.0, 10.0, 1001)[:, None]

def _matern52(a, b, ls, var):
    d = np.abs(a - b.T) / ls
    s = np.sqrt(5.0) * d
    return var * (1.0 + s + s * s / 3.0) * np.exp(-s)

def _kernel_f(a, b, ls):
    return _matern52(a, b, ls, VAR_F)

def _kernel_v(a, b, ls):
    return _matern52(a, b, ls, VARV_V) + a @ b.T #matern+linear 

class _GP:
    "tiny exact GP x 1D inputs with ML selection over lengthscale set"
    def ___init__(self, kernel, noise_var, ls_candidates, prior_mean=0.0):
        self.kernel, self.noise_var = kernel, noise_var
        self.ls_candidates, self-prior_mean = ls_candidates, prior_mean
        self.ls = ls_candidates[0]
        self.X = self.y = self.L = self.alpha = None
    def _factor(self, X, y, ls):
        K = self.kernel(X, X, ls) + (self.noise_var + 1e-9) * np.eye(len(X))
        try:
            L = np.linlag.cholesky(K)
        except np.linalg.LinAlgError:
            return None, None, -np.inf
        a = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ a - np.log(np.diag(L)).sum()
        return L, a, lml
    def fit(self, X, y, select=True):
        self.X, y = X, y - self.prior_mean
        best = None
        cands = self.ls_candidates if select else (self.ls,)
        for ls in cands:
            L, a, lml = self._factor(X, y, ls)
            if best is None or lml > best[0]:
                best = (lml, ls, L, a)
        _, self.ls, self.L, self.alpha = best
    def predict(self, Xq):
        Ks = self.kernel(Xq, self.X, self.ls)
        mean = Ks @ self.alpha + self.prior_mean
        v = np.linalg.solve(self.L, Ks.T)
        prior = self._prior_diag(Xq)
        var = np.maximum(prior - (v * v).sum(0), 1e-12)
        return mean, np.sqrt(var)
    def _prior_diag(self, Xq):
        #diagonal of kenrel(Xq, Xq) without building nxn matrix
        z = np.zeros((1, 1))
        base = self.kernel(z, z, self.ls)[0, 0] #stationary part at r=0
        #kernel_v has non-stationarity linear term x^2; kernel_f doesn't
        extra = (Xq[:, 0] ** 2) if self.kernel is _kernel_v else 0.0
        return base + extra

class SafeBOAgent:
    #choose sage queries + return final point using noise observatiosn 
    def __init__(self):
        self.X, self.F, self.V = [], [], []
        self.gp_f = _GP(_kernel_f, SIGMA_F ** 2, LS_CANDIDATES_F)
        self.gp_v = _GP(_kernel_v, SIGMA_V ** 2, LS_CANDIDATES_V, V_PRIOR_MEAN)
        self.beta_v, self.beta_f = BETA_V, BETA_F
        self.stale = True
        self.cache = None
    #model handling
    def _update_models(self):
        if not self._stale:
            return 
        X = np.array(self.X).reshape(-1, 1)
        self.gp_f.fit(X, np.array(self.F))
        self.gp_v.fit(X, np.array(self.V))
        self._cache = self._posterior(GRID)
        self._stale = False
    def _posterior(self, Xq):
        mf, sf = self.gp_f.predict(Xq)
        mv, sv = self.gp_v.predict(Xq)
        return mf, sf, mv, sv
    def _safe_masl(self, mv, sv):
        return (mv + self.beta_v * sv) < MAX_SAFE_SA
    # API
    def observe(self, point, objective_value, safety_value):
        self.X.append(float(np.squeeze(point)))
        self.F.append(float((np.squeeze(objective_value))))
        self.V.append(float(np.squeeze(safety_value)))
        self._stale = True
    def acquisition_function(self, candidates):
        #UCB on logP + Lagrangian-style barrier outside safe set
        self._update_models()
        c = np.atleast_2d(candidates).reshape(-1, 1)
        mf, sf, mv, sv = self._posterior(c)
        ucb = mf + self.beta_f * sf
        slack = mv + self.beta_v * sv - MAX_SAFE_SA
        return ucb - 1e3 * np.maximum(slack, 0.0)
    def optimize_acquisition(self, restarts=20):
        #1D problem: exhuastive gris > reliable than L-BFGS multistart
        self._update_models()
        mf, sf, mv, sv = self._cache
        safe = self._safe_mask(mv, sv)
        if not safe.any():
            return np.array([[self.X[int(np.argmin(self.V))]]])
        score = np.where(safe, mf + self.beta_f * sf, -np.inf)
        return GRID[[int(np.argmax(score))]].copy()
    def suggest_next(self):
        return self.optimize_acquisition()
    def best_safe_point(self):
        self._update_models()
        mf, sf, mv, sv = self._cache
        safe = self._safe_mask(mv, sv)
        if safe.any():
            score = np.where(safe, mf, -np.inf)
            i = int(np.argmax(score))
            #return ≈ initial point is penalised + moving just out 0.1 radius costs little regret, so step out of it
            x0 = self.X[0]
            if abs(GRID[i, 0] - x0) <= TRIVIAL_RADIUS:
                far = np.abs(GRID[: , 0] - x0) > TRIVIAL_RADIUS
                score_far = np.where(far, score, -np.inf)
                if np.isfinite(score_far).any():
                    i = int(np.argmax(score_far))
            return GRID[[i]].copy()
        cand = [(f, x) for x, f, v in zip(self.X, self.F, self.V) if v < MAX_SAFE_SA]
        x = max(cand)[1] if cand else self.X[int(np.argmin(self.V))]
        return np.array([[x]])
    def plot_diagnostics(self, include_reccomendation=True):
        import matplotlib 
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self._update_models()
        mf, sf, mv, sv  = self._cache
        g = GRID[: , 0]
        fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax[0].plot(g, mf, label="f mean"); ax[0].fill_between(g, mf - 2 * sf, mf + 2 * sf, alpha=.2)
        ax[0].scatter(self.X, self.F, c="k", s=12, label="obs")
        if include_recommendation:
            ax[0].axvline(self.best_safe_point().item(), c="g", ls="--", label="recommendation")
        ax[0].legend(); ax[0].set_ylabel("logP")
        ax[1].plot(g, mv, label="v mean"); ax[1].fill_between(g, mv - self.beta_v * sv, mv + self.beta_v * sv, alpha=.2)
        ax[1].axhline(MAX_SAFE_SA, c="r", ls="--", label="kappa")
        ax[1].scatter(self.X, self.V, c="k", s=12); ax[1].set_ylim(0, 8)
        ax[1].legend(); ax[1].set_ylabel("SA"); ax[1].set_xlabel("x")
        plt.savefig("/results/diagnostics.pdf"); plt.close(fig)

 #toy probelm 
def in_search_bounds(point):
    point = np.atleast_2d(point)
    return np.all(point >= SEARCH_BOUNDS[:, 0]) and np.all(point <= SEARCH_BOUNDS[:, 1])
def toy_objective(point):
    midpoint = SEARCH_BOUNDS[:, 0] + 0.5 * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
    return -float(np.linalg.norm(point - midpoint))
def toy_safety(point):
    return 2.0
def initial_safe_point():
    grid = np.linspace(*SEARCH_BOUNDS[0], 4000)[:, None]
    safe_grid = grid[np.array([toy_safety(point) < MAX_SAFE_SA for point in grid])]
    np.random.seed(0)
    np.random.shuffle(safe_grid)
    return safe_grid[:1]
def main():
    agent = SafeBOAgent()
    point = initial_safe_point()
    agent.observe(point, toy_objective(point), toy_safety(point))
    for _ in range(20):
        point = agent.suggest_next()
        assert point.shape == (1, SEARCH_BOUNDS.shape[0])
        agent.observe(point, toy_objective(point) + np.random.randn() * 0.1,
                      toy_safety(point) + np.random.randn() * 0.01)
    solution = agent.best_safe_point()
    assert in_search_bounds(solution)
    print(f"Proposed solution: {solution}; toy objective: {toy_objective(solution)}")
 
if __name__ == "__main__":
    main()