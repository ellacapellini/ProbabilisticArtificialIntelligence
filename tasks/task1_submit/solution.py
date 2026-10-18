import os
import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import brentq
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel, RationalQuadratic


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 500  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0

# ----------------------------------------------------------------------------------------------------------------
# Model constants
# ----------------------------------------------------------------------------------------------------------------
N_HYPER_POINTS = 1200   
TILE = 0.05            
KNN = 300  
MARGIN_UNUSED = 0.12          
SHIFT_SCALE = 0.9   
FIXED_NOISE = 1.8     
CAL_NEIGHBOURS = 100   
CAL_FOLDS = 5         
WEIGHT_UNDER = 50.0    
STACK_NEIGHBOURS = 20  
RAW_NEIGHBOURS = 16    
USE_EXPECTILE = True      
USE_EXTRA_KERNELS = False 
CAL_FLOOR = 0.25      
JITTERS = (0.0, 1e-6, 1e-4, 1e-2)  


class Model(object):
    def __init__(self):
        self.rng = np.random.default_rng(seed=0)
        self.z = self._expectile_z(COST_W_UNDERPREDICT / COST_W_NORMAL)

    @staticmethod
    def _expectile_z(w):
        g = lambda z: w * (norm.pdf(z) - z * norm.sf(z)) - (norm.pdf(z) + z * norm.cdf(z))
        return brentq(g, 0.0, 10.0)

    # Don't change the name or the signature of this function
    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        X, y = train_coordinates, train_pollution_targets
        self.mu0 = y.mean()                   
        self.X, self.y = X, y - self.mu0
        subset = self.rng.permutation(len(y))[:N_HYPER_POINTS]
        kernel = C(300.0, (1e-1, 1e5)) * Matern(0.05, (1e-3, 1.0), nu=1.5) + WhiteKernel(1.0, (1e-3, 1e2))
        fitted = GaussianProcessRegressor(kernel, random_state=0).fit(X[subset], self.y[subset]).kernel_
        self.signal_kernel = fitted.k1                      
        self.noise = fitted.k2.noise_level                 
        self.prior_var = fitted.k1.k1.constant_value       
        if FIXED_NOISE is not None:
            self.noise = FIXED_NOISE
        print(f'Learned kernel: {fitted} | noise used: {self.noise:.2f}')
        self.extra = []                                     
        if USE_EXTRA_KERNELS:
            for base in (Matern(0.05, (1e-3, 1.0), nu=2.5), RationalQuadratic(0.05, 1.0, (1e-3, 1.0), (0.1, 50.0))):
                k0 = C(300.0, (1e-1, 1e5)) * base + WhiteKernel(1.0, (1e-3, 1e2))
                kf = GaussianProcessRegressor(k0, random_state=0).fit(X[subset], self.y[subset]).kernel_
                self.extra.append((kf.k1, kf.k1.k1.constant_value))
                print(f'Extra kernel: {kf.k1}')
        self._fit_second_stage()

    def _cholesky(self, K):
        for jitter in JITTERS:
            try:
                return np.linalg.cholesky(K + jitter * self.prior_var * np.eye(len(K)))
            except np.linalg.LinAlgError:
                continue
        return None

    def _predict_latent(self, Xq, Xtr=None, ytr=None, kernel=None, prior_var=None):
        Xtr = self.X if Xtr is None else Xtr
        ytr = self.y if ytr is None else ytr
        kernel = self.signal_kernel if kernel is None else kernel
        prior_var = self.prior_var if prior_var is None else prior_var
        mean, std = np.zeros(len(Xq)), np.full(len(Xq), np.sqrt(prior_var)) #def = prior
        tree = cKDTree(Xtr)
        cell = np.floor(Xq / TILE).astype(int) #which tile does each query point lie in?
        tiles, inverse = np.unique(cell, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        k = min(KNN, len(Xtr))
        for j, (cx, cy) in enumerate(tiles):
            q = np.where(inverse == j)[0]
            centre = (np.array([cx, cy]) + 0.5) * TILE
            _, idx = tree.query(centre, k=k)# k training points nearest to the tile
            Xn, yn = Xtr[idx], ytr[idx]
            L = self._cholesky(kernel(Xn) + self.noise * np.eye(len(Xn)))
            if L is None:
                continue                                                            
            Ks = kernel(Xn, Xq[q])
            v = solve_triangular(L, Ks, lower=True)
            mean[q] = Ks.T @ cho_solve((L, True), yn)
            std[q] = np.sqrt(np.maximum(prior_var - np.sum(v ** 2, axis=0), 1e-12))
        return mean + self.mu0, std

    def _features(self, Xq, mean, std, self_in_train, extras=()):
        k = STACK_NEIGHBOURS + int(self_in_train)
        d, idx = self.tree.query(Xq, k=k)
        if self_in_train:  #training point must not see its own label
            d, idx = d[:, 1:], idx[:, 1:]
        yn = self.y[idx] + self.mu0
        off = self.X[idx] - Xq[:, None, :]
        nn = RAW_NEIGHBOURS
        A = np.concatenate([off, np.ones((len(Xq), off.shape[1], 1))], axis=2)         
        AtA = np.einsum('nki,nkj->nij', A, A) + 1e-9 * np.eye(3)
        Aty = np.einsum('nki,nk->ni', A, yn)
        coef = np.linalg.solve(AtA, Aty[..., None])[..., 0]
        return np.column_stack([Xq, mean, std, d[:, 0], d.mean(axis=1), yn.std(axis=1),
                                np.abs(yn - mean[:, None]).mean(axis=1), yn.max(axis=1) - yn.min(axis=1),
                                off[:, :nn, 0], off[:, :nn, 1], (yn - mean[:, None])[:, :nn], coef[:, :2]]
                               + [np.column_stack([em - mean, es]) for em, es in extras])

    def _fit_second_stage(self):
        n = len(self.y)
        fold = np.random.default_rng(1).permutation(n) % CAL_FOLDS
        oof_mean, oof_std = np.zeros(n), np.zeros(n)
        oof_extra = [(np.zeros(n), np.zeros(n)) for _ in self.extra]
        for f in range(CAL_FOLDS):
            tr, te = fold != f, fold == f
            oof_mean[te], oof_std[te] = self._predict_latent(self.X[te], self.X[tr], self.y[tr])
            for (em, es), (ker, pv) in zip(oof_extra, self.extra):
                em[te], es[te] = self._predict_latent(self.X[te], self.X[tr], self.y[tr], ker, pv)
        self.tree = cKDTree(self.X)
        feats = self._features(self.X, oof_mean, oof_std, True, oof_extra)
        resid = (self.y + self.mu0) - oof_mean
        kw = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, l2_regularization=5.0, random_state=0)
        self.mean_model = HistGradientBoostingRegressor(**kw).fit(feats, resid)
        cv_corr = np.zeros(n)                               
        for f in range(CAL_FOLDS):
            tr, te = fold != f, fold == f
            cv_corr[te] = HistGradientBoostingRegressor(**kw).fit(feats[tr], resid[tr]).predict(feats[te])
        self.scale_model = HistGradientBoostingRegressor(**kw).fit(feats, (resid - cv_corr) ** 2)
        if USE_EXPECTILE:
            r2 = resid - cv_corr
            w = np.ones(n)
            kw2 = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=5.0, random_state=0)
            for _ in range(6):
                self.expectile_model = HistGradientBoostingRegressor(**kw2).fit(feats, r2, sample_weight=w)
                w = np.where(r2 > self.expectile_model.predict(feats), WEIGHT_UNDER, 1.0)

    # Don't change the name or the signature of this function
    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gp_mean, gp_std = self._predict_latent(test_coordinates)
        extras = [self._predict_latent(test_coordinates, None, None, ker, pv) for ker, pv in self.extra]
        feats = self._features(test_coordinates, gp_mean, gp_std, False, extras)
        gp_mean = gp_mean + self.mean_model.predict(feats)                      
        if USE_EXPECTILE:
            q = np.maximum(self.expectile_model.predict(feats), 0.0)
            total_var = np.maximum((q / self.z) ** 2, 0.5)
        else:
            total_var = np.maximum(self.scale_model.predict(feats), 0.5)         
        gp_std = np.sqrt(np.maximum(total_var - self.noise, CAL_FLOOR * gp_std ** 2))   
        shift = SHIFT_SCALE * self.z * gp_std * np.asarray(test_residential_flags, dtype=bool)
        predictions = gp_mean + shift
        return predictions, gp_mean, gp_std


# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray) -> float:
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape
    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL
    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(residential_flag) for residential_flag in residential_flags]
    weights[mask] = COST_W_UNDERPREDICT
    # Weigh the cost and return the average
    return np.mean(cost * weights)

# You don't have to change this function
def is_inside_circle(coordinate, circle_parameters):
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function
def determine_residential_flags(grid_coordinates):
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])

    residential_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        residential_flags[i] = any([is_inside_circle(coordinate, circ) for circ in circles])
    return residential_flags

# Don't change the name or the signature of this function
def perform_extended_model_evaluation(model: Model, output_dir: str = '/results'):
    print('Performing extended evaluation')
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)
    predictions, gp_mean, gp_stddev = model.predict_pollution(visualization_grid, grid_residential_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    vmin, vmax = 0.0, 65.0
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

# You do not need to change the name or the signature of this function; convenience helper used by main() 
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_residential_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_residential_flags = np.zeros((test_x.shape[0],), dtype=bool)
    train_coordinates = train_x[:, :2].astype(float)
    train_residential_flags = train_x[:, 2].astype(bool)
    test_coordinates = test_x[:, :2].astype(float)
    test_residential_flags = test_x[:, 2].astype(bool)
    assert train_coordinates.shape[0] == train_residential_flags.shape[0] and test_coordinates.shape[0] == test_residential_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_residential_flags.ndim == 1 and test_residential_flags.ndim == 1
    return train_coordinates, train_residential_flags, test_coordinates, test_residential_flags

# you don't have to change this function
def main():
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)
    train_coordinates, train_residential_flags, test_coordinates, test_residential_flags = get_city_area_data(train_x, test_x)
    print('Training model')
    model = Model()
    model.fit_model(train_coordinates, train_y, train_residential_flags)
    print('Predicting on test features')
    predictions = model.predict_pollution(test_coordinates, test_residential_flags)
    print(predictions)
    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')

if __name__ == "__main__":
    main()