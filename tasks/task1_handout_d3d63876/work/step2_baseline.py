import sys; sys.path.insert(0, "..")
import numpy as np
from scipy.stats import norm 
from scipy.optimize import brentq
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from solution import calculate_cost #PI cost

X = np.loadtxt("../train_x.csv", delimiter=",", skiprows=1)
y = np.loadtxt("../train_y.csv", delimiter=",", skiprows=1)
xy, area = X[:, :2], X[:, 2].astype(bool)
p = np.random.default_rng(0).permutation(len(y))
va, tr = p[:3000], p[3000:] #fixed holdout to reause every later step

#1 global GP pn 2000 random points 
sub, mu0 = tr[:2000], y[tr].mean()
gp = GPR(C(300.) * Matern(0.05, nu=1.5) + WhiteKernel(1.), random_state=0)
gp.fit(xy[sub], y[sub] - mu0)
print("kernel:", gp.kernel_)
noise = gp.kernel_.k2.noise_level
m, s = gp.predict(xy[va], return_std=True)
m += mu0
s = np.sqrt(np.maximum(s**2 - noise, 1e-9)) #want std of true field (sklearn's std includes noise)

#shift up by z*std (only inside residential areas)
#optimal point estimate for 50x if underpredic (50/51 expectile)
g = lambda z: 50 * (norm.pdf(z) - z * norm.sf(z)) - (norm.pdf(z) + z * norm.cdf(z))
z = brentq(g, 0, 10); print("expectile z=", round(z, 3), "(quantile would be", round(norm.pdf(50/51), 3), ")")
a = area[va]
print("cost, plain mean:", calculate_cost(y[va], m, a).round(2))
print("cost, expectile:", calculate_cost(y[va], m +z * s* a, a).round(2))
