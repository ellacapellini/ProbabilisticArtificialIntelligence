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