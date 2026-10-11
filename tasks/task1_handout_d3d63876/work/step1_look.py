import sys; sys.path.insert(0, "..") #import untouched template
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import ckdtree
from solution import determine_resiential_flags #prof helper

X = np.loadtxt("../train_x.csv", delimiter=",", skiprows=1)
Y = np.loadtxt("../train_y.csv", delimiter=",", skiprows=1)
xy, area = X[:, :2], X[:,2].astype(bool)

print()