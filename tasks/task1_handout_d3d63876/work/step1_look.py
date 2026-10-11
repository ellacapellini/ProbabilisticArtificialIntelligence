import sys; sys.path.insert(0, "..") #import untouched template
import numpy as np, matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from solution import determine_residential_flags  #PI helper

X = np.loadtxt("../train_x.csv", delimiter=",", skiprows=1)
y = np.loadtxt("../train_y.csv", delimiter=",", skiprows=1)
xy, area = X[:, :2], X[:,2].astype(bool)

print("n =", len(y), "| residential share=", area.mean().round(3))
print("y mean/Std overall:", y.mean().round(2), y.std().round(2))
print("y mean in / out of residential:", y[area].mean().round(2), y[~area].mean().round(2))

# model free noise estimate from nearest neighbour pairs
d, j = cKDTree(xy).query(xy, k=2)
close = d[:, 1] < 0.002 #grid spacing 0.00125
diff = y[close] - y[j[close, 1]]
print("pairs used:", close.sum(), "| spacing:", np.median(d[close, 1]).round(5))
print("noise variance estimate:", (diff.var() / 2).round(2))

#map
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
s = ax[0].scatter(*xy.T, c=y, s=3, cmap="viridis"); fig.colorbar(s, ax=ax[0]); ax[0].set_title("train y")
ax[1].scatter(*xy.T, c=area, s=3, cmap="coolwarm"); ax[1].set_title("residential flag")
plt.tight_layout(); plt.savefig("step1_map.png", dpi=120); plt.show()

