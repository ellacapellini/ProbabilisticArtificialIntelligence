import os
import typing
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process.kernels import ConstantKernel as C
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import brentq
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm


# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = False
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict_pollution methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.rng = np.random.default_rng(seed=0)
        self.n_hyper = 1200      # subsample size used only for marginal-likelihood hyperparameter fitting
        self.tile = 0.1          # side of the tiles used for local prediction
        self.margin = 0.12       # training points this far outside a tile still condition it (~2.5 length scales)
        self.z = self._expectile_z(COST_W_UNDERPREDICT / COST_W_NORMAL)

    @staticmethod
    def _expectile_z(w):
        """
        For loss  (f - m)^2 * (w if m <= f else 1)  and a Gaussian posterior f ~ N(mu, s^2), the Bayes-optimal
        point estimate is the w/(1+w) *expectile* (not quantile!) of the posterior: m = mu + z*s, where z solves
        w * E[(f-m)+] = E[(m-f)+]. In standardised units, E[(f-m)+] = phi(z) - z(1-Phi(z)), E[(m-f)+] = phi(z) + z Phi(z).
        """
        g = lambda z: w * (norm.pdf(z) - z * norm.sf(z)) - (norm.pdf(z) + z * norm.cdf(z))
        return brentq(g, 0.0, 10.0)

    # Don't change the name or the signature of this function
    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        """
        Fit your model on the given training data.
        Step 1: learn kernel hyperparameters by maximising the marginal likelihood on a random subset (O(m^3), m small).
        Step 2: keep ALL data; prediction conditions only on the local neighbourhood of each query tile (see predict_pollution).
        """
        X, y = train_coordinates, train_pollution_targets
        self.mu0 = y.mean()
        idx = self.rng.permutation(len(y))[: self.n_hyper]

        kernel = C(300.0, (1e-1, 1e5)) * Matern(0.05, (1e-3, 1.0), nu=1.5) + WhiteKernel(1.0, (1e-3, 1e2))
        gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=0, random_state=0)
        gp.fit(X[idx], y[idx] - self.mu0)

        self.kernel_ = gp.kernel_
        self.signal_kernel = gp.kernel_.k1                # C * Matern  -> latent function f
        self.noise = gp.kernel_.k2.noise_level            # measurement noise variance (absent in the test targets)
        print(f'Learned kernel: {gp.kernel_}')

        self.X, self.y = X, y - self.mu0

    def _predict_latent(self, Xq):
        """Posterior mean/std of the noise-free latent function, using one local GP per spatial tile."""
        mean, std = np.zeros(len(Xq)), np.zeros(len(Xq))
        cell = np.floor(Xq / self.tile).astype(int)
        keys, inv = np.unique(cell, axis=0, return_inverse=True)
        inv = inv.ravel()
        for j, (cx, cy) in enumerate(keys):
            q = np.where(inv == j)[0]
            lo = np.array([cx, cy]) * self.tile - self.margin
            hi = np.array([cx + 1, cy + 1]) * self.tile + self.margin
            near = np.all((self.X >= lo) & (self.X <= hi), axis=1)
            Xn, yn = self.X[near], self.y[near]
            if len(Xn) == 0:   # no data around: fall back to the prior
                mean[q], std[q] = 0.0, np.sqrt(self.signal_kernel.k1.constant_value)
                continue
            K = self.signal_kernel(Xn) + self.noise * np.eye(len(Xn))
            L = np.linalg.cholesky(K)
            Ks = self.signal_kernel(Xn, Xq[q])
            alpha = cho_solve((L, True), yn)
            v = solve_triangular(L, Ks, lower=True)
            mean[q] = Ks.T @ alpha
            std[q] = np.sqrt(np.maximum(self.signal_kernel.k1.constant_value - np.sum(v ** 2, axis=0), 1e-12))
        return mean + self.mu0, std

    # Don't change the name or the signature of this function
    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :return: (predictions, GP posterior mean, GP posterior stddev)
        """
        gp_mean, gp_std = self._predict_latent(test_coordinates)

        # Asymmetric cost -> shift upwards by z posterior stddevs, but only inside residential areas.
        shift = self.z * gp_std * np.asarray(test_residential_flags, dtype=bool)
        predictions = gp_mean + shift

        return predictions, gp_mean, gp_std

# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
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
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function
def determine_residential_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
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
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution(visualization_grid, grid_residential_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

# You do not need to change the name or the signature of this function; convenience helper used by main() 
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
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
    # Load the training dataset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_residential_flags, test_coordinates, test_residential_flags = get_city_area_data(train_x, test_x)

    # Fit the model
    print('Training model')
    model = Model()
    model.fit_model(train_coordinates, train_y, train_residential_flags)

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.predict_pollution(test_coordinates, test_residential_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()