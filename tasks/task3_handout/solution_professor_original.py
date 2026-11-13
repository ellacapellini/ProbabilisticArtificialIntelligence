"""Solution template."""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b


SEARCH_BOUNDS = np.array([[0.0, 10.0]])
MAX_SAFE_SA = 4.0
PLOT_DIAGNOSTICS = False  # Debug only: plot the first trial, then stop.


class SafeBOAgent:
    """Choose safe queries and return a final point using noisy observations."""

    def __init__(self):
        # TODO: Set up your models and observation storage.
        pass

    def suggest_next(self) -> np.ndarray:
        """Return the next query as an array of shape (1, 1)."""
        # TODO: Fit or update your models and choose the next query.
        # You may use optimize_acquisition() below.
        raise NotImplementedError

    def optimize_acquisition(self, restarts: int = 20) -> np.ndarray:
        """Maximize acquisition_function() over SEARCH_BOUNDS.

        Returns an array of shape (1, 1). You may modify this helper.
        """
        def objective(candidate):
            return -self.acquisition_function(np.atleast_2d(candidate)).item()

        candidates = []
        scores = []
        for _ in range(restarts):
            start = SEARCH_BOUNDS[:, 0] + (
                SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0]
            ) * np.random.rand(SEARCH_BOUNDS.shape[0])
            point, negative_score, _ = fmin_l_bfgs_b(
                objective, x0=start, bounds=SEARCH_BOUNDS, approx_grad=True
            )
            candidates.append(np.clip(point, SEARCH_BOUNDS[:, 0], SEARCH_BOUNDS[:, 1]))
            scores.append(-negative_score)

        return np.atleast_2d(candidates[int(np.argmax(scores))])

    def acquisition_function(self, candidates: np.ndarray) -> np.ndarray:
        """Score candidate points; accept an array of shape (n, 1)."""
        # TODO: Define an acquisition score that accounts for the SA constraint.
        raise NotImplementedError

    def observe(
        self, point: np.ndarray, objective_value: float, safety_value: float
    ) -> None:
        """Record a query and its noisy logP and SA observations."""
        # TODO: Add this observation to your models or training data.
        raise NotImplementedError

    def best_safe_point(self) -> np.ndarray:
        """Return your predicted safe optimum as an array of shape (1, 1)."""
        # TODO: Choose a point expected to have SA below MAX_SAFE_SA.
        raise NotImplementedError

    def plot_diagnostics(self, include_recommendation: bool = True) -> None:
        """Optional plot; called by checker when PLOT_DIAGNOSTICS is True.

        Save plots that you create here under the docker results directory (default is "/results"). Example:
        ```
        # plot stuff
        plt.savefig(f"/results/{filename}.pdf")
        ```
        """
        pass


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def in_search_bounds(point: np.ndarray) -> bool:
    """Check whether a point is inside the search interval."""
    point = np.atleast_2d(point)
    return np.all(point >= SEARCH_BOUNDS[:, 0]) and np.all(point <= SEARCH_BOUNDS[:, 1])


def toy_objective(point: np.ndarray) -> float:
    """Toy logP objective."""
    midpoint = SEARCH_BOUNDS[:, 0] + 0.5 * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
    return -float(np.linalg.norm(point - midpoint))


def toy_safety(point: np.ndarray) -> float:
    """Toy SA value."""
    return 2.0


def initial_safe_point() -> np.ndarray:
    """Choose an initial point with toy SA below the threshold."""
    grid = np.linspace(*SEARCH_BOUNDS[0], 4000)[:, None]
    safe_grid = grid[np.array([toy_safety(point) < MAX_SAFE_SA for point in grid])]
    np.random.seed(0)
    np.random.shuffle(safe_grid)
    return safe_grid[:1]


def main() -> None:
    """Illustration only; the checker does not call this function."""
    agent = SafeBOAgent()
    point = initial_safe_point()
    agent.observe(point, toy_objective(point), toy_safety(point))

    for _ in range(20):
        point = agent.suggest_next()
        assert point.shape == (1, SEARCH_BOUNDS.shape[0]), (
            f"suggest_next() must return shape (1, {SEARCH_BOUNDS.shape[0]})"
        )
        agent.observe(
            point,
            toy_objective(point) + np.random.randn() * 0.1,
            toy_safety(point) + np.random.randn() * 0.01,
        )

    solution = agent.best_safe_point()
    assert in_search_bounds(solution), "best_safe_point() returned a point outside the domain"
    print(f"Proposed solution: {solution}; toy objective: {toy_objective(solution)}")


if __name__ == "__main__":
    main()
