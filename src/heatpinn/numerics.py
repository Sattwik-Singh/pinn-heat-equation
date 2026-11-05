"""Second-order finite-difference solvers on a uniform grid.

Neumann boundaries use a mirrored ghost node, which keeps the scheme
second-order accurate in space. Time stepping is FTCS (explicit, first order)
or Crank-Nicolson (implicit, second order, unconditionally stable).
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from heatpinn.problems import HeatProblem


def _laplacian(n_cells: int, boundary: str) -> sparse.csc_matrix:
    """Return the discrete d2/dx2 operator on the n_cells + 1 grid nodes."""
    h = 1.0 / n_cells
    size = n_cells + 1
    lap = sparse.diags(
        [np.ones(size - 1), -2.0 * np.ones(size), np.ones(size - 1)], [-1, 0, 1]
    ).tolil()

    if boundary == "neumann":
        lap[0, 1] = 2.0
    else:
        lap[0, :] = 0.0
    if boundary == "dirichlet":
        lap[-1, :] = 0.0
    else:
        lap[-1, -2] = 2.0
    return (lap / h**2).tocsc()


def stability_number(problem: HeatProblem, n_cells: int, n_steps: int) -> float:
    """Return r = alpha * dt / h^2. FTCS is stable only for r <= 1/2."""
    return problem.alpha * (problem.t_end / n_steps) * n_cells**2


def solve(
    problem: HeatProblem, n_cells: int, n_steps: int, scheme: str = "crank-nicolson"
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate to t_end and return the grid and the final solution."""
    x = np.linspace(0.0, 1.0, n_cells + 1)
    u = problem.initial(x)
    dt = problem.t_end / n_steps
    lap = _laplacian(n_cells, problem.boundary)
    identity = sparse.identity(n_cells + 1, format="csc")

    if scheme == "ftcs":
        step = identity + problem.alpha * dt * lap
        for _ in range(n_steps):
            u = step @ u
    elif scheme == "crank-nicolson":
        half = 0.5 * problem.alpha * dt * lap
        solver = splu((identity - half).tocsc())
        rhs = identity + half
        for _ in range(n_steps):
            u = solver.solve(rhs @ u)
    else:
        raise ValueError(f"unknown scheme: {scheme!r}")
    return x, u


def relative_l2_error(predicted: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(predicted - exact) / np.linalg.norm(exact))
