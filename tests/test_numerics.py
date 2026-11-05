import numpy as np
import pytest

from heatpinn.numerics import relative_l2_error, solve, stability_number
from heatpinn.problems import BOUNDARY_TYPES, HeatProblem


def error(problem, n_cells, n_steps, scheme):
    x, u = solve(problem, n_cells, n_steps, scheme)
    return relative_l2_error(u, problem.exact(x, problem.t_end))


@pytest.mark.parametrize("boundary", BOUNDARY_TYPES)
def test_exact_solution_satisfies_pde_and_boundaries(boundary):
    p = HeatProblem(boundary=boundary)
    x, t, e = np.linspace(0.05, 0.95, 19), 0.3, 1e-4
    u_t = (p.exact(x, t + e) - p.exact(x, t - e)) / (2 * e)
    u_xx = (p.exact(x + e, t) - 2 * p.exact(x, t) + p.exact(x - e, t)) / e**2
    assert np.max(np.abs(u_t - p.alpha * u_xx)) < 1e-5

    def u_x(at):
        return (p.exact(at + e, t) - p.exact(at - e, t)) / (2 * e)

    left = u_x(0.0) if boundary == "neumann" else p.exact(0.0, t)
    right = p.exact(1.0, t) if boundary == "dirichlet" else u_x(1.0)
    assert abs(left) < 1e-7 and abs(right) < 1e-7


@pytest.mark.parametrize("boundary", BOUNDARY_TYPES)
def test_crank_nicolson_is_second_order(boundary):
    p = HeatProblem(boundary=boundary)
    coarse, fine = error(p, 40, 40, "crank-nicolson"), error(p, 80, 80, "crank-nicolson")
    assert np.log2(coarse / fine) == pytest.approx(2.0, abs=0.15)


@pytest.mark.parametrize("boundary", BOUNDARY_TYPES)
def test_ftcs_converges_when_stable(boundary):
    p = HeatProblem(boundary=boundary)
    assert stability_number(p, 40, 800) <= 0.5
    assert error(p, 40, 800, "ftcs") < 1e-2


def test_ftcs_blows_up_when_unstable():
    p = HeatProblem()
    assert stability_number(p, 40, 200) > 0.5
    assert error(p, 40, 200, "ftcs") > 1.0
    assert error(p, 40, 200, "crank-nicolson") < 1e-2


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        HeatProblem(boundary="periodic")
    with pytest.raises(ValueError):
        solve(HeatProblem(), 10, 10, "euler")
