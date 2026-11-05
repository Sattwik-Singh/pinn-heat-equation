import numpy as np
import pytest
import torch

from heatpinn.pinn import (
    DTYPE,
    HeatPINN,
    PINNConfig,
    boundary_error,
    evaluate,
    exact_torch,
    pde_residual,
    train,
    train_inverse,
)
from heatpinn.problems import BOUNDARY_TYPES, HeatProblem

TINY = dict(width=16, depth=2, n_collocation=200, n_boundary=40, n_initial=40)


@pytest.mark.parametrize("boundary", BOUNDARY_TYPES)
def test_hard_model_satisfies_conditions_before_training(boundary):
    torch.manual_seed(0)
    problem = HeatProblem(boundary=boundary)
    model = HeatPINN(problem, PINNConfig(strategy="hard", **TINY))
    metrics = evaluate(model)
    assert metrics["bc_rms"] < 1e-12
    assert metrics["ic_rms"] < 1e-12


@pytest.mark.parametrize("boundary", BOUNDARY_TYPES)
def test_residual_of_exact_solution_vanishes(boundary):
    problem = HeatProblem(boundary=boundary)
    x = torch.rand(50, 1, dtype=DTYPE)
    t = torch.rand(50, 1, dtype=DTYPE)
    residual = pde_residual(lambda a, b: exact_torch(problem, a, b), x, t, problem.alpha)
    assert residual.abs().max() < 1e-12


def test_exact_torch_matches_numpy():
    problem = HeatProblem(boundary="mixed")
    x = torch.linspace(0, 1, 11, dtype=DTYPE).reshape(-1, 1)
    t = 0.4 * torch.ones_like(x)
    assert np.allclose(exact_torch(problem, x, t).numpy(), problem.exact(x.numpy(), t.numpy()))


@pytest.mark.parametrize("strategy", ["soft", "hard"])
def test_training_is_reproducible_and_improves(strategy):
    problem = HeatProblem()
    config = PINNConfig(strategy=strategy, adam_steps=150, lbfgs_steps=0, **TINY)
    _, first = train(problem, config)
    _, second = train(problem, config)
    assert first["relative_l2"] == second["relative_l2"]
    assert first["loss_history"][-1] < first["loss_history"][0]


def test_lbfgs_refines_short_training():
    config = PINNConfig(adam_steps=300, lbfgs_steps=200, **TINY)
    _, record = train(HeatProblem(), config)
    assert record["relative_l2"] < record["relative_l2_adam"]


def test_inverse_moves_alpha_toward_truth():
    problem = HeatProblem()
    rng = np.random.default_rng(0)
    x, t = rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)
    config = PINNConfig(adam_steps=500, lbfgs_steps=300, **TINY)
    _, record = train_inverse(problem, config, x, t, problem.exact(x, t), initial_alpha=0.3)
    assert abs(record["alpha"] - problem.alpha) < 0.1


def test_invalid_strategy_is_rejected():
    with pytest.raises(ValueError):
        HeatPINN(HeatProblem(), PINNConfig(strategy="medium"))


def test_boundary_error_detects_a_violation():
    problem = HeatProblem(boundary="dirichlet")
    model = lambda x, t: 1.0 + 0 * x + 0 * t  # noqa: E731
    model = type("M", (), {"problem": problem, "__call__": staticmethod(model)})()
    left, right = boundary_error(model, torch.zeros(3, 1, dtype=DTYPE))
    assert torch.allclose(left, torch.ones_like(left))
    assert torch.allclose(right, torch.ones_like(right))
