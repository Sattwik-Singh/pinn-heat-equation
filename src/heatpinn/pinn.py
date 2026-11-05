"""Physics-informed neural networks for the 1D heat equation.

Two ways of imposing the boundary and initial conditions are implemented:

soft: the network is unconstrained and violations are penalised in the loss,
      L = L_pde + weight * (L_bc + L_ic).
hard: the network output is transformed so that the conditions hold exactly
      for any parameters, leaving only L_pde to minimise.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from heatpinn.problems import HeatProblem

DTYPE = torch.float64


@dataclass(frozen=True)
class PINNConfig:
    strategy: str = "hard"  # "hard" or "soft"
    bc_weight: float = 1.0  # used by "soft" only
    width: int = 50
    depth: int = 4
    n_collocation: int = 2000
    n_boundary: int = 200
    n_initial: int = 200
    adam_steps: int = 5000
    learning_rate: float = 1e-3
    lbfgs_steps: int = 1000
    seed: int = 0


def exact_torch(problem: HeatProblem, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate the exact solution with differentiable torch operations."""
    out = torch.zeros_like(x + t)
    mode = torch.cos if problem.boundary == "neumann" else torch.sin
    for a, k in zip(problem.amplitudes, problem.wavenumbers()):
        out = out + a * torch.exp(-problem.alpha * float(k) ** 2 * t) * mode(float(k) * x)
    return out


class HeatPINN(nn.Module):
    """A tanh MLP u(x, t), optionally constrained to satisfy BCs and the IC."""

    def __init__(self, problem: HeatProblem, config: PINNConfig):
        super().__init__()
        if config.strategy not in ("soft", "hard"):
            raise ValueError(f"unknown strategy: {config.strategy!r}")
        self.problem = problem
        self.hard = config.strategy == "hard"
        layers: list[nn.Module] = [nn.Linear(2, config.width), nn.Tanh()]
        for _ in range(config.depth - 1):
            layers += [nn.Linear(config.width, config.width), nn.Tanh()]
        layers.append(nn.Linear(config.width, 1))
        self.net = nn.Sequential(*layers).to(DTYPE)

    def _boundary_transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the spatial input feature and the factor multiplying the net.

        dirichlet: x(1 - x) vanishes at both ends.
        neumann:   cos(pi x) has zero slope at both ends, so any function of it
                   also has zero slope there.
        mixed:     sin(pi x / 2) is zero at x = 0 and has zero slope at x = 1.
        """
        boundary = self.problem.boundary
        if boundary == "dirichlet":
            return 2 * x - 1, 4 * x * (1 - x)
        if boundary == "neumann":
            return torch.cos(math.pi * x), torch.ones_like(x)
        s = torch.sin(0.5 * math.pi * x)
        return 2 * s - 1, s

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.problem.t_end
        if not self.hard:
            return self.net(torch.cat([2 * x - 1, 2 * tau - 1], dim=1))
        feature, factor = self._boundary_transform(x)
        raw = self.net(torch.cat([feature, 2 * tau - 1], dim=1))
        return exact_torch(self.problem, x, torch.zeros_like(x)) + tau * factor * raw


def gradient(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def pde_residual(model, x: torch.Tensor, t: torch.Tensor, alpha) -> torch.Tensor:
    """Return u_t - alpha * u_xx at the given points."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = model(x, t)
    u_x = gradient(u, x)
    return gradient(u, t) - alpha * gradient(u_x, x)


def boundary_error(model, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the residuals of the left and right boundary conditions."""
    boundary = model.problem.boundary
    x0 = torch.zeros_like(t, requires_grad=True)
    x1 = torch.ones_like(t, requires_grad=True)
    u0, u1 = model(x0, t), model(x1, t)
    left = gradient(u0, x0) if boundary == "neumann" else u0
    right = u1 if boundary == "dirichlet" else gradient(u1, x1)
    return left, right


def sobol_points(n: int, dim: int, seed: int) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    return engine.draw(n).to(DTYPE)


def evaluate(model: HeatPINN, n_x: int = 201, n_t: int = 101) -> dict[str, float]:
    """Compare the model with the exact solution on a uniform space-time grid."""
    problem = model.problem
    x = torch.linspace(0, 1, n_x, dtype=DTYPE)
    t = torch.linspace(0, problem.t_end, n_t, dtype=DTYPE)
    xx, tt = torch.meshgrid(x, t, indexing="xy")
    with torch.no_grad():
        predicted = model(xx.reshape(-1, 1), tt.reshape(-1, 1)).numpy().ravel()
    exact = problem.exact(xx.numpy().ravel(), tt.numpy().ravel())
    left, right = boundary_error(model, t.reshape(-1, 1).clone())
    ic = model(x.reshape(-1, 1), torch.zeros(n_x, 1, dtype=DTYPE)).detach().numpy().ravel()
    return {
        "relative_l2": float(np.linalg.norm(predicted - exact) / np.linalg.norm(exact)),
        "max_abs_error": float(np.max(np.abs(predicted - exact))),
        "bc_rms": float(torch.cat([left, right]).pow(2).mean().sqrt().detach()),
        "ic_rms": float(np.sqrt(np.mean((ic - problem.initial(x.numpy())) ** 2))),
    }


def _adam(params, closure, config: PINNConfig, log_every: int = 100):
    """Run Adam with cosine decay; return the loss history and elapsed time."""
    history: list[float] = []
    adam = torch.optim.Adam(params, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, max(config.adam_steps, 1))
    start = time.perf_counter()
    for step in range(config.adam_steps):
        adam.zero_grad()
        loss = closure()
        loss.backward()
        adam.step()
        scheduler.step()
        if step % log_every == 0:
            history.append(float(loss.detach()))
    return history, time.perf_counter() - start


def _lbfgs(params, closure, steps: int) -> float:
    if steps == 0:
        return 0.0
    optimiser = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=steps,
        max_eval=int(1.25 * steps),
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def step_closure():
        optimiser.zero_grad()
        loss = closure()
        loss.backward()
        return loss

    start = time.perf_counter()
    optimiser.step(step_closure)
    return time.perf_counter() - start


def train(problem: HeatProblem, config: PINNConfig) -> tuple[HeatPINN, dict]:
    """Train a forward PINN and return it with its evaluation record."""
    torch.manual_seed(config.seed)
    model = HeatPINN(problem, config)
    points = sobol_points(config.n_collocation, 2, config.seed)
    x_pde, t_pde = points[:, :1], points[:, 1:] * problem.t_end
    t_bc = sobol_points(config.n_boundary, 1, config.seed + 1) * problem.t_end
    x_ic = sobol_points(config.n_initial, 1, config.seed + 2)
    t_ic = torch.zeros_like(x_ic)
    u_ic = exact_torch(problem, x_ic, t_ic)

    def closure() -> torch.Tensor:
        loss = pde_residual(model, x_pde, t_pde, problem.alpha).pow(2).mean()
        if not model.hard:
            left, right = boundary_error(model, t_bc)
            ic = model(x_ic, t_ic) - u_ic
            penalty = left.pow(2).mean() + right.pow(2).mean() + ic.pow(2).mean()
            loss = loss + config.bc_weight * penalty
        return loss

    params = list(model.parameters())
    history, adam_seconds = _adam(params, closure, config)
    after_adam = evaluate(model)
    lbfgs_seconds = _lbfgs(params, closure, config.lbfgs_steps)
    final = evaluate(model)
    return model, {
        **final,
        "relative_l2_adam": after_adam["relative_l2"],
        "adam_seconds": adam_seconds,
        "total_seconds": adam_seconds + lbfgs_seconds,
        "final_loss": float(closure().detach()),
        "loss_history": history,
        "n_parameters": sum(p.numel() for p in params),
    }


def train_inverse(
    problem: HeatProblem,
    config: PINNConfig,
    x_obs: np.ndarray,
    t_obs: np.ndarray,
    u_obs: np.ndarray,
    initial_alpha: float,
    data_weight: float = 100.0,
) -> tuple[HeatPINN, dict]:
    """Recover alpha from sparse measurements with a hard-constrained PINN.

    The boundary and initial conditions are treated as known; the value of
    problem.alpha is not used in training, only in the reported field error.
    """
    torch.manual_seed(config.seed)
    model = HeatPINN(problem, config)
    log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha), dtype=DTYPE))
    points = sobol_points(config.n_collocation, 2, config.seed)
    x_pde, t_pde = points[:, :1], points[:, 1:] * problem.t_end
    x_data = torch.tensor(x_obs, dtype=DTYPE).reshape(-1, 1)
    t_data = torch.tensor(t_obs, dtype=DTYPE).reshape(-1, 1)
    u_data = torch.tensor(u_obs, dtype=DTYPE).reshape(-1, 1)

    def closure() -> torch.Tensor:
        pde = pde_residual(model, x_pde, t_pde, log_alpha.exp()).pow(2).mean()
        return pde + data_weight * (model(x_data, t_data) - u_data).pow(2).mean()

    params = [*model.parameters(), log_alpha]
    _, adam_seconds = _adam(params, closure, config)
    lbfgs_seconds = _lbfgs(params, closure, config.lbfgs_steps)
    return model, {
        **evaluate(model),
        "alpha": float(log_alpha.exp().detach()),
        "total_seconds": adam_seconds + lbfgs_seconds,
    }
