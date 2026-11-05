"""Run every experiment in the study and append records to results/*.jsonl.

Finished jobs are skipped, so an interrupted run can simply be restarted.

    python experiments/run_all.py --workers 2
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar

from heatpinn.numerics import relative_l2_error, solve_field
from heatpinn.pinn import DTYPE, PINNConfig, train, train_inverse
from heatpinn.problems import BOUNDARY_TYPES, HeatProblem

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASE = dict(adam_steps=3000, lbfgs_steps=1000)
STRATEGIES = {
    "soft (w=1)": dict(strategy="soft", bc_weight=1.0),
    "soft (w=10)": dict(strategy="soft", bc_weight=10.0),
    "soft (w=100)": dict(strategy="soft", bc_weight=100.0),
    "hard": dict(strategy="hard"),
}
SENSOR_X = (0.1, 0.3, 0.5, 0.7, 0.9)
SENSOR_T = tuple(np.linspace(0.1, 1.0, 10).round(2).tolist())


def single_mode(n: int) -> tuple[float, ...]:
    return tuple(1.0 if i == n - 1 else 0.0 for i in range(n))


def save_error_field(model, problem: HeatProblem, name: str) -> None:
    x = torch.linspace(0, 1, 201, dtype=DTYPE)
    t = torch.linspace(0, problem.t_end, 101, dtype=DTYPE)
    xx, tt = torch.meshgrid(x, t, indexing="xy")
    with torch.no_grad():
        u = model(xx.reshape(-1, 1), tt.reshape(-1, 1)).numpy().reshape(xx.shape)
    error = u - problem.exact(xx.numpy(), tt.numpy())
    directory = RESULTS / "fields"
    directory.mkdir(exist_ok=True)
    np.savez_compressed(directory / f"{name}.npz", error=error.astype(np.float32))


def boundary_job(boundary: str, label: str, seed: int) -> dict:
    problem = HeatProblem(boundary=boundary)
    model, record = train(problem, PINNConfig(seed=seed, **BASE, **STRATEGIES[label]))
    if seed == 0 and label in ("soft (w=10)", "hard"):
        save_error_field(model, problem, f"{boundary}_{label.split()[0]}")
    return record


def frequency_job(mode: int, seed: int) -> dict:
    problem = HeatProblem(amplitudes=single_mode(mode))
    return train(problem, PINNConfig(seed=seed, **BASE))[1]


def frequency_soft_job(mode: int, seed: int) -> dict:
    """Check whether high-mode failure is specific to hard enforcement."""
    problem = HeatProblem(amplitudes=single_mode(mode))
    config = PINNConfig(seed=seed, **BASE, **STRATEGIES["soft (w=10)"])
    return train(problem, config)[1]


def collocation_job(n_collocation: int, seed: int) -> dict:
    config = PINNConfig(seed=seed, n_collocation=n_collocation, **BASE)
    return train(HeatProblem(), config)[1]


def sensor_data(problem: HeatProblem, noise: float, seed: int):
    xx, tt = np.meshgrid(SENSOR_X, SENSOR_T, indexing="xy")
    x, t = xx.ravel(), tt.ravel()
    clean = problem.exact(x, t)
    rng = np.random.default_rng(seed)
    return x, t, clean + noise * np.sqrt(np.mean(clean**2)) * rng.standard_normal(clean.shape)


def fd_inverse(problem: HeatProblem, x, t, u) -> tuple[float, float]:
    """Fit alpha by bounded least squares around a Crank-Nicolson solve."""
    points = np.column_stack([t, x])

    def misfit(log_alpha: float) -> float:
        trial = HeatProblem(problem.boundary, float(np.exp(log_alpha)), problem.t_end, problem.amplitudes)
        grid_x, grid_t, field = solve_field(trial, 100, 200)
        predicted = RegularGridInterpolator((grid_t, grid_x), field)(points)
        return float(np.mean((predicted - u) ** 2))

    start = time.perf_counter()
    result = minimize_scalar(misfit, bounds=(np.log(1e-3), np.log(1.0)), method="bounded")
    return float(np.exp(result.x)), time.perf_counter() - start


def inverse_job(noise: float, seed: int) -> dict:
    problem = HeatProblem()
    x, t, u = sensor_data(problem, noise, seed)
    config = PINNConfig(seed=seed, **BASE)
    _, record = train_inverse(problem, config, x, t, u, initial_alpha=0.3)
    fd_alpha, fd_seconds = fd_inverse(problem, x, t, u)
    return {**record, "fd_alpha": fd_alpha, "fd_seconds": fd_seconds}


def fd_sweep() -> None:
    path = RESULTS / "finite_difference.jsonl"
    if path.exists():
        return
    records = []
    for boundary in BOUNDARY_TYPES:
        problem = HeatProblem(boundary=boundary)
        for n_cells in (10, 20, 40, 80, 160, 320, 640):
            n_steps_by_scheme = {
                "crank-nicolson": n_cells,
                "ftcs": int(np.ceil(problem.alpha * problem.t_end * n_cells**2 / 0.4)),
            }
            for scheme, n_steps in n_steps_by_scheme.items():
                times = []
                for _ in range(5):
                    start = time.perf_counter()
                    x, t, field = solve_field(problem, n_cells, n_steps, scheme)
                    times.append(time.perf_counter() - start)
                exact = problem.exact(x[None, :], t[:, None])
                records.append(
                    dict(
                        boundary=boundary,
                        scheme=scheme,
                        n_cells=n_cells,
                        n_steps=n_steps,
                        unknowns=(n_cells + 1) * n_steps,
                        relative_l2=relative_l2_error(field, exact),
                        seconds=float(np.median(times)),
                    )
                )
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def run_job(job: tuple[str, dict]) -> str:
    name, params = job
    torch.set_num_threads(1)
    function = {
        "boundary": boundary_job,
        "frequency": frequency_job,
        "frequency_soft": frequency_soft_job,
        "collocation": collocation_job,
        "inverse": inverse_job,
    }[name]
    record = function(**params)
    with open(RESULTS / f"{name}.jsonl", "a") as handle:
        handle.write(json.dumps({**params, **record}) + "\n")
    return f"{name} {params} done"


def all_jobs() -> list[tuple[str, dict]]:
    jobs = []
    for boundary in BOUNDARY_TYPES:
        for label in STRATEGIES:
            jobs += [("boundary", dict(boundary=boundary, label=label, seed=s)) for s in range(5)]
    jobs += [("frequency", dict(mode=m, seed=s)) for m in (1, 2, 4, 8) for s in range(3)]
    jobs += [("frequency_soft", dict(mode=m, seed=s)) for m in (4, 8) for s in range(3)]
    jobs += [("collocation", dict(n_collocation=n, seed=s)) for n in (50, 200, 1000, 4000) for s in range(3)]
    jobs += [("inverse", dict(noise=n, seed=s)) for n in (0.0, 0.01, 0.05) for s in range(5)]
    return jobs


def completed(job: tuple[str, dict]) -> bool:
    name, params = job
    path = RESULTS / f"{name}.jsonl"
    if not path.exists():
        return False
    return any(
        all(json.loads(line).get(k) == v for k, v in params.items())
        for line in path.read_text().splitlines()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    fd_sweep()
    jobs = [job for job in all_jobs() if not completed(job)]
    print(f"{len(jobs)} jobs to run", flush=True)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, message in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {message}", flush=True)


if __name__ == "__main__":
    main()
