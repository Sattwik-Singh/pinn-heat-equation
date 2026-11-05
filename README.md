# PINNs vs. Finite Differences for the 1D Heat Equation

How accurately, and at what cost, can a physics-informed neural network (PINN)
solve the heat equation compared with classical finite-difference schemes? And
how much does the way boundary conditions are enforced change the answer?

The study solves

$$u_t = \alpha\,u_{xx}, \qquad x \in [0, 1],\ t \in [0, 1],\ \alpha = 0.1$$

with Dirichlet, Neumann and mixed boundary conditions. Every test problem has an
exact series solution, so all errors are measured against the true answer.

## Experiments

| Study | Question | Design |
|---|---|---|
| Boundary enforcement | Does soft or hard enforcement give better accuracy? | 3 boundary types, soft weights 1/10/100 and hard, 5 seeds |
| Numerical comparison | How do error and runtime compare with FTCS and Crank-Nicolson? | Grid refinement from 10 to 640 cells |
| Frequency | How does accuracy change for higher modes? | Modes 1, 2, 4, 8 (hard); 4, 8 (soft), 3 seeds |
| Collocation | How many residual points are needed? | 50 to 4000 points, 3 seeds |
| Inverse problem | Can alpha be recovered from noisy sensors? | PINN vs. Crank-Nicolson fit, 0/1/5% noise, 5 seeds |

Results and figures will be added here when the full study has been checked.

## Method

- **Network:** tanh MLP with 4 hidden layers of 50 units, in double precision.
- **Training:** 3000 Adam steps with cosine decay, then up to 1000 L-BFGS iterations.
- **Soft enforcement:** minimises `L_pde + w * (L_bc + L_ic)`.
- **Hard enforcement:** builds the conditions into the output, so only `L_pde` is minimised:

$$\hat u(x, t) = u_0(x) + \frac{t}{T}\, g(x)\, N_\theta(\phi(x), t)$$

| Boundary | `g(x)` | Input feature `phi(x)` |
|---|---|---|
| Dirichlet | `4x(1 - x)` | `2x - 1` |
| Neumann | `1` | `cos(pi x)` |
| Mixed | `sin(pi x / 2)` | `2 sin(pi x / 2) - 1` |

## Reproduce

```bash
pip install -e ".[dev]"
pytest
PYTHONPATH=src python experiments/run_all.py --workers 2
```

The full study runs 105 training jobs and takes about 1.5 to 2 hours on 2 CPU
cores. Runs are seeded, and finished jobs are skipped if a run is restarted.

## Layout

```text
src/heatpinn/problems.py   Test problems and exact solutions
src/heatpinn/numerics.py   FTCS and Crank-Nicolson solvers
src/heatpinn/pinn.py       PINN models, training and evaluation
experiments/run_all.py     Full experiment runner
results/                   Raw results as JSON lines
tests/                     Unit tests
```
