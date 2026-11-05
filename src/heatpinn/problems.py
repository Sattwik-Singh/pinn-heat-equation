"""Heat-equation test problems with exact series solutions.

Each problem solves u_t = alpha * u_xx on x in [0, 1], t in [0, t_end], with
homogeneous boundary conditions of one of three types:

    dirichlet: u(0, t) = 0,   u(1, t) = 0
    neumann:   u_x(0, t) = 0, u_x(1, t) = 0
    mixed:     u(0, t) = 0,   u_x(1, t) = 0
"""

from dataclasses import dataclass

import numpy as np

BOUNDARY_TYPES = ("dirichlet", "neumann", "mixed")


@dataclass(frozen=True)
class HeatProblem:
    """A problem defined by the amplitudes of its eigenmodes (n = 1, 2, ...)."""

    boundary: str = "dirichlet"
    alpha: float = 0.1
    t_end: float = 1.0
    amplitudes: tuple[float, ...] = (1.0, 0.5)

    def __post_init__(self) -> None:
        if self.boundary not in BOUNDARY_TYPES:
            raise ValueError(f"unknown boundary type: {self.boundary!r}")
        if self.alpha <= 0 or self.t_end <= 0:
            raise ValueError("alpha and t_end must be positive")

    def wavenumbers(self) -> np.ndarray:
        n = np.arange(1, len(self.amplitudes) + 1, dtype=float)
        return (n - 0.5) * np.pi if self.boundary == "mixed" else n * np.pi

    def _mode(self, k: float, x: np.ndarray) -> np.ndarray:
        return np.cos(k * x) if self.boundary == "neumann" else np.sin(k * x)

    def exact(self, x, t) -> np.ndarray:
        """Evaluate the exact solution; x and t broadcast against each other."""
        x, t = np.asarray(x, dtype=float), np.asarray(t, dtype=float)
        return sum(
            a * np.exp(-self.alpha * k**2 * t) * self._mode(k, x)
            for a, k in zip(self.amplitudes, self.wavenumbers())
        )

    def initial(self, x) -> np.ndarray:
        return self.exact(x, 0.0)
