"""Build every figure and the summary table from the raw results."""

import collections
import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID, INK, MUTED = "#e4e3df", "#0b0b0b", "#52514e"
BOUNDARIES = ("dirichlet", "neumann", "mixed")
LABELS = ("soft (w=1)", "soft (w=10)", "soft (w=100)", "hard")
DIVERGING = LinearSegmentedColormap.from_list("error", [BLUE, "#f0efec", ORANGE])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    }
)


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def group(records, *keys):
    groups = collections.defaultdict(list)
    for record in records:
        groups[tuple(record[k] for k in keys)].append(record)
    return groups


def percent(records) -> list[float]:
    return [100 * r["relative_l2"] for r in records]


def save(figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(figure)


def boundary_figure(records) -> None:
    groups = group(records, "boundary", "label")
    figure, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), sharey=True)
    for ax, boundary in zip(axes, BOUNDARIES):
        for i, label in enumerate(LABELS):
            values = percent(groups[boundary, label])
            color = BLUE if label == "hard" else ORANGE
            offsets = i + np.linspace(-0.14, 0.14, len(values))
            ax.scatter(offsets, values, s=13, color=color, alpha=0.55, linewidths=0)
            ax.plot([i - 0.25, i + 0.25], [median(values)] * 2, color=color)
        ax.set_yscale("log")
        ax.set_xticks(range(4), ["1", "10", "100", "hard"])
        ax.set_xlabel("Soft weight w, or hard")
        ax.set_title(boundary.capitalize(), fontsize=9)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Relative L2 error (%)")
    save(figure, "boundary_enforcement")


def cost_figure(boundary_records, fd_records) -> None:
    figure, ax = plt.subplots(figsize=(4.6, 3.0))
    styles = {"crank-nicolson": ("Crank-Nicolson", BLUE, "o"), "ftcs": ("FTCS", AQUA, "s")}
    for scheme, (name, color, marker) in styles.items():
        rows = [r for r in fd_records if r["boundary"] == "dirichlet" and r["scheme"] == scheme]
        ax.plot(
            [r["seconds"] for r in rows],
            percent(rows),
            marker=marker,
            markersize=4,
            color=color,
            label=name,
        )
    for label, name, marker in (("hard", "PINN, hard", "D"), ("soft (w=10)", "PINN, soft (w=10)", "^")):
        rows = [r for r in boundary_records if r["boundary"] == "dirichlet" and r["label"] == label]
        ax.scatter(
            [r["total_seconds"] for r in rows],
            percent(rows),
            s=20,
            marker=marker,
            color=ORANGE,
            label=name,
        )
    ax.set(xscale="log", yscale="log", xlabel="Runtime on one CPU thread (s)", ylabel="Relative L2 error (%)")
    ax.legend(loc="upper center", ncols=2, fontsize=7.5)
    save(figure, "accuracy_vs_runtime")


def field_figure() -> None:
    fields = {
        (b, s): np.load(RESULTS / "fields" / f"{b}_{s}.npz")["error"] * 100
        for b in BOUNDARIES
        for s in ("soft", "hard")
    }
    limit = max(np.abs(v).max() for v in fields.values())
    norm = TwoSlopeNorm(0.0, -limit, limit)
    figure, axes = plt.subplots(
        2, 3, figsize=(6.8, 3.6), sharex=True, sharey=True, constrained_layout=True
    )
    for row, strategy in enumerate(("soft", "hard")):
        for column, boundary in enumerate(BOUNDARIES):
            ax = axes[row, column]
            image = ax.imshow(
                fields[boundary, strategy],
                extent=(0, 1, 0, 1),
                origin="lower",
                aspect="auto",
                cmap=DIVERGING,
                norm=norm,
            )
            ax.grid(visible=False)
            if row == 0:
                ax.set_title(boundary.capitalize(), fontsize=9)
            if row == 1:
                ax.set_xlabel("x")
        axes[row, 0].set_ylabel(("Soft (w=10)", "Hard")[row] + "\nt")
    figure.colorbar(image, ax=axes, label=r"$\hat{u}-u$ (units of $10^{-2}$)", shrink=0.85)
    save(figure, "error_fields")


def frequency_figure(hard, soft, collocation) -> None:
    figure, (left, right) = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for records, name, color, marker in ((hard, "Hard", BLUE, "D"), (soft, "Soft (w=10)", ORANGE, "^")):
        groups = group(records, "mode")
        for i, (mode,) in enumerate(sorted(groups)):
            left.scatter(
                [mode] * 3,
                percent(groups[mode,]),
                s=16,
                color=color,
                marker=marker,
                alpha=0.75,
                label=name if i == 0 else None,
            )
    left.set(
        xscale="log",
        yscale="log",
        xlabel="Eigenmode n",
        ylabel="Relative L2 error (%)",
        title="Higher modes decay faster",
    )
    left.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    left.minorticks_off()
    left.legend()
    groups = group(collocation, "n_collocation")
    keys = sorted(k[0] for k in groups)
    right.scatter(
        [k for k in keys for _ in groups[k,]],
        [v for k in keys for v in percent(groups[k,])],
        s=16,
        color=BLUE,
        marker="D",
        alpha=0.75,
    )
    right.set(
        xscale="log",
        yscale="log",
        xlabel="Collocation points",
        title="Error plateaus near 1000 points",
    )
    right.title.set_fontsize(9)
    left.title.set_fontsize(9)
    save(figure, "frequency_collocation")


def inverse_figure(records) -> None:
    groups = group(records, "noise")
    figure, ax = plt.subplots(figsize=(4.6, 2.5))
    noises = sorted(k[0] for k in groups)
    for i, noise in enumerate(noises):
        rows = groups[noise,]
        offsets = np.linspace(-0.06, 0.06, len(rows))
        ax.scatter(
            i - 0.14 + offsets,
            [r["alpha"] for r in rows],
            s=16,
            color=ORANGE,
            marker="^",
            label="PINN" if i == 0 else None,
        )
        ax.scatter(
            i + 0.14 + offsets,
            [r["fd_alpha"] for r in rows],
            s=16,
            color=BLUE,
            marker="o",
            label="Crank-Nicolson fit" if i == 0 else None,
        )
    ax.axhline(0.1, color=MUTED, linewidth=1, linestyle="--", label=r"True $\alpha$")
    ax.set_xticks(range(len(noises)), [f"{100 * n:.0f}% noise" for n in noises])
    ax.set_ylabel(r"Estimated $\alpha$")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", fontsize=7.5)
    save(figure, "inverse_alpha")


def summary(boundary, fd, hard, soft, collocation, inverse) -> dict:
    out: dict = {"boundary": {}, "frequency": {}, "collocation": {}, "inverse": {}, "fd": {}}
    for (b, label), rows in group(boundary, "boundary", "label").items():
        values = percent(rows)
        out["boundary"].setdefault(b, {})[label] = {
            "median_pct": median(values),
            "min_pct": min(values),
            "max_pct": max(values),
            "median_adam_pct": median(100 * r["relative_l2_adam"] for r in rows),
            "median_bc_rms": median(r["bc_rms"] for r in rows),
            "median_seconds": median(r["total_seconds"] for r in rows),
        }
    for name, records in (("hard", hard), ("soft", soft)):
        for (mode,), rows in group(records, "mode").items():
            out["frequency"].setdefault(name, {})[mode] = percent(rows)
    for (n,), rows in group(collocation, "n_collocation").items():
        out["collocation"][n] = {
            "median_pct": median(percent(rows)),
            "median_seconds": median(r["total_seconds"] for r in rows),
        }
    for (noise,), rows in group(inverse, "noise").items():
        pinn = np.array([r["alpha"] for r in rows])
        classical = np.array([r["fd_alpha"] for r in rows])
        out["inverse"][noise] = {
            "pinn_mean": pinn.mean(),
            "pinn_mean_abs_pct_error": 100 * np.abs(pinn / 0.1 - 1).mean(),
            "fd_mean": classical.mean(),
            "fd_mean_abs_pct_error": 100 * np.abs(classical / 0.1 - 1).mean(),
            "pinn_median_seconds": median(r["total_seconds"] for r in rows),
            "fd_median_seconds": median(r["fd_seconds"] for r in rows),
        }
    for r in fd:
        out["fd"].setdefault(r["boundary"], {}).setdefault(r["scheme"], {})[r["n_cells"]] = {
            "pct": 100 * r["relative_l2"],
            "ms": 1000 * r["seconds"],
        }
    return out


def main() -> None:
    boundary, fd = load("boundary"), load("finite_difference")
    hard, soft = load("frequency"), load("frequency_soft")
    collocation, inverse = load("collocation"), load("inverse")
    boundary_figure(boundary)
    cost_figure(boundary, fd)
    field_figure()
    frequency_figure(hard, soft, collocation)
    inverse_figure(inverse)
    text = json.dumps(summary(boundary, fd, hard, soft, collocation, inverse), indent=1)
    (RESULTS / "summary.json").write_text(text)


if __name__ == "__main__":
    main()
