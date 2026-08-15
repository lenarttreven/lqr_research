"""Create a combined CDC figure for the aircraft-pitch and UAV-2D systems."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

from plot_one_system import (
    ALGORITHM_STYLES,
    AX_TITLE_FONT_SIZE,
    DEFAULT_ALGO_ORDER,
    DEFAULT_LINESTYLES,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    LINE_WIDTH,
    X_TICKS_FONT_SIZE,
    load_benchmark_npz,
    load_timing_pkl,
    reorder_algorithms,
)

# ---------------------------------------------------------------------------
# Hyperparameter: choose the aggregation method for line plots.
#   True  -> median + bootstrapped 95 % CI
#   False -> mean + standard error of the mean (SEM)
# ---------------------------------------------------------------------------
USE_MEDIAN_AND_BOOTSTRAP = True

N_BOOT = 2000
ALPHA = 0.3


def bootstrap_median_band(
    samples: np.ndarray,
    n_boot: int = N_BOOT,
    lo_pct: float = 20,
    hi_pct: float = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo, median, hi) using a bootstrap confidence interval on the median."""
    n = samples.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(samples[idx], axis=1)
    median = np.median(samples, axis=0)
    lo = np.percentile(boot_medians, lo_pct, axis=0)
    hi = np.percentile(boot_medians, hi_pct, axis=0)
    return lo, median, hi


def mean_sem_band(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lo, mean, hi) where the band is mean ± SEM."""
    mean = np.mean(samples, axis=0)
    sem = np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])
    return mean - sem, mean, mean + sem


def _compute_band(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the chosen aggregation method."""
    if USE_MEDIAN_AND_BOOTSTRAP:
        return bootstrap_median_band(samples)
    return mean_sem_band(samples)


def set_two_power_ticks(ax) -> None:
    """Label a log timing axis only at 10^0 and 10^1 milliseconds."""
    exponents = (0, 1)
    ticks = [10.0**exponent for exponent in exponents]
    labels = [rf"$10^{{{exponent}}}$" for exponent in exponents]
    ax.set_yticks(ticks, labels=labels)
    ax.yaxis.set_minor_formatter(NullFormatter())


SYSTEM_NAME_FONTSIZE = 19
LABEL_FONT_SIZE = 18
AX_TITLE_FONT_SIZE = 18


# Keep the standalone script runnable without a full TeX installation.


# Keep the standalone script runnable without a full TeX installation.
plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"\usepackage{amsmath}"
    r"\usepackage{bm}"
    r"\def\vx{{\bm{x}}}"
    r"\def\vf{{\bm{f}}}",
)

LOWER_QUANTILE = 0.2
UPPER_QUANTILE = 0.8
EXCLUDED_ALGORITHMS = set()


def system_specs(results_dir: Path) -> tuple[dict, ...]:
    """Return the saved inputs produced by ``run_2_systems.py``."""
    return (
        {
            "label": r"\textbf{Aircraft pitch}",
            "results_file": results_dir / "aircraft_pitch.npz",
            "timing_file": results_dir / "timing_physical.pkl",
        },
        {
            "label": r"\textbf{UAV 2D}",
            "results_file": results_dir / "uav_2d.npz",
            "timing_file": results_dir / "timing_stabl.pkl",
        },
    )


def build_algorithm_colors(
    results_dict: dict[str, dict[str, np.ndarray]],
) -> dict[str, str | None]:
    """Assign stable colors so all rows/columns match."""
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors: dict[str, str | None] = {}
    for index, name in enumerate(results_dict):
        style = ALGORITHM_STYLES.get(name, {})
        color = style.get("color")
        if color is None and default_colors:
            color = default_colors[index % len(default_colors)]
        colors[name] = color
    return colors


def plot_system_row(
    axes_row: np.ndarray,
    results_dict: dict[str, dict[str, np.ndarray]],
    timing_data: dict[str, dict],
    *,
    algo_colors: dict[str, str | None],
    show_legend_labels: bool,
    show_titles: bool,
    row_index: int,
) -> None:
    """Plot one system into a 3-panel row."""
    ax_regret, ax_updates, ax_timing = axes_row

    for index, (name, results) in enumerate(results_dict.items()):
        style = ALGORITHM_STYLES.get(name, {})
        linestyle = style.get(
            "linestyle",
            DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)],
        )
        color = algo_colors.get(name)
        label = style.get("label", name) if show_legend_labels else None

        base_line_kwargs = {
            "linestyle": linestyle,
            "linewidth": LINE_WIDTH,
        }
        if color is not None:
            base_line_kwargs["color"] = color

        t = np.arange(results["regrets"].shape[1])

        cumulative_regret = np.cumsum(results["regrets"], axis=1)
        cumulative_regret = np.clip(cumulative_regret, a_min=-1e30, a_max=1e30)
        q_low, q_med, q_high = _compute_band(cumulative_regret)
        regret_line_kwargs = dict(base_line_kwargs)
        if label is not None:
            regret_line_kwargs["label"] = label
        ax_regret.plot(t, q_med, **regret_line_kwargs)
        ax_regret.fill_between(
            t,
            q_low,
            q_high,
            alpha=ALPHA,
            color=color,
        )

        cumulative_updates = results["cum_updates"]
        q_low, q_med, q_high = _compute_band(cumulative_updates)
        ax_updates.plot(t, q_med, **base_line_kwargs)
        ax_updates.fill_between(
            t,
            q_low,
            q_high,
            alpha=ALPHA,
            color=color,
        )

    algo_names = [name for name in results_dict if name in timing_data]
    medians_ms = []
    q20_ms = []
    q80_ms = []
    for name in algo_names:
        times = np.asarray(timing_data[name]["update_times"])
        if len(times) > 0:
            medians_ms.append(float(np.median(times)) * 1e3)
            q20_ms.append(float(np.quantile(times, LOWER_QUANTILE)) * 1e3)
            q80_ms.append(float(np.quantile(times, UPPER_QUANTILE)) * 1e3)
        else:
            medians_ms.append(0.0)
            q20_ms.append(0.0)
            q80_ms.append(0.0)

    medians_ms = np.asarray(medians_ms)
    q20_ms = np.asarray(q20_ms)
    q80_ms = np.asarray(q80_ms)
    x = np.arange(len(algo_names))
    yerr = np.array([medians_ms - q20_ms, q80_ms - medians_ms])
    ax_timing.bar(
        x,
        medians_ms,
        yerr=yerr,
        capsize=4,
        color=[algo_colors.get(name) for name in algo_names],
    )
    ax_timing.set_xticks(x)
    ax_timing.set_xticklabels(
        [ALGORITHM_STYLES.get(name, {}).get("label", name) for name in algo_names],
        rotation=30,
        ha="right",
        fontsize=X_TICKS_FONT_SIZE,
    )

    if row_index == 0:
        pass
        # ax_regret.set_ylim(200)
    elif row_index == 1:
        pass
        # ax_regret.set_ylim(1e-2)

    if row_index == 0:
        ax_regret.set_yscale("log")
    ax_timing.set_yscale("log")
    set_two_power_ticks(ax_timing)

    ax_regret.set_ylabel(r"$\mathcal{R}_T$", fontsize=LABEL_FONT_SIZE)
    ax_updates.set_ylabel("Num Updates", fontsize=LABEL_FONT_SIZE)
    ax_timing.set_ylabel("Time (ms)", fontsize=LABEL_FONT_SIZE)

    if show_titles:
        ax_regret.set_title("Cumulative Regret", fontsize=AX_TITLE_FONT_SIZE)
        ax_updates.set_title(
            "Cumulative Controller Updates",
            fontsize=AX_TITLE_FONT_SIZE,
        )
        ax_timing.set_title("Controller Update Time", fontsize=AX_TITLE_FONT_SIZE)


def plot_saved_results(
    *,
    results_dir: Path = Path("results"),
    save_path: Path = Path("cdc_2_systems.pdf"),
    show: bool = True,
) -> None:
    """Plot existing benchmark and timing files without running simulations."""
    ordered_systems = []
    for spec in system_specs(results_dir):
        for input_path in (spec["results_file"], spec["timing_file"]):
            if not input_path.is_file():
                raise FileNotFoundError(
                    f"missing {input_path}; run run_2_systems.py first"
                )
        system_name, results = load_benchmark_npz(spec["results_file"])
        ordered_results = reorder_algorithms(results, DEFAULT_ALGO_ORDER)
        filtered_results = {
            name: algo_results
            for name, algo_results in ordered_results.items()
            if name not in EXCLUDED_ALGORITHMS
        }
        ordered_systems.append(
            {
                "label": spec["label"],
                "system_name": system_name,
                "results": filtered_results,
                "timing": load_timing_pkl(spec["timing_file"], system_name=system_name),
            }
        )

    algo_colors = build_algorithm_colors(ordered_systems[0]["results"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 4.0), sharex="col", squeeze=False)

    for row_index, system in enumerate(ordered_systems):
        plot_system_row(
            axes[row_index],
            system["results"],
            system["timing"],
            algo_colors=algo_colors,
            show_legend_labels=row_index == 0,
            show_titles=row_index == 0,
            row_index=row_index,
        )

    for ax in axes[0]:
        ax.tick_params(labelbottom=False)

    axes[1, 0].set_xlabel(r"$T$", fontsize=LABEL_FONT_SIZE)
    axes[1, 1].set_xlabel(r"$T$", fontsize=LABEL_FONT_SIZE)

    fig.subplots_adjust(
        left=0.11, right=0.99, bottom=0.10, top=0.82, wspace=0.28, hspace=0.18
    )

    for row_index, system in enumerate(ordered_systems):
        bbox = axes[row_index, 0].get_position()
        fig.text(
            0.045,
            0.5 * (bbox.y0 + bbox.y1),
            system["label"],
            rotation=90,
            va="center",
            ha="center",
            fontsize=SYSTEM_NAME_FONTSIZE,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.05),
        fontsize=LEGEND_FONT_SIZE,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved outputs from run_2_systems.py without rerunning simulations."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figure-path", type=Path, default=Path("cdc_2_systems.pdf"))
    parser.add_argument("--no-show", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_saved_results(
        results_dir=args.results_dir,
        save_path=args.figure_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
