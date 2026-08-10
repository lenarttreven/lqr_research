"""Reproduce and tweak a benchmark plot from a saved ``.npz`` result file.

This script is intentionally standalone so it is easy to tweak without
touching the generic benchmark plotting entrypoint.

Examples
--------
    # Reproduce the aircraft-pitch plot interactively
    ./.venv/bin/python plot_cdc.py

    # Save a publication figure without opening a window
    ./.venv/bin/python plot_cdc.py --no-show --save-path figures/cdc_aircraft_pitch.pdf

    # Match the generic benchmark script more closely
    ./.venv/bin/python plot_cdc.py --log-y
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt


plt.rc("text", usetex=True)
plt.rc(
    "text.latex",
    preamble=r"\usepackage{amsmath}"
    r"\usepackage{bm}"
    r"\def\vx{{\bm{x}}}"
    r"\def\vf{{\bm{f}}}",
)
import matplotlib as mpl


LEGEND_FONT_SIZE = 18
X_TICKS_FONT_SIZE = 18
AX_TITLE_FONT_SIZE = 24
FIGURE_TITLE_FONT_SIZE = 30
TABLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 20
TICKS_SIZE = 24
LINE_WIDTH = 3

mpl.rcParams["xtick.labelsize"] = TICKS_SIZE
mpl.rcParams["ytick.labelsize"] = TICKS_SIZE

DEFAULT_FILE = Path("results/aircraft_pitch.npz")
DEFAULT_ALGO_ORDER = ("IR-LQR", "TS", "CEC+PE", "LagLQ", "OSLO")
DEFAULT_LINESTYLES = ("-", "--", ":", "-.")
ALGORITHM_STYLES = {
    "IR-LQR": {"label": "IR-LQR (ours)"},
    "TS": {"label": "TS"},
    "CEC+PE": {"label": "CEC+PE"},
    "LagLQ": {"label": "LagLQ"},
    "OSLO": {"label": "OSLO"},
}


def load_benchmark_npz(
    path: str | Path,
) -> tuple[str, dict[str, dict[str, np.ndarray]]]:
    """Load one benchmark result file.

    The saved format is the flattened schema produced by ``run_benchmark.py``:
    ``algorithm_name/key -> array`` plus metadata entries.
    """
    data = np.load(path, allow_pickle=True)
    system_name = str(data["_system_name"])
    algo_names = [str(name) for name in data["_algo_names"]]

    results: dict[str, dict[str, np.ndarray]] = {}
    for name in algo_names:
        prefix = name.replace(" ", "_")
        algo_results: dict[str, np.ndarray] = {}
        for full_key in data.files:
            if full_key.startswith(f"{prefix}/"):
                short_key = full_key[len(prefix) + 1 :]
                algo_results[short_key] = np.asarray(data[full_key])
        results[name] = algo_results

    return system_name, results


def reorder_algorithms(
    results: dict[str, dict[str, np.ndarray]],
    algorithm_order: list[str] | tuple[str, ...] | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Return a reordered copy of ``results`` while keeping unknown names."""
    if not algorithm_order:
        return dict(results)

    ordered: dict[str, dict[str, np.ndarray]] = {}
    for name in algorithm_order:
        if name in results:
            ordered[name] = results[name]
    for name, algo_results in results.items():
        if name not in ordered:
            ordered[name] = algo_results
    return ordered


def quantile_band(
    samples: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower / median / upper summaries across the trial axis."""
    lower, median, upper = np.quantile(
        samples,
        [lower_quantile, 0.5, upper_quantile],
        axis=0,
    )
    return lower, median, upper


def load_timing_pkl(
    path: str | Path,
    system_name: str | None = None,
) -> dict[str, dict] | None:
    """Load timing results from a pickle file saved by ``run_timing.py``.

    Returns ``{algo_name: algo_results}`` for the requested system,
    or for the first (only) system if *system_name* is ``None``.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    if system_name is not None:
        return results.get(system_name)
    # Return the first system's results
    return next(iter(results.values()))


def plot_benchmark_file(
    results_dict: dict[str, dict[str, np.ndarray]],
    *,
    title: str = "",
    log_y: bool = True,
    symlog_regret: bool = False,
    save_path: str | Path | None = None,
    plot_cost: bool = False,
    timing_data: dict[str, dict] | None = None,
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
    show: bool = True,
) -> None:
    """Plot cumulative regret, optional excess cost, controller updates, and timing."""
    # plt = configure_matplotlib(show=show)

    ncols = 1 + int(plot_cost) + 1 + int(timing_data is not None)
    figsize = (6.5 * ncols, 4.0)
    fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)

    col = 0
    ax_regret = axes[0, col]
    col += 1
    ax_cost = axes[0, col] if plot_cost else None
    if plot_cost:
        col += 1
    ax_updates = axes[0, col]
    col += 1
    ax_timing = axes[0, col] if timing_data is not None else None

    # Track per-algorithm colors so the timing bar chart matches
    algo_colors: dict[str, str] = {}

    for index, (name, results) in enumerate(results_dict.items()):
        style = ALGORITHM_STYLES.get(name, {})
        linestyle = style.get(
            "linestyle",
            DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)],
        )
        label = style.get("label", name)
        color = style.get("color")

        t = np.arange(results["regrets"].shape[1])
        line_kwargs = {"label": label, "linestyle": linestyle, "linewidth": LINE_WIDTH}
        if color is not None:
            line_kwargs["color"] = color

        cumulative_regret = np.cumsum(results["regrets"], axis=1)
        cumulative_regret = np.clip(cumulative_regret, a_min=-1e30, a_max=1e30)
        q_low, q_med, q_high = quantile_band(
            cumulative_regret,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        (regret_line,) = ax_regret.plot(t, q_med, **line_kwargs)
        line_color = regret_line.get_color()
        algo_colors[name] = line_color
        ax_regret.fill_between(
            t,
            q_low,
            q_high,
            alpha=0.15,
            color=line_color,
        )
        ax_regret.set_ylim(200)

        if plot_cost:
            excess_cost = results["expected_costs"] - results["optimal_cost"][:, None]
            q_low, q_med, q_high = quantile_band(
                excess_cost,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            (cost_line,) = ax_cost.plot(t, q_med, **line_kwargs)
            ax_cost.fill_between(
                t,
                q_low,
                q_high,
                alpha=0.15,
                color=cost_line.get_color(),
            )

        cumulative_updates = results["cum_updates"]
        q_low, q_med, q_high = quantile_band(
            cumulative_updates,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        (updates_line,) = ax_updates.plot(t, q_med, **line_kwargs)
        ax_updates.fill_between(
            t,
            q_low,
            q_high,
            alpha=0.15,
            color=updates_line.get_color(),
        )

    # Timing bar chart
    if ax_timing is not None and timing_data is not None:
        algo_names = [n for n in results_dict if n in timing_data]
        medians_ms = []
        q20_ms = []
        q80_ms = []
        bar_colors = []
        for name in algo_names:
            times = np.array(timing_data[name]["update_times"])
            if len(times) > 0:
                medians_ms.append(float(np.median(times)) * 1e3)
                q20_ms.append(float(np.quantile(times, lower_quantile)) * 1e3)
                q80_ms.append(float(np.quantile(times, upper_quantile)) * 1e3)
            else:
                medians_ms.append(0.0)
                q20_ms.append(0.0)
                q80_ms.append(0.0)
            bar_colors.append(algo_colors.get(name, None))

        medians_ms = np.array(medians_ms)
        q20_ms = np.array(q20_ms)
        q80_ms = np.array(q80_ms)
        x = np.arange(len(algo_names))
        yerr = np.array([medians_ms - q20_ms, q80_ms - medians_ms])
        bars = ax_timing.bar(x, medians_ms, yerr=yerr, capsize=4, color=bar_colors)
        labels = [ALGORITHM_STYLES.get(n, {}).get("label", n) for n in algo_names]
        ax_timing.set_xticks(x)
        ax_timing.set_xticklabels(
            labels, rotation=30, ha="right", fontsize=X_TICKS_FONT_SIZE
        )
        ax_timing.set_yscale("log")
        ax_timing.set_ylabel("Update time (ms)", fontsize=LABEL_FONT_SIZE)
        ax_timing.set_title("Controller Update Time", fontsize=AX_TITLE_FONT_SIZE)

    active_axes = [ax_regret, ax_updates]
    if plot_cost:
        active_axes.insert(1, ax_cost)

    for ax in active_axes:
        ax.set_xlabel("Time step", fontsize=LABEL_FONT_SIZE)

    if symlog_regret:
        ax_regret.set_yscale("symlog", linthresh=1.0)
    elif log_y:
        ax_regret.set_yscale("log")

    if plot_cost and log_y:
        ax_cost.set_yscale("log")
    if log_y:
        # ax_updates.set_yscale("log")
        pass

    ax_regret.set_ylabel(r"$\mathcal{R}_T$", fontsize=LABEL_FONT_SIZE)
    ax_regret.set_title("Cumulative Regret", fontsize=AX_TITLE_FONT_SIZE)

    if plot_cost:
        ax_cost.set_ylabel("Expected Cost - Optimal Cost")
        ax_cost.set_title("Expected Per-Step Excess Cost")

    ax_updates.set_ylabel("Controller Updates", fontsize=LABEL_FONT_SIZE)
    ax_updates.set_title("Cumulative Controller Updates", fontsize=AX_TITLE_FONT_SIZE)

    if title:
        fig.suptitle("Aircraft pitch", fontsize=FIGURE_TITLE_FONT_SIZE)
        legendpos = 0.88
    else:
        legendpos = 1.01

    fig.tight_layout(rect=[0, 0, 1, 0.88])
    handles, labels = ax_regret.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, legendpos),
        fontsize=LEGEND_FONT_SIZE,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce and customize a benchmark plot from a saved .npz file."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"Benchmark result file to plot. Default: {DEFAULT_FILE}.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="If set, save the figure to this path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override the figure title. Defaults to the system name in the file.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Suppress the figure title.",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="*",
        default=None,
        help="Optional ordered subset of algorithm names to plot.",
    )
    parser.add_argument(
        "--plot-cost",
        action="store_true",
        help="Add the expected excess-cost panel.",
    )
    parser.add_argument(
        "--quantile-band",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.2, 0.8),
        help="Lower and upper quantiles for the shaded band. Default: 0.2 0.8.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        default=True,
        help="Use log scale on the y-axes. Matches plot_benchmark.py.",
    )
    parser.add_argument(
        "--no-log-y",
        action="store_false",
        dest="log_y",
        help="Use linear y-axes instead.",
    )
    parser.add_argument(
        "--symlog-regret",
        action="store_true",
        help="Use a symmetric log scale on cumulative regret so negative values remain visible.",
    )
    parser.add_argument(
        "--no-show",
        action="store_false",
        dest="show",
        default=True,
        help="Do not open an interactive window.",
    )
    parser.add_argument(
        "--timing-file",
        type=Path,
        default=None,
        help="Path to a timing .pkl file saved by run_timing.py.",
    )
    parser.add_argument(
        "--timing-system",
        type=str,
        default=None,
        help="System name to use from the timing file. Defaults to the first system.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    lower_quantile, upper_quantile = args.quantile_band
    if not (0.0 <= lower_quantile < 0.5 < upper_quantile <= 1.0):
        parser.error(
            "--quantile-band must satisfy LOW < 0.5 < HIGH and stay within [0, 1]."
        )

    system_name, results = load_benchmark_npz(args.file)
    algorithm_order = args.algorithms or list(DEFAULT_ALGO_ORDER)
    results = reorder_algorithms(results, algorithm_order)

    if args.no_title:
        title = ""
    else:
        title = args.title if args.title is not None else system_name

    timing_data = None
    if args.timing_file is not None:
        timing_data = load_timing_pkl(args.timing_file, system_name=args.timing_system)

    plot_benchmark_file(
        results,
        title=title,
        log_y=args.log_y,
        symlog_regret=args.symlog_regret,
        save_path=args.save_path,
        plot_cost=args.plot_cost,
        timing_data=timing_data,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        show=args.show,
    )


if __name__ == "__main__":
    main()
