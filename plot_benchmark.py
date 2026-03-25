"""Plot benchmark results from saved .npz files.

Usage:
    # Plot all systems in the results/ directory
    python plot_benchmark.py

    # Plot from a custom directory
    python plot_benchmark.py --input-dir results/my_experiment

    # Plot a single system file
    python plot_benchmark.py --files results/Pendulum.npz

    # Save figures to disk instead of showing interactively
    python plot_benchmark.py --save-dir figures
"""

import argparse
import os

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from simulation import plot_results
from run_benchmark import load_data


def main():
    parser = argparse.ArgumentParser(description="Plot LQR benchmark results")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results",
        help="Directory containing .npz result files. Default: results.",
    )
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="Specific .npz files to plot. Overrides --input-dir.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, save figures to this directory instead of showing them.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        default=True,
        help="Use log scale on y-axis. Default: True.",
    )
    parser.add_argument(
        "--no-log-y",
        action="store_false",
        dest="log_y",
        help="Use linear scale on y-axis.",
    )
    args = parser.parse_args()

    # Collect .npz files
    if args.files:
        npz_files = args.files
    else:
        npz_files = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith(".npz")
        )

    if not npz_files:
        print(f"No .npz files found in {args.input_dir}")
        return

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for path in npz_files:
        system_name, results = load_data(path)
        print(f"Plotting: {system_name} ({path})")

        save_path = None
        if args.save_dir:
            safe_name = system_name.replace(" ", "_").replace("/", "_")
            save_path = os.path.join(args.save_dir, f"regret_{safe_name}.pdf")

        plot_results(
            results,
            title=system_name,
            log_y=args.log_y,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
