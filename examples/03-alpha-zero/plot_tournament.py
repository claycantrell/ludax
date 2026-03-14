import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import pickle
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tournament Elo ratings")
    parser.add_argument("--game", type=str, default="reversi", help="Game name (used for input/output filenames)")
    parser.add_argument("--games_per_batch", type=int, default=4096, help="Frames per iteration step")
    parser.add_argument("--smooth_window", type=int, default=5, help="Rolling window size for mean/var plot")
    parser.add_argument("--input_path", type=str, default=None, help="Path to input pkl (default: examples/03-alpha-zero/data/elo_{game}.pkl)")
    parser.add_argument("--output_dir", type=str, default="examples/03-alpha-zero/data", help="Directory to save output figures")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load and parse tournament data
# ---------------------------------------------------------------------------

def grouped_elo(data):
    """
    Parse tournament pickle into two DataFrames (ldx and pgx), one column per run.
    Also extracts the baseline Elo if it exists in the checkpoint list.
    """
    elos = data["elos"]
    checkpoints = data["checkpoints"]

    ldx_records = {}
    pgx_records = {}
    baseline_elo = None

    for ckpt in checkpoints:
        label = ckpt["label"]
        run_name = ckpt["run_name"]
        iteration = ckpt["iteration"]
        elo = elos[label]

        # Extract baseline Elo separately and skip standard run parsing
        if run_name == "pgx_baseline":
            baseline_elo = elo
            continue

        # Parse env_type from run_name: "{game}_{env_type}_{timestamp}"
        m = re.search(r"_(ldx|pgx)_(\d+)$", run_name)
        if m is None:
            print(f"Warning: could not parse env_type from run_name '{run_name}', skipping.")
            continue

        env_type = m.group(1)   # "ldx" or "pgx"
        timestamp = m.group(2)  # unique run identifier

        bucket = ldx_records if env_type == "ldx" else pgx_records
        if timestamp not in bucket:
            bucket[timestamp] = {}
        bucket[timestamp][iteration] = elo

    def records_to_df(records):
        """Convert {timestamp: {iteration: elo}} to a DataFrame indexed by iteration."""
        if not records:
            return pd.DataFrame()
        series = {ts: pd.Series(iters) for ts, iters in records.items()}
        df = pd.DataFrame(series)
        df.index.name = "iteration"
        df.sort_index(inplace=True)
        return df

    return records_to_df(ldx_records), records_to_df(pgx_records), baseline_elo


# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.loc": "best",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

colors = ['#3599BF', '#CE6661', '#64B268', '#3298BE', '#E68656']


def setup_neurips_plot(width=3.5, height=None):
    if height is None:
        height = width / 1.618
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    return fig, ax


if __name__ == "__main__":
    args = parse_args()

    input_path = args.input_path or f"{args.output_dir}/elo_{args.game}.pkl"
    data = pickle.load(open(input_path, "rb"))
    ldx_elo_df, pgx_elo_df, baseline_elo = grouped_elo(data)

    frame_multiple = args.games_per_batch  # frames per iteration step

    # ---------------------------------------------------------------------------
    # Plot 1: Individual raw runs
    # ---------------------------------------------------------------------------

    fig, ax = setup_neurips_plot()

    for df, label, color in zip([ldx_elo_df, pgx_elo_df], ["Ludax", "PGX"], colors[:2]):
        for run in df.columns:
            ax.plot(
                df.index * frame_multiple / 1e6,
                df[run],
                alpha=1, color=color, linestyle="-", linewidth=0.6,
            )

    raw_legend_lines = [
        Line2D([0], [0], color=colors[0], linewidth=1, linestyle="-", alpha=1, label="Ludax runs"),
        Line2D([0], [0], color=colors[1], linewidth=1, linestyle="-", alpha=1, label="PGX runs"),
    ]

    if baseline_elo is not None:
        ax.axhline(y=baseline_elo, color='gray', linestyle='--', linewidth=1.2, zorder=0)
        raw_legend_lines.append(Line2D([0], [0], color='gray', linewidth=1.2, linestyle='--', label="Pgx Baseline"))

    handles, _ = ax.get_legend_handles_labels()
    handles.extend(raw_legend_lines)

    ax.set_xlabel("Millions of frames")
    ax.set_ylabel("Elo")
    ax.set_title(f"$\\bf{{{args.game.capitalize()}}}:$ Individual Runs")
    ax.legend(handles=handles, loc="lower right", framealpha=0.9)
    fig.savefig(f"{args.output_dir}/{args.game}_elo_raw_runs_frames.pdf", bbox_inches="tight")
    fig.savefig(f"{args.output_dir}/{args.game}_elo_raw_runs_frames.png", bbox_inches="tight", dpi=300)


    # ---------------------------------------------------------------------------
    # Plot 2: Mean ± 1σ (smoothed)
    # ---------------------------------------------------------------------------

    fig, ax = setup_neurips_plot()

    for elo_df, label, color in zip(
        [ldx_elo_df, pgx_elo_df],
        ["Ludax", "PGX"],
        colors[:2],
    ):
        if elo_df.empty:
            continue

        mean_raw    = elo_df.mean(axis=1)
        var_raw     = elo_df.var(axis=1)
        mean_smooth = mean_raw.rolling(window=args.smooth_window, min_periods=1, center=True).mean()
        std_smooth  = np.sqrt(var_raw.rolling(window=args.smooth_window, min_periods=1, center=True).mean())

        frames = elo_df.index * frame_multiple / 1e6

        ax.plot(frames, mean_smooth, color=color, label=f"{label} mean (smoothed)")
        ax.fill_between(
            frames,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=color, alpha=0.07,
            label="  ±1σ",
        )

    if baseline_elo is not None:
        ax.axhline(y=baseline_elo, color='gray', linestyle='--', linewidth=1.2, label="Pgx Baseline", zorder=0)

    ax.set_xlabel("Millions of frames")
    ax.set_ylabel("         ")
    ax.set_title("Mean and Variance")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(f"{args.output_dir}/{args.game}_elo_mean_frames.pdf", bbox_inches="tight")
    fig.savefig(f"{args.output_dir}/{args.game}_elo_mean_frames.png", bbox_inches="tight", dpi=300)
