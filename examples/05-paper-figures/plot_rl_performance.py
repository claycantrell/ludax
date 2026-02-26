import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import lineStyles
from matplotlib.lines import Line2D
from scipy.stats import alpha
import pickle

GAMES_PER_BATCH = 4096
GAME_NAME = "reversi"  # "hex"
SMOOTH_WINDOW = 5 # frames to smooth over for mean/var plot

data = pickle.load(open(f"./data/rl_runs/elo_{GAME_NAME}.pkl", "rb"))


ldx_elo_df, pgx_elo_df = grouped_elo(data)


# General configuration
mpl.rcParams.update({
    "text.usetex": False,                   # Set True if you're compiling with LaTeX
    "font.family": "serif",                # 'serif' or 'sans-serif' as preferred
    "font.size": 8,                        # NeurIPS papers often use 8 pt font
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
    "pdf.fonttype": 42,                    # Embed fonts in PDF (important for submission)
    "ps.fonttype": 42
})

width = 3.5  # inches for single column, 7 for double
height = width / 1.618


colors = ['#3599BF', '#CE6661', '#64B268', '#3298BE', '#E68656']  # Colorblind-safe
markers = ['o', 's', '^', 'D', 'x']  # Use distinct shapes

def setup_neurips_plot(width=3.5, height=None):
    """
    Create a Matplotlib figure and axis with NeurIPS-friendly styling.

    Parameters:
        width (float): Width of the figure in inches (3.5 for single-column).
        height (float): Height in inches. Defaults to golden ratio if None.

    Returns:
        fig, ax: Matplotlib figure and axis.
    """
    # Golden ratio for aesthetics
    if height is None:
        height = width / 1.618

    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)

    return fig, ax



# build figure
fig, ax = setup_neurips_plot()

for df, label, color in zip([ldx_elo_df, pgx_elo_df], ["Ludax", "PGX"], colors[:2]):

    # plot raw runs
    for run in df.columns:
        ax.plot(df.index * FRAME_MULTIPLE / 10**6, df[run], alpha=1, color=color, linestyle="-", linewidth=0.6)

raw_legend_lines = [
    Line2D([0], [0], color=colors[0], linewidth=1, linestyle="-", alpha=1, label="Ludax runs"),
    Line2D([0], [0], color=colors[1], linewidth=1, linestyle="-", alpha=1, label="PGX runs"),
    Line2D([0], [0], color="gray", linewidth=1, linestyle="--", alpha=1, label="(PGX) Baseline"),
]
handles, labels = ax.get_legend_handles_labels()
handles.extend(raw_legend_lines)

# Set the legend

# PGX baseline
ax.axhline(0.5, linestyle="--", color="gray", alpha=1, linewidth=0.8)

# labels & legend
ax.set_xlabel("Millions of frames")
ax.set_ylabel("Elo")
ax.set_title("$\\bf{Reversi}:$ Individual Runs")
ax.legend(handles=handles, loc="lower right", framealpha=0.9)
fig.savefig("../data/rl_runs/reversi/reversi_elo_raw_runs_frames.pdf", bbox_inches="tight")



# ——————————————————————————————————————————————————————
# Now plot Elo **vs** Training Time (hours)
fig, ax = setup_neurips_plot()

for elo_df, label, color in zip(
    [ldx_elo_df, pgx_elo_df],
    ["Ludax", "PGX"],
    colors[:2]
):
    # a) compute per-step mean & var of elo
    mean_raw   = elo_df.mean(axis=1)
    var_raw    = elo_df.var(axis=1)
    mean_smooth= mean_raw.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()
    std_smooth = np.sqrt(var_raw.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean())

    frames = elo_df.index * FRAME_MULTIPLE / 10**6

    # c) plot the smoothed mean vs time
    ax.plot(frames, mean_smooth, color=color, label=f"{label} mean (smoothed)")
    
    # d) shade ±1σ around it
    lower = mean_smooth - std_smooth
    upper = mean_smooth + std_smooth
    ax.fill_between(frames, lower, upper,
                    color=color, alpha=0.07,
                    label=f"  ±1σ")

# PGX baseline
ax.axhline(0.5, linestyle="--", color="gray", alpha=1, linewidth=0.8) #

# y limit
ax.set_ylim(0, 0.7)

# labels & legend
ax.set_xlabel("Millions of frames")
ax.set_ylabel("         ")
ax.set_title("Mean and Variance")
ax.legend(loc="lower right", framealpha=0.9)

fig.savefig("/ludax/examples/05-paper-figures/reversi_elo_mean_frames.pdf", bbox_inches="tight")