import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\Run_1-3_woCtrl.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\Run_CV"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

target_col = "Assay"

npx_run_cols = ["NPX_Run1", "NPX_Run2", "NPX_Run3"]
pg_run_cols = ["pg/mL_Run1", "pg/mL_Run2", "pg/mL_Run3"]

for col in npx_run_cols + pg_run_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =============================================================================
# Calculate CV% per target within each run
# =============================================================================

def calculate_run_cv(data, run_cols, value_type):
    all_cv = []

    for run_col in run_cols:
        run_name = run_col.split("_")[-1]

        temp = (
            data
            .dropna(subset=[run_col])
            .groupby(target_col, as_index=False)
            .agg(
                N=(run_col, "count"),
                Mean_Value=(run_col, "mean"),
                SD_Value=(run_col, "std")
            )
        )

        temp["CV%"] = (temp["SD_Value"] / temp["Mean_Value"]) * 100
        temp["Run"] = run_name
        temp["Value_Type"] = value_type

        temp = temp.replace([np.inf, -np.inf], np.nan)
        temp = temp.dropna(subset=["CV%"])

        all_cv.append(temp)

    return pd.concat(all_cv, ignore_index=True)

cv_npx = calculate_run_cv(df, npx_run_cols, "NPX")
cv_pg = calculate_run_cv(df, pg_run_cols, "pg/mL")

# =============================================================================
# Binning
# =============================================================================

npx_edges = [0, 1, 2, 4, 8, np.inf]
npx_labels = ["0–1", "1–2", "2–4", "4–8", ">8"]

cv_npx["NPX Mean Range"] = pd.cut(
    cv_npx["Mean_Value"],
    bins=npx_edges,
    labels=npx_labels,
    include_lowest=True,
    right=True
)

cv_npx = cv_npx.dropna(subset=["NPX Mean Range"]).copy()

cv_npx["NPX Mean Range"] = pd.Categorical(
    cv_npx["NPX Mean Range"],
    categories=npx_labels,
    ordered=True
)

pg_conditions = [
    (cv_pg["Mean_Value"] >= 0) & (cv_pg["Mean_Value"] <= 10),
    (cv_pg["Mean_Value"] > 10) & (cv_pg["Mean_Value"] <= 200),
    (cv_pg["Mean_Value"] > 200) & (cv_pg["Mean_Value"] <= 500),
    (cv_pg["Mean_Value"] > 500) & (cv_pg["Mean_Value"] <= 1000),
    (cv_pg["Mean_Value"] > 1000)
]

pg_labels = ["0–10", "10–200", "200–500", "500–1000", ">1000"]

cv_pg["pg/mL Mean Range"] = np.select(
    pg_conditions,
    pg_labels,
    default=None
)

cv_pg = cv_pg.dropna(subset=["pg/mL Mean Range"]).copy()

cv_pg["pg/mL Mean Range"] = pd.Categorical(
    cv_pg["pg/mL Mean Range"],
    categories=pg_labels,
    ordered=True
)

# =============================================================================
# Save CV tables
# =============================================================================

cv_npx.to_csv(
    os.path.join(save_dir, f"{file_stem}_NPX_CV_per_target_per_run.csv"),
    index=False
)

cv_pg.to_csv(
    os.path.join(save_dir, f"{file_stem}_pg_mL_CV_per_target_per_run.csv"),
    index=False
)

# =============================================================================
# Plot helpers
# =============================================================================

box_style = dict(
    boxprops=dict(edgecolor="black", linewidth=1.2),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2)
)

palette_runs = {
    "Run1": "#EB11A6FF",
    "Run2": "#E8A811F9",
    "Run3": "#AC0FF078"
}

run_order = ["Run1", "Run2", "Run3"]

def format_axis(ax):
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.5,
        alpha=0.5
    )

    ax.axhline(
        y=20,
        color="red",
        linestyle="--",
        linewidth=1.5
    )

    ax.axhline(
        y=35,
        color="red",
        linestyle="--",
        linewidth=1.5
    )

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

def get_boxplot_data(data, group_cols):
    """
    Return data only for bins/groups with >=3 CV% points.
    Dots are always plotted from the full data.
    """
    temp = data.copy()

    temp["Plot_N"] = (
        temp
        .groupby(group_cols, observed=False)["CV%"]
        .transform("count")
    )

    return temp[temp["Plot_N"] >= 3].copy()

def add_run_legend(ax):
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette_runs["Run1"],
            markeredgecolor="black",
            markersize=8,
            label="Run1"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette_runs["Run2"],
            markeredgecolor="black",
            markersize=8,
            label="Run2"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=palette_runs["Run3"],
            markeredgecolor="black",
            markersize=8,
            label="Run3"
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="20% CV"
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="35% CV"
        )
    ]

    ax.legend(
        handles=legend_elements,
        title="Run",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

# =============================================================================
# Combined plot: all runs together
# Boxplots only for Run x Range groups with >=3 CV% values
# Purple/run-colored dots include all CV% values
# =============================================================================

def plot_cv_by_run_and_range(data, range_col, range_order, x_label, value_type, save_prefix):
    fig_file = os.path.join(
        save_dir,
        f"{file_stem}_{save_prefix}_CV_by_run_and_mean_range.png"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    box_df = get_boxplot_data(data, [range_col, "Run"])

    if not box_df.empty:
        sns.boxplot(
            data=box_df,
            x=range_col,
            y="CV%",
            hue="Run",
            hue_order=run_order,
            order=range_order,
            color="white",
            width=0.65,
            showfliers=False,
            **box_style,
            ax=ax
        )

    sns.stripplot(
        data=data,
        x=range_col,
        y="CV%",
        hue="Run",
        hue_order=run_order,
        order=range_order,
        dodge=True,
        palette=palette_runs,
        size=4,
        alpha=0.85,
        jitter=0.18,
        ax=ax
    )

    format_axis(ax)
    add_run_legend(ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel("CV%")
    ax.set_title(f"{file_stem} - {value_type} CV% per Target by Run")

    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Combined figure saved to:\n{fig_file}")

# =============================================================================
# Separate plot: one figure per run
# Boxplots only for bins with >=3 CV% values
# Dots include all CV% values
# =============================================================================

def plot_cv_by_range_for_each_run(data, range_col, range_order, x_label, value_type, save_prefix):
    for run in run_order:

        run_df = data[data["Run"] == run].copy()

        if run_df.empty:
            print(f"Skipping {run}: no data")
            continue

        fig_file = os.path.join(
            save_dir,
            f"{file_stem}_{save_prefix}_CV_by_mean_range_{run}.png"
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        box_df = get_boxplot_data(run_df, [range_col])

        if not box_df.empty:
            sns.boxplot(
                data=box_df,
                x=range_col,
                y="CV%",
                order=range_order,
                color="white",
                width=0.45,
                showfliers=False,
                **box_style,
                ax=ax
            )

        sns.stripplot(
            data=run_df,
            x=range_col,
            y="CV%",
            order=range_order,
            color=palette_runs[run],
            size=4,
            alpha=0.85,
            jitter=0.18,
            ax=ax
        )

        format_axis(ax)

        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=palette_runs[run],
                    markeredgecolor="black",
                    markersize=8,
                    label=run
                ),
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label="20% CV"
                ),
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    label="35% CV"
                )
            ],
            title="Run",
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel("CV%")
        ax.set_title(f"{file_stem} - {value_type} CV% per Target - {run}")

        plt.tight_layout()
        plt.savefig(fig_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Separate run figure saved to:\n{fig_file}")

# =============================================================================
# Generate figures
# =============================================================================

plot_cv_by_run_and_range(
    data=cv_npx,
    range_col="NPX Mean Range",
    range_order=npx_labels,
    x_label="NPX mean range",
    value_type="NPX",
    save_prefix="NPX"
)

plot_cv_by_run_and_range(
    data=cv_pg,
    range_col="pg/mL Mean Range",
    range_order=pg_labels,
    x_label="pg/mL mean range",
    value_type="pg/mL",
    save_prefix="pg_mL"
)

plot_cv_by_range_for_each_run(
    data=cv_npx,
    range_col="NPX Mean Range",
    range_order=npx_labels,
    x_label="NPX mean range",
    value_type="NPX",
    save_prefix="NPX"
)

plot_cv_by_range_for_each_run(
    data=cv_pg,
    range_col="pg/mL Mean Range",
    range_order=pg_labels,
    x_label="pg/mL mean range",
    value_type="pg/mL",
    save_prefix="pg_mL"
)

print("\nAnalysis complete.")

