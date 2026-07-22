import os
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\concordance_woCtrl_2cal.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\Concordance"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

sample_col = "SampleID"
group_col = "Group"
target_col = "Assay"
npx_col = "NPX"
pg_col = "pg/mL"

df[npx_col] = pd.to_numeric(df[npx_col], errors="coerce")
df[pg_col] = pd.to_numeric(df[pg_col], errors="coerce")

df = df.dropna(subset=[sample_col, group_col, target_col])

print("Columns found:")
print(df.columns.tolist())

print("\nGroups:")
print(df[group_col].value_counts())

# =============================================================================
# Plot style
# =============================================================================

box_style = dict(
    boxprops=dict(edgecolor="black", linewidth=1.2),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2)
)

def format_spines(ax):
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

# =============================================================================
# Helpers
# =============================================================================

def clean_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))

group_order = df[group_col].dropna().unique().tolist()

palette = {
    group_order[0]: "#1F77B4" if len(group_order) > 0 else "#1F77B4",
    group_order[1]: "#FF7F0E" if len(group_order) > 1 else "#FF7F0E",
    group_order[2]: "#2CA02C" if len(group_order) > 2 else "#2CA02C",
}

extra_colors = sns.color_palette("tab10", n_colors=max(len(group_order), 3))

for i, g in enumerate(group_order):
    if g not in palette:
        palette[g] = extra_colors[i]

sample_order_df = (
    df[[group_col, sample_col]]
    .drop_duplicates()
    .sort_values([group_col, sample_col])
)

sample_order = sample_order_df[sample_col].tolist()
sample_to_group = dict(zip(sample_order_df[sample_col], sample_order_df[group_col]))

def add_group_bar(ax, sample_order, sample_to_group, palette):
    transform = ax.get_xaxis_transform()

    for i, sample in enumerate(sample_order):
        group = sample_to_group[sample]

        rect = Rectangle(
            (i - 0.45, -0.20),
            0.90,
            0.045,
            transform=transform,
            color=palette[group],
            clip_on=False
        )

        ax.add_patch(rect)

    group_positions = {}

    for i, sample in enumerate(sample_order):
        group = sample_to_group[sample]
        group_positions.setdefault(group, []).append(i)

    for group, positions in group_positions.items():
        center = np.mean(positions)

        ax.text(
            center,
            -0.28,
            group,
            ha="center",
            va="top",
            transform=transform,
            fontsize=9,
            color=palette[group],
            fontweight="bold",
            clip_on=False
        )

# =============================================================================
# Figure 1: Sample-level boxplots grouped by category
# =============================================================================

def plot_sample_boxplot(value_col, y_label, file_suffix):
    plot_df = df.dropna(subset=[value_col]).copy()

    fig_file = os.path.join(
        save_dir,
        f"{file_stem}_{file_suffix}_by_SampleID_grouped_boxplot.png"
    )

    fig, ax = plt.subplots(figsize=(18, 7))

    sns.boxplot(
        data=plot_df,
        x=sample_col,
        y=value_col,
        hue=group_col,
        order=sample_order,
        hue_order=group_order,
        palette=palette,
        width=0.65,
        showfliers=False,
        **box_style,
        ax=ax
    )

    sns.stripplot(
        data=plot_df,
        x=sample_col,
        y=value_col,
        order=sample_order,
        color="#7B1FA2",
        size=3,
        alpha=0.75,
        jitter=0.18,
        ax=ax
    )

    add_group_bar(ax, sample_order, sample_to_group, palette)
    format_spines(ax)

    ax.set_xlabel("SampleID")
    ax.set_ylabel(y_label)
    ax.set_title(f"{file_stem} - {y_label} by SampleID grouped by category")
    ax.tick_params(axis="x", labelrotation=45)

    ax.legend(
        title="Group",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.subplots_adjust(bottom=0.30, right=0.82)
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFigure saved to:\n{fig_file}")

plot_sample_boxplot(
    value_col=npx_col,
    y_label="NPX",
    file_suffix="NPX"
)

plot_sample_boxplot(
    value_col=pg_col,
    y_label="pg/mL",
    file_suffix="pg_mL"
)

# =============================================================================
# Calculate CV% per target within each group
# =============================================================================

def calculate_cv(data, value_col, mean_col_name):
    cv_df = (
        data
        .dropna(subset=[value_col])
        .groupby([group_col, target_col], as_index=False)
        .agg(
            N=(value_col, "count"),
            Mean_Value=(value_col, "mean"),
            SD_Value=(value_col, "std")
        )
    )

    cv_df["CV%"] = (cv_df["SD_Value"] / cv_df["Mean_Value"]) * 100
    cv_df = cv_df.replace([np.inf, -np.inf], np.nan)
    cv_df = cv_df.dropna(subset=["CV%"])

    cv_df = cv_df[cv_df["N"] >= 3].copy()
    cv_df = cv_df.rename(columns={"Mean_Value": mean_col_name})

    return cv_df

cv_npx = calculate_cv(df, npx_col, "Mean_NPX")
cv_pg = calculate_cv(df, pg_col, "Mean_pg_mL")

# =============================================================================
# Bin CV tables
# =============================================================================

npx_edges = [0, 1, 2, 4, 8, np.inf]
npx_labels = ["0–1", "1–2", "2–4", "4–8", ">8"]

cv_npx["NPX Mean Range"] = pd.cut(
    cv_npx["Mean_NPX"],
    bins=npx_edges,
    labels=npx_labels,
    include_lowest=True,
    right=True
)

pg_conditions = [
    (cv_pg["Mean_pg_mL"] >= 0) & (cv_pg["Mean_pg_mL"] <= 10),
    (cv_pg["Mean_pg_mL"] > 10) & (cv_pg["Mean_pg_mL"] <= 200),
    (cv_pg["Mean_pg_mL"] > 200) & (cv_pg["Mean_pg_mL"] <= 500),
    (cv_pg["Mean_pg_mL"] > 500) & (cv_pg["Mean_pg_mL"] <= 1000),
    (cv_pg["Mean_pg_mL"] > 1000)
]

pg_labels = ["0–10", "10–200", "200–500", "500–1000", ">1000"]

cv_pg["pg/mL Mean Range"] = np.select(
    pg_conditions,
    pg_labels,
    default=None
)

cv_npx = cv_npx.dropna(subset=["NPX Mean Range"]).copy()
cv_pg = cv_pg.dropna(subset=["pg/mL Mean Range"]).copy()

cv_npx["NPX Mean Range"] = pd.Categorical(
    cv_npx["NPX Mean Range"],
    categories=npx_labels,
    ordered=True
)

cv_pg["pg/mL Mean Range"] = pd.Categorical(
    cv_pg["pg/mL Mean Range"],
    categories=pg_labels,
    ordered=True
)

# =============================================================================
# Save CV tables
# =============================================================================

cv_npx_file = os.path.join(
    save_dir,
    f"{file_stem}_target_CV_NPX_by_group_and_mean_range.csv"
)

cv_pg_file = os.path.join(
    save_dir,
    f"{file_stem}_target_CV_pg_mL_by_group_and_mean_range.csv"
)

cv_npx.to_csv(cv_npx_file, index=False)
cv_pg.to_csv(cv_pg_file, index=False)

print(f"\nNPX CV table saved to:\n{cv_npx_file}")
print(f"\npg/mL CV table saved to:\n{cv_pg_file}")

# =============================================================================
# Figure 2A: CV% by mean range, separately for each group
# =============================================================================

def plot_cv_by_range_per_group(data, range_col, range_order, value_label, save_prefix):
    for group in group_order:

        group_df = data[data[group_col] == group].copy()

        if group_df.empty:
            print(f"Skipping {group}: no CV data")
            continue

        fig_file = os.path.join(
            save_dir,
            f"{file_stem}_{save_prefix}_CV_by_mean_range_{clean_filename(group)}.png"
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        sns.boxplot(
            data=group_df,
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
            data=group_df,
            x=range_col,
            y="CV%",
            order=range_order,
            color="#7B1FA2",
            size=4,
            alpha=0.75,
            jitter=0.18,
            ax=ax
        )

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
            linewidth=1.5,
            label="20% CV"
        )

        ax.axhline(
            y=35,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="35% CV"
        )

        format_spines(ax)

        ax.legend(frameon=False)

        ax.set_xlabel(f"{value_label} mean range")
        ax.set_ylabel("CV%")
        ax.set_title(f"{file_stem} - {value_label} target CV%: {group}")

        plt.tight_layout()
        plt.savefig(fig_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"\nCV figure saved to:\n{fig_file}")

# =============================================================================
# Figure 2B: merged CV% figure, all groups together
# =============================================================================

def plot_cv_by_range_all_groups(data, range_col, range_order, value_label, save_prefix):

    fig_file = os.path.join(
        save_dir,
        f"{file_stem}_{save_prefix}_CV_by_mean_range_ALL_GROUPS.png"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=data,
        x=range_col,
        y="CV%",
        hue=group_col,
        order=range_order,
        hue_order=group_order,
        palette=palette,
        width=0.65,
        showfliers=False,
        **box_style,
        ax=ax
    )

    sns.stripplot(
        data=data,
        x=range_col,
        y="CV%",
        hue=group_col,
        order=range_order,
        hue_order=group_order,
        dodge=True,
        color="#7B1FA2",
        size=4,
        alpha=0.75,
        jitter=0.18,
        ax=ax
    )

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

    format_spines(ax)

    legend_elements = [
        Patch(facecolor=palette[g], edgecolor="black", label=g)
        for g in group_order
    ]

    legend_elements.extend([
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
    ])

    ax.legend(
        handles=legend_elements,
        title="Group",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.set_xlabel(f"{value_label} mean range")
    ax.set_ylabel("CV%")
    ax.set_title(f"{file_stem} - {value_label} target CV% by Group")

    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nMerged CV figure saved to:\n{fig_file}")

# =============================================================================
# Generate CV figures
# =============================================================================

plot_cv_by_range_per_group(
    data=cv_npx,
    range_col="NPX Mean Range",
    range_order=npx_labels,
    value_label="NPX",
    save_prefix="NPX"
)

plot_cv_by_range_per_group(
    data=cv_pg,
    range_col="pg/mL Mean Range",
    range_order=pg_labels,
    value_label="pg_mL",
    save_prefix="pg_mL"
)

plot_cv_by_range_all_groups(
    data=cv_npx,
    range_col="NPX Mean Range",
    range_order=npx_labels,
    value_label="NPX",
    save_prefix="NPX"
)

plot_cv_by_range_all_groups(
    data=cv_pg,
    range_col="pg/mL Mean Range",
    range_order=pg_labels,
    value_label="pg_mL",
    save_prefix="pg_mL"
)

print("\nAnalysis complete.")
