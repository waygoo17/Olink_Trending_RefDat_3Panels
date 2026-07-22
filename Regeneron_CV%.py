import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\intra_CV%_REG.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\Group_CV_Bins"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

# =============================================================================
# Columns
# =============================================================================

sample_col = "SampleID"
group_col = "Group"
target_col = "Assay"
npx_col = "NPX"
pg_col = "pg/mL"

# =============================================================================
# Bin definitions from your scripts
# =============================================================================

NPX_BIN_EDGES = [0, 1, 2, 4, 8, np.inf]
NPX_BIN_LABELS = ["0–1", "1–2", "2–4", "4–8", ">8"]

PG_BIN_EDGES = [0, 10, 200, 500, 1000, np.inf]
PG_BIN_LABELS = ["0–10", "10–200", "200–500", "500–1000", ">1000"]

# =============================================================================
# Style
# =============================================================================

sns.set_style("whitegrid")

Y_MAX = 100
Y_TICK_STEP = 10

box_style = dict(
    boxprops=dict(edgecolor="black", linewidth=1.1),
    medianprops=dict(color="black", linewidth=1.4),
    whiskerprops=dict(color="black", linewidth=1.1),
    capprops=dict(color="black", linewidth=1.1),
)

# =============================================================================
# Robust CSV reader (handles cp1252 / utf-8 / latin1 exports)
# =============================================================================

def read_csv_robust(path: str) -> pd.DataFrame:
    encodings = ["cp1252", "utf-8-sig", "utf-8", "latin1"]
    last_err = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded CSV using encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_err = e

    raise last_err if last_err is not None else ValueError("Unable to read CSV.")

# =============================================================================
# Read data
# =============================================================================

df = read_csv_robust(data_file)
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())

required_cols = [sample_col, group_col, target_col, npx_col, pg_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean text columns: remove non-breaking spaces and normalize whitespace
for col in [sample_col, group_col, target_col]:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

df[npx_col] = pd.to_numeric(df[npx_col], errors="coerce")
df[pg_col] = pd.to_numeric(df[pg_col], errors="coerce")
df = df.dropna(subset=[sample_col, group_col, target_col])

print("\nGroups found:")
print(df[group_col].value_counts())

# =============================================================================
# Replicate grouping
# Assumes technical replicates look like:
#   3608805833_2
#   3608805833_3
#   3608805833_4
# and should be grouped as BioSampleID = 3608805833
# =============================================================================

def make_biosample_id(sample_id: object) -> str:
    s = str(sample_id)
    m = re.match(r"^(.*?)(?:_[0-9]+)$", s)
    return m.group(1) if m else s

df["BioSampleID"] = df[sample_col].map(make_biosample_id)

# =============================================================================
# Group order + palette
# =============================================================================

present_groups = df[group_col].dropna().unique().tolist()
group_order = present_groups[:]  # preserve appearance order

# purple family palette
palette_list = sns.color_palette("Purples", n_colors=max(len(group_order) + 2, 4))
palette_list = palette_list[1:1 + len(group_order)]
group_palette = dict(zip(group_order, palette_list))

# =============================================================================
# CV calculation
# CV only when N >= 3 replicates per BioSampleID x Group x Assay
# =============================================================================

def calculate_cv_table(data: pd.DataFrame, value_col: str) -> pd.DataFrame:
    cv_df = (
        data.dropna(subset=[value_col])
        .groupby(["BioSampleID", group_col, target_col], as_index=False)
        .agg(
            N=(value_col, "count"),
            Mean_Value=(value_col, "mean"),
            SD_Value=(value_col, "std"),
        )
    )

    cv_df = cv_df[cv_df["N"] >= 3].copy()
    cv_df["CV%"] = (cv_df["SD_Value"] / cv_df["Mean_Value"]) * 100
    cv_df = cv_df.replace([np.inf, -np.inf], np.nan)
    cv_df = cv_df.dropna(subset=["CV%"])

    return cv_df

cv_npx = calculate_cv_table(df, npx_col)
cv_pg = calculate_cv_table(df, pg_col)

# =============================================================================
# Binning
# =============================================================================

def add_npx_bins(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["NPX Mean Range"] = pd.cut(
        out["Mean_Value"],
        bins=NPX_BIN_EDGES,
        labels=NPX_BIN_LABELS,
        include_lowest=True,
        right=True,
    )
    out = out.dropna(subset=["NPX Mean Range"]).copy()
    out["NPX Mean Range"] = pd.Categorical(
        out["NPX Mean Range"],
        categories=NPX_BIN_LABELS,
        ordered=True,
    )
    return out

def add_pg_bins(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    conditions = [
        (out["Mean_Value"] >= 0) & (out["Mean_Value"] <= 10),
        (out["Mean_Value"] > 10) & (out["Mean_Value"] <= 200),
        (out["Mean_Value"] > 200) & (out["Mean_Value"] <= 500),
        (out["Mean_Value"] > 500) & (out["Mean_Value"] <= 1000),
        (out["Mean_Value"] > 1000),
    ]
    out["pg/mL Mean Range"] = np.select(conditions, PG_BIN_LABELS, default=None)
    out = out.dropna(subset=["pg/mL Mean Range"]).copy()
    out["pg/mL Mean Range"] = pd.Categorical(
        out["pg/mL Mean Range"],
        categories=PG_BIN_LABELS,
        ordered=True,
    )
    return out

cv_npx = add_npx_bins(cv_npx)
cv_pg = add_pg_bins(cv_pg)

# =============================================================================
# Audit / summary tables
# =============================================================================

replicate_audit = (
    df.groupby(["BioSampleID", group_col, target_col], as_index=False)
    .size()
    .rename(columns={"size": "Replicate_N"})
)

replicate_audit_csv = os.path.join(save_dir, f"{file_stem}_replicate_audit.csv")
replicate_audit.to_csv(replicate_audit_csv, index=False)

npx_cv_csv = os.path.join(save_dir, f"{file_stem}_group_CV_NPX_by_target.csv")
pg_cv_csv = os.path.join(save_dir, f"{file_stem}_group_CV_pg_mL_by_target.csv")

cv_npx.to_csv(npx_cv_csv, index=False)
cv_pg.to_csv(pg_cv_csv, index=False)

def summary_by_bin(data: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    return (
        data.groupby([group_col, bin_col], observed=False)["CV%"]
        .agg(
            N="count",
            Mean_CV_percent="mean",
            Median_CV_percent="median",
            SD_CV_percent="std",
            Min_CV_percent="min",
            Max_CV_percent="max",
        )
        .reset_index()
        .sort_values([group_col, bin_col])
    )

npx_summary = summary_by_bin(cv_npx, "NPX Mean Range")
pg_summary = summary_by_bin(cv_pg, "pg/mL Mean Range")

npx_summary_csv = os.path.join(save_dir, f"{file_stem}_group_CV_NPX_summary.csv")
pg_summary_csv = os.path.join(save_dir, f"{file_stem}_group_CV_pg_mL_summary.csv")

npx_summary.to_csv(npx_summary_csv, index=False)
pg_summary.to_csv(pg_summary_csv, index=False)

print(f"\nReplicate audit saved to:\n{replicate_audit_csv}")
print(f"NPX CV table saved to:\n{npx_cv_csv}")
print(f"pg/mL CV table saved to:\n{pg_cv_csv}")
print(f"NPX summary saved to:\n{npx_summary_csv}")
print(f"pg/mL summary saved to:\n{pg_summary_csv}")

print("\nReplicate count distribution:")
print(replicate_audit["Replicate_N"].value_counts().sort_index())

# =============================================================================
# Plot helpers
# =============================================================================

def format_axis(ax, ylabel="CV%"):
    ax.set_ylim(0, Y_MAX)
    ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_STEP))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axhline(y=20, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(y=30, color="red", linestyle="--", linewidth=1.5)
    ax.set_ylabel(ylabel)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.1)

def boxplot_subset(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Keep only bin/group combinations with >=3 CV points for boxplots.
    Dots are always plotted from the full data.
    """
    counts = data.groupby(group_cols, observed=False)["CV%"].transform("count")
    return data[counts >= 3].copy()

def cutoff_legend():
    return [
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="20% cutoff"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="30% cutoff"),
    ]

def group_legend_handles():
    return [
        Patch(facecolor=group_palette[g], edgecolor="black", label=g)
        for g in group_order
    ]

def clean_filename(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))

# =============================================================================
# Plotting
# =============================================================================

def plot_all_groups(data: pd.DataFrame, bin_col: str, bin_order: list[str], value_label: str, save_path: str):
    fig, ax = plt.subplots(figsize=(11, 6))

    box_df = boxplot_subset(data, [bin_col, group_col])

    if not box_df.empty:
        sns.boxplot(
            data=box_df,
            x=bin_col,
            y="CV%",
            hue=group_col,
            hue_order=group_order,
            order=bin_order,
            palette=group_palette,
            width=0.7,
            showfliers=False,
            ax=ax,
            **box_style,
        )

    sns.stripplot(
        data=data,
        x=bin_col,
        y="CV%",
        hue=group_col,
        hue_order=group_order,
        order=bin_order,
        dodge=True,
        palette=group_palette,
        size=4.5,
        alpha=0.85,
        jitter=0.18,
        ax=ax,
    )

    handles = group_legend_handles() + cutoff_legend()
    ax.legend(
        handles=handles,
        title="Group",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.set_xlabel(value_label)
    ax.set_title(f"{file_stem} - {value_label} CV% by Group and Bin")
    format_axis(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to:\n{save_path}")

def plot_per_group(data: pd.DataFrame, bin_col: str, bin_order: list[str], value_label: str, save_prefix: str):
    for g in group_order:
        group_df = data[data[group_col] == g].copy()
        if group_df.empty:
            print(f"Skipping {g}: no data")
            continue

        fig, ax = plt.subplots(figsize=(9, 6))

        box_df = boxplot_subset(group_df, [bin_col])

        if not box_df.empty:
            sns.boxplot(
                data=box_df,
                x=bin_col,
                y="CV%",
                order=bin_order,
                color="white",
                width=0.45,
                showfliers=False,
                ax=ax,
                **box_style,
            )

        sns.stripplot(
            data=group_df,
            x=bin_col,
            y="CV%",
            order=bin_order,
            color=group_palette.get(g, "#7B1FA2"),
            size=4.5,
            alpha=0.85,
            jitter=0.18,
            ax=ax,
        )

        ax.legend(
            handles=cutoff_legend(),
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )

        ax.set_xlabel(value_label)
        ax.set_title(f"{file_stem} - {value_label} CV% - {g}")
        format_axis(ax)

        plt.tight_layout()
        out_file = os.path.join(save_dir, f"{file_stem}_{save_prefix}_{clean_filename(g)}_CV_by_bin.png")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"\nFigure saved to:\n{out_file}")

# =============================================================================
# Generate figures
# =============================================================================

plot_all_groups(
    data=cv_npx,
    bin_col="NPX Mean Range",
    bin_order=NPX_BIN_LABELS,
    value_label="NPX Mean Range",
    save_path=os.path.join(save_dir, f"{file_stem}_NPX_CV_by_bins_all_groups.png"),
)

plot_per_group(
    data=cv_npx,
    bin_col="NPX Mean Range",
    bin_order=NPX_BIN_LABELS,
    value_label="NPX Mean Range",
    save_prefix="NPX",
)

plot_all_groups(
    data=cv_pg,
    bin_col="pg/mL Mean Range",
    bin_order=PG_BIN_LABELS,
    value_label="pg/mL Mean Range",
    save_path=os.path.join(save_dir, f"{file_stem}_pg_mL_CV_by_bins_all_groups.png"),
)

plot_per_group(
    data=cv_pg,
    bin_col="pg/mL Mean Range",
    bin_order=PG_BIN_LABELS,
    value_label="pg/mL Mean Range",
    save_prefix="pg_mL",
)

print("\nAnalysis complete.")

