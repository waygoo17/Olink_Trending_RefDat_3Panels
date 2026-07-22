import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\pg_CV%.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\CV_pg_mL"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

output_csv = os.path.join(save_dir, f"{file_stem}_CV_percent_by_pg_mL_range.csv")
summary_csv = os.path.join(save_dir, f"{file_stem}_CV_percent_by_pg_mL_range_summary.csv")

save_file_100 = os.path.join(
    save_dir,
    f"{file_stem}_CV_percent_by_pg_mL_range_boxplot_Y0-100.png"
)

save_file_95 = os.path.join(
    save_dir,
    f"{file_stem}_CV_percent_by_pg_mL_range_boxplot_95thPercentile.png"
)

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())

# =============================================================================
# Define columns
# =============================================================================

cv_col = "CV%"
mean_col = "Mean value"

# =============================================================================
# Clean columns
# =============================================================================

df[cv_col] = (
    df[cv_col]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)

df[cv_col] = pd.to_numeric(df[cv_col], errors="coerce")
df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")

df = df.dropna(subset=[mean_col, cv_col])

# =============================================================================
# Bin by pg/mL mean value
# 0-10, 10-200, 200-500, 500-1000, >1000
# =============================================================================

conditions = [
    (df[mean_col] >= 0) & (df[mean_col] <= 10),
    (df[mean_col] > 10) & (df[mean_col] <= 200),
    (df[mean_col] > 200) & (df[mean_col] <= 500),
    (df[mean_col] > 500) & (df[mean_col] <= 1000),
    (df[mean_col] > 1000)
]

choices = ["0–10", "10–200", "200–500", "500–1000", ">1000"]

df["pg_mL_Range"] = np.select(
    conditions,
    choices,
    default=None
)

group_order = ["0–10", "10–200", "200–500", "500–1000", ">1000"]

df_plot = df.dropna(subset=["pg_mL_Range"]).copy()

df_plot["pg_mL_Range"] = pd.Categorical(
    df_plot["pg_mL_Range"],
    categories=group_order,
    ordered=True
)

df_plot.to_csv(output_csv, index=False)

# =============================================================================
# Summary statistics
# =============================================================================

summary_df = (
    df_plot
    .groupby("pg_mL_Range", observed=False)[cv_col]
    .agg(
        N="count",
        Mean_CV_percent="mean",
        Median_CV_percent="median",
        SD_CV_percent="std",
        Min_CV_percent="min",
        Max_CV_percent="max"
    )
    .reset_index()
)

summary_df["P75_CV_percent"] = (
    df_plot.groupby("pg_mL_Range", observed=False)[cv_col]
    .quantile(0.75)
    .values
)

summary_df["P90_CV_percent"] = (
    df_plot.groupby("pg_mL_Range", observed=False)[cv_col]
    .quantile(0.90)
    .values
)

summary_df["P95_CV_percent"] = (
    df_plot.groupby("pg_mL_Range", observed=False)[cv_col]
    .quantile(0.95)
    .values
)

summary_df.to_csv(summary_csv, index=False)

print(f"\nRows in cleaned data: {len(df)}")
print(f"Rows used for plotting: {len(df_plot)}")

print("\nCounts by pg/mL range:")
print(df_plot["pg_mL_Range"].value_counts().sort_index())

print("\nSummary statistics:")
print(summary_df)

print(f"\nGrouped data saved to:\n{output_csv}")
print(f"\nSummary statistics saved to:\n{summary_csv}")

# =============================================================================
# Function for plotting
# =============================================================================

def make_cv_boxplot(ymax, title_suffix, save_file):
    fig, ax = plt.subplots(figsize=(9, 6))

    sns.boxplot(
        data=df_plot,
        x="pg_mL_Range",
        y=cv_col,
        order=group_order,
        color="white",
        width=0.45,
        showfliers=False,
        ax=ax
    )

    sns.stripplot(
        data=df_plot,
        x="pg_mL_Range",
        y=cv_col,
        order=group_order,
        color="#7B1FA2",
        size=4,
        alpha=0.75,
        jitter=0.18,
        ax=ax
    )

    ax.set_ylim(0, ymax)
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

    ax.legend(frameon=False)

    ax.set_xlabel("pg/mL range")
    ax.set_ylabel("CV%")
    ax.set_title(f"{file_stem} - CV% by pg/mL Range {title_suffix}")

    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFigure saved to:\n{save_file}")

# =============================================================================
# Plot 1 - Fixed Y-axis 0-100
# =============================================================================

make_cv_boxplot(
    ymax=100,
    title_suffix="",
    save_file=save_file_100
)

# =============================================================================
# Plot 2 - 95th percentile Y-axis
# =============================================================================

ymax95 = df_plot[cv_col].quantile(0.95)

print(f"\n95th percentile CV% = {ymax95:.2f}")

make_cv_boxplot(
    ymax=100,
    title_suffix="(95th Percentile)",
    save_file=save_file_95
)

print("\nAnalysis complete.")


