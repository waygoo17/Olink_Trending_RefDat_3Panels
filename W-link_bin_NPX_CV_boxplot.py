import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\NPX_CV%.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\CV"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

output_csv = os.path.join(save_dir, f"{file_stem}_CV_percent_by_NPX_Mean_Range.csv")
summary_csv = os.path.join(save_dir, f"{file_stem}_CV_percent_by_NPX_Mean_Range_summary.csv")

save_file_100 = os.path.join(
    save_dir,
    f"{file_stem}_CV_percent_by_NPX_Mean_Range_boxplot_Y0-100.png"
)

save_file_95 = os.path.join(
    save_dir,
    f"{file_stem}_CV_percent_by_NPX_Mean_Range_boxplot_95thPercentile.png"
)

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

# =============================================================================
# Clean columns
# =============================================================================

df["CV%"] = (
    df["CV%"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.strip()
)

df["CV%"] = pd.to_numeric(df["CV%"], errors="coerce")
df["Mean value"] = pd.to_numeric(df["Mean value"], errors="coerce")

df = df.dropna(subset=["Mean value", "CV%"])

# =============================================================================
# Bin by NPX Mean value
# 0-1, 1-2, 2-4, 4-8, >8
# =============================================================================

bin_edges = [0, 1, 2, 4, 8, np.inf]
bin_labels = ["0–1", "1–2", "2–4", "4–8", ">8"]

df["NPX Mean Range"] = pd.cut(
    df["Mean value"],
    bins=bin_edges,
    labels=bin_labels,
    include_lowest=True,
    right=True
)

df_plot = df.dropna(subset=["NPX Mean Range"]).copy()

df_plot["NPX Mean Range"] = pd.Categorical(
    df_plot["NPX Mean Range"],
    categories=bin_labels,
    ordered=True
)

df_plot.to_csv(output_csv, index=False)

# =============================================================================
# Summary statistics
# =============================================================================

summary_df = (
    df_plot
    .groupby("NPX Mean Range", observed=False)["CV%"]
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
    df_plot.groupby("NPX Mean Range", observed=False)["CV%"]
    .quantile(0.75)
    .values
)

summary_df["P90_CV_percent"] = (
    df_plot.groupby("NPX Mean Range", observed=False)["CV%"]
    .quantile(0.90)
    .values
)

summary_df["P95_CV_percent"] = (
    df_plot.groupby("NPX Mean Range", observed=False)["CV%"]
    .quantile(0.95)
    .values
)

summary_df.to_csv(summary_csv, index=False)

print(f"Rows in original cleaned data: {len(df)}")
print(f"Rows used for plotting: {len(df_plot)}")

print("\nCounts by NPX Mean Range:")
print(df_plot["NPX Mean Range"].value_counts().sort_index())

print("\nSummary statistics:")
print(summary_df)

print(f"\nGrouped data saved to:\n{output_csv}")
print(f"\nSummary statistics saved to:\n{summary_csv}")

# =============================================================================
# Plot 1 - Fixed Y-axis 0-100
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(
    data=df_plot,
    x="NPX Mean Range",
    y="CV%",
    order=bin_labels,
    color="white",
    width=0.45,
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=df_plot,
    x="NPX Mean Range",
    y="CV%",
    order=bin_labels,
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

ax.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20% CV")
ax.axhline(y=35, color="red", linestyle="--", linewidth=1.5, label="35% CV")

ax.legend(frameon=False)

ax.set_xlabel("NPX Mean Range")
ax.set_ylabel("CV%")
ax.set_title(f"{file_stem} - CV% by NPX Mean Range")

plt.tight_layout()
plt.savefig(save_file_100, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFixed Y-axis figure saved to:\n{save_file_100}")

# =============================================================================
# Plot 2 - 95th percentile Y-axis
# =============================================================================

ymax95 = df_plot["CV%"].quantile(0.95)

print(f"\n95th percentile CV% = {ymax95:.2f}")

fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(
    data=df_plot,
    x="NPX Mean Range",
    y="CV%",
    order=bin_labels,
    color="white",
    width=0.45,
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=df_plot,
    x="NPX Mean Range",
    y="CV%",
    order=bin_labels,
    color="#7B1FA2",
    size=4,
    alpha=0.75,
    jitter=0.18,
    ax=ax
)

ax.set_ylim(0, ymax95)
ax.yaxis.set_major_locator(MultipleLocator(10))

ax.grid(
    axis="y",
    linestyle=":",
    linewidth=0.5,
    alpha=0.5
)

ax.axhline(y=20, color="red", linestyle="--", linewidth=1.5, label="20% CV")
ax.axhline(y=35, color="red", linestyle="--", linewidth=1.5, label="35% CV")

ax.legend(frameon=False)

ax.set_xlabel("NPX Mean Range")
ax.set_ylabel("CV%")
ax.set_title(f"{file_stem} - CV% by NPX Mean Range (95th Percentile)")

plt.tight_layout()
plt.savefig(save_file_95, dpi=300, bbox_inches="tight")
plt.show()

print(f"\n95th percentile figure saved to:\n{save_file_95}")

print("\nAnalysis complete.")





