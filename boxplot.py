import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\corr.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\CV"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

save_file = os.path.join(save_dir, f"{file_stem}_CV_percent_boxplot.png")
output_csv = os.path.join(save_dir, f"{file_stem}_CV_percent_results.csv")

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

# Use numeric columns only
run_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("Run columns used:")
print(run_cols)

# =============================================================================
# Calculate row-wise CV%
# CV% = standard deviation / mean * 100
# Exclude rows with 2 or fewer valid data points
# =============================================================================

df["Valid_N"] = df[run_cols].notna().sum(axis=1)

df_cv = df[df["Valid_N"] > 2].copy()

df_cv["Mean"] = df_cv[run_cols].mean(axis=1)
df_cv["SD"] = df_cv[run_cols].std(axis=1, ddof=1)
df_cv["CV_percent"] = (df_cv["SD"] / df_cv["Mean"]) * 100

# Remove invalid CV values
df_cv = df_cv.replace([np.inf, -np.inf], np.nan)
df_cv = df_cv.dropna(subset=["CV_percent"])

# Save CV results
df_cv.to_csv(output_csv, index=False)

print(f"\nRows in original file: {len(df)}")
print(f"Rows used for CV%: {len(df_cv)}")
print(f"CV% results saved to:\n{output_csv}")

# =============================================================================
# Boxplot with purple dots
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 6))

sns.boxplot(
    y=df_cv["CV_percent"],
    color="white",
    width=0.35,
    showfliers=False,
    ax=ax
)

sns.stripplot(
    y=df_cv["CV_percent"],
    color="#7B1FA2",
    size=4,
    alpha=0.75,
    jitter=0.18,
    ax=ax
)

ax.set_ylabel("CV%")
ax.set_xlabel("")
ax.set_title(f"{file_stem} - Row-wise CV% Across Three Runs")

plt.tight_layout()

plt.savefig(save_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure saved to:\n{save_file}")


