from email.mime import base
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# =============================================================================
# Input / Output
# =============================================================================
data_file = r"C:\Users\wei.guo2\Python\datasheet\T48Immune_woOOR.csv"

save_dir = r"C:\Users\wei.guo2\Python\PCA_Regeneron"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

# =============================================================================
# Helper: robust CSV reader
# =============================================================================

def read_csv_with_fallback(path):
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    last_error = None

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded CSV using encoding: {enc}")
            return df
        except UnicodeDecodeError as e:
            last_error = e
            print(f"Failed with encoding: {enc}")

    raise last_error

# =============================================================================
# Read data
# =============================================================================

df = read_csv_with_fallback(data_file)

# Clean header whitespace and hidden non-breaking spaces
df.columns = (
    df.columns.astype(str)
    .str.replace("\ufeff", "", regex=False)
    .str.replace("ï»¿", "", regex=False)
    .str.replace("\xa0", " ", regex=False)
    .str.strip()
)

sample_col = "SampleID"
target_col = "Assay"
group_col = "Group"
value_col = "NPX"

required_columns = [sample_col, target_col, group_col, value_col]

print("\nColumns detected:")
print(df.columns.tolist())

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required column(s): {missing}\n"
        f"Columns found: {df.columns.tolist()}"
    )


# Clean possible hidden spaces in key text columns
for col in [sample_col, target_col, group_col]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

# Convert NPX to numeric
df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
df = df.dropna(subset=[sample_col, target_col, group_col, value_col])

print("Columns found:")
print(df.columns.tolist())

print("\nData shape:")
print(df.shape)

print("\nGroups:")
print(df[group_col].value_counts())

print("\nNumber of samples:")
print(df[sample_col].nunique())

print("\nNumber of targets:")
print(df[target_col].nunique())

# =============================================================================
# Create PCA matrix
# Rows = samples
# Columns = targets
# Values = NPX
# =============================================================================

matrix_df = df.pivot_table(
    index=sample_col,
    columns=target_col,
    values=value_col,
    aggfunc="mean"
)

metadata = (
    df[[sample_col, group_col]]
    .drop_duplicates()
    .set_index(sample_col)
    .loc[matrix_df.index]
)

# Fill missing values by target median
matrix_df = matrix_df.apply(lambda x: x.fillna(x.median()), axis=0)

# Remove zero-variance targets
matrix_df = matrix_df.loc[:, matrix_df.std(axis=0) > 0]

print(f"\nPCA matrix shape: {matrix_df.shape}")
print("Rows = samples")
print("Columns = targets")

matrix_file = os.path.join(save_dir, f"{file_stem}_NPX_PCA_matrix.csv")
matrix_df.to_csv(matrix_file)

# =============================================================================
# Standardize and PCA using NumPy SVD
# =============================================================================

X = matrix_df.values
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

U, S, VT = np.linalg.svd(X_scaled, full_matrices=False)

scores = U[:, :2] * S[:2]

explained_variance = (S ** 2) / (X_scaled.shape[0] - 1)
explained_ratio = explained_variance / explained_variance.sum()

pca_df = metadata.copy()
pca_df["PC1"] = scores[:, 0]
pca_df["PC2"] = scores[:, 1]

loading_df = pd.DataFrame(
    VT[:2].T,
    index=matrix_df.columns,
    columns=["PC1_Loading", "PC2_Loading"]
)

pca_scores_file = os.path.join(save_dir, f"{file_stem}_NPX_PCA_scores.csv")
loading_file = os.path.join(save_dir, f"{file_stem}_NPX_PCA_loadings.csv")

pca_df.to_csv(pca_scores_file)
loading_df.to_csv(loading_file)

print(f"\nPCA matrix saved to:\n{matrix_file}")
print(f"PCA scores saved to:\n{pca_scores_file}")
print(f"PCA loadings saved to:\n{loading_file}")

# =============================================================================
# Plot helper
# =============================================================================

def format_plot(ax):
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.grid(
        linestyle=":",
        linewidth=0.5,
        alpha=0.5
    )

# =============================================================================
# Figure 1: PCA colored by Group
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_PCA_colored_by_Group.png")

fig, ax = plt.subplots(figsize=(7, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue=group_col,
    s=90,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
ax.set_title(f"{file_stem} - PCA colored by Group")
ax.legend(frameon=False, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")

format_plot(ax)

plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 1 saved to:\n{fig_file}")

# =============================================================================
# Figure 2: PCA labeled by SampleID
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_PCA_labeled_by_SampleID.png")

fig, ax = plt.subplots(figsize=(8, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue=group_col,
    s=80,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

for sample_id, row in pca_df.iterrows():
    ax.text(
        row["PC1"],
        row["PC2"],
        str(sample_id),
        fontsize=7,
        alpha=0.75
    )

ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
ax.set_title(f"{file_stem} - PCA labeled by SampleID")
ax.legend(frameon=False, title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")

format_plot(ax)

plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 2 saved to:\n{fig_file}")

# =============================================================================
# Figure 3: Hierarchical clustering heatmap
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_NPX_hierarchical_clustering_heatmap.png")

group_palette = dict(
    zip(
        metadata[group_col].dropna().unique(),
        sns.color_palette("tab10", metadata[group_col].nunique())
    )
)

row_colors = metadata[group_col].map(group_palette)

g = sns.clustermap(
    matrix_df,
    z_score=1,
    method="average",
    metric="correlation",
    cmap="vlag",
    center=0,
    row_colors=row_colors,
    figsize=(12, 10),
    xticklabels=True,
    yticklabels=True
)

g.fig.suptitle(
    f"{file_stem} - NPX Hierarchical Clustering Heatmap",
    y=1.02
)

g.fig.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 3 saved to:\n{fig_file}")

# =============================================================================
# Figure 4: Sample correlation heatmap
# =============================================================================

corr_df = matrix_df.T.corr(method="pearson")

corr_file = os.path.join(save_dir, f"{file_stem}_sample_correlation_matrix.csv")
corr_df.to_csv(corr_file)

fig_file = os.path.join(save_dir, f"{file_stem}_sample_correlation_heatmap.png")

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_df,
    cmap="vlag",
    center=0,
    vmin=0.5,
    vmax=1,
    square=True,
    cbar_kws={"label": "Pearson r"}
)

plt.title(f"{file_stem} - Sample Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 4 saved to:\n{fig_file}")
print(f"Correlation matrix saved to:\n{corr_file}")

# =============================================================================
# Figure 5: Top PCA loadings
# =============================================================================

top_n = 10

loading_df["Abs_PC1_Loading"] = loading_df["PC1_Loading"].abs()
loading_df["Abs_PC2_Loading"] = loading_df["PC2_Loading"].abs()

top_pc1 = loading_df.sort_values("Abs_PC1_Loading", ascending=False).head(top_n).copy()
top_pc2 = loading_df.sort_values("Abs_PC2_Loading", ascending=False).head(top_n).copy()

top_pc1_file = os.path.join(save_dir, f"{file_stem}_top_PC1_loadings.csv")
top_pc2_file = os.path.join(save_dir, f"{file_stem}_top_PC2_loadings.csv")

top_pc1.to_csv(top_pc1_file)
top_pc2.to_csv(top_pc2_file)

# PC1 loading plot
fig_file = os.path.join(save_dir, f"{file_stem}_top_PC1_loadings.png")

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=top_pc1.reset_index(),
    x="PC1_Loading",
    y=target_col,
    color="#7B1FA2",
    ax=ax
)

ax.set_xlabel("PC1 loading")
ax.set_ylabel("Target")
ax.set_title(f"{file_stem} - Top {top_n} PC1 Loading Targets")
ax.xaxis.set_major_locator(MaxNLocator(5))

format_plot(ax)

plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 5A saved to:\n{fig_file}")

# PC2 loading plot
fig_file = os.path.join(save_dir, f"{file_stem}_top_PC2_loadings.png")

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=top_pc2.reset_index(),
    x="PC2_Loading",
    y=target_col,
    color="#7B1FA2",
    ax=ax
)

ax.set_xlabel("PC2 loading")
ax.set_ylabel("Target")
ax.set_title(f"{file_stem} - Top {top_n} PC2 Loading Targets")
ax.xaxis.set_major_locator(MaxNLocator(5))

format_plot(ax)

plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 5B saved to:\n{fig_file}")

print("\nAnalysis completed successfully.")
