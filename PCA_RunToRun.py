import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Optional hierarchical clustering
try:
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\PCA_woCtrl_RtR.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\PCA_Clustering_NPX"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

# =============================================================================
# Column names
# =============================================================================

sample_col = "SampleID"
target_col = "Assay"
group_col = "Group"   # optional; script still works if missing

npx_run_cols = ["NPX_Run1", "NPX_Run2", "NPX_Run3"]

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())

for col in npx_run_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

required_cols = [sample_col, target_col] + npx_run_cols
df = df.dropna(subset=[sample_col, target_col])

# =============================================================================
# Convert to PCA matrix
# Rows = Sample_Run
# Columns = Targets
# =============================================================================

long_df = df.melt(
    id_vars=[sample_col, target_col] + ([group_col] if group_col in df.columns else []),
    value_vars=npx_run_cols,
    var_name="Run",
    value_name="NPX"
)

long_df["Run"] = long_df["Run"].str.replace("NPX_", "", regex=False)
long_df["Sample_Run"] = long_df[sample_col].astype(str) + "_" + long_df["Run"].astype(str)

long_df = long_df.dropna(subset=["NPX"])

matrix_df = long_df.pivot_table(
    index="Sample_Run",
    columns=target_col,
    values="NPX",
    aggfunc="mean"
)

metadata = (
    long_df[[sample_col, "Run", "Sample_Run"] + ([group_col] if group_col in long_df.columns else [])]
    .drop_duplicates()
    .set_index("Sample_Run")
    .loc[matrix_df.index]
)

# Fill missing target values with target median
matrix_df = matrix_df.apply(lambda x: x.fillna(x.median()), axis=0)

# Remove targets with no variance
matrix_df = matrix_df.loc[:, matrix_df.std(axis=0) > 0]

print(f"\nPCA matrix shape: {matrix_df.shape}")
print("Rows = sample-run combinations")
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

print(f"\nPCA scores saved to:\n{pca_scores_file}")
print(f"PCA loadings saved to:\n{loading_file}")

# =============================================================================
# Plot style helper
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
# Figure 1: PCA colored by Run
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_PCA_colored_by_Run.png")

fig, ax = plt.subplots(figsize=(7, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Run",
    s=70,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
ax.set_title(f"{file_stem} - PCA colored by Run")
ax.legend(frameon=False, title="Run")

format_plot(ax)

plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 1 saved to:\n{fig_file}")

# =============================================================================
# Figure 2: PCA colored by Group
# =============================================================================

if group_col in pca_df.columns:

    fig_file = os.path.join(save_dir, f"{file_stem}_PCA_colored_by_Group.png")

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue=group_col,
        style="Run",
        s=70,
        edgecolor="black",
        linewidth=0.5,
        ax=ax
    )

    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    ax.set_title(f"{file_stem} - PCA colored by Group")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    format_plot(ax)

    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFigure 2 saved to:\n{fig_file}")

else:
    print("\nNo Group column found. Skipping PCA colored by Group.")

# =============================================================================
# Figure 3: PCA with same sample connected across Run1, Run2, Run3
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_PCA_sample_connections.png")

fig, ax = plt.subplots(figsize=(8, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Run",
    s=70,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)

for sample_id, sub in pca_df.groupby(sample_col):
    sub = sub.sort_values("Run")

    if len(sub) >= 2:
        ax.plot(
            sub["PC1"],
            sub["PC2"],
            color="gray",
            linewidth=0.8,
            alpha=0.45
        )

ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
ax.set_title(f"{file_stem} - PCA with Run Connections per Sample")
ax.legend(frameon=False, title="Run")

format_plot(ax)

plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 3 saved to:\n{fig_file}")

# =============================================================================
# Figure 4: Sample-to-sample Pearson correlation heatmap
# =============================================================================

corr_df = matrix_df.T.corr(method="pearson")

corr_file = os.path.join(save_dir, f"{file_stem}_sample_run_correlation_matrix.csv")
corr_df.to_csv(corr_file)

fig_file = os.path.join(save_dir, f"{file_stem}_sample_run_correlation_heatmap.png")

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_df,
    cmap="vlag",
    center=0,
    vmin=0.5,
    vmax=1,
    square=True,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"label": "Pearson r"}
)

plt.title(f"{file_stem} - Sample/Run Pearson Correlation Heatmap")
plt.tight_layout()
plt.savefig(fig_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure 4 saved to:\n{fig_file}")
print(f"Correlation matrix saved to:\n{corr_file}")

# =============================================================================
# Figure 5: Hierarchical clustering heatmap
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_hierarchical_clustering_heatmap.png")

if SCIPY_AVAILABLE:

    row_colors = None

    if group_col in metadata.columns:
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
        yticklabels=False
    )

    g.fig.suptitle(
        f"{file_stem} - Hierarchical Clustering Heatmap",
        y=1.02
    )

    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFigure 5 saved to:\n{fig_file}")

else:
    print("\nSciPy is not installed, so true hierarchical clustering cannot be generated.")
    print("Creating non-clustered NPX heatmap instead.")

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        matrix_df,
        cmap="vlag",
        center=0,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "NPX"}
    )

    plt.title(f"{file_stem} - NPX Heatmap")
    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nFallback heatmap saved to:\n{fig_file}")

print("\nAnalysis complete.")

