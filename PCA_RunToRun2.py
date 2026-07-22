import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

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
group_col = "Group"

npx_run_cols = ["NPX_Run1", "NPX_Run2", "NPX_Run3"]
run_order = ["Run1", "Run2", "Run3"]

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())

for col in npx_run_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[sample_col, target_col])

# =============================================================================
# Convert wide run columns to long format
# =============================================================================

id_vars = [sample_col, target_col]

if group_col in df.columns:
    id_vars.append(group_col)

long_df = df.melt(
    id_vars=id_vars,
    value_vars=npx_run_cols,
    var_name="Run",
    value_name="NPX"
)

long_df["Run"] = long_df["Run"].str.replace("NPX_", "", regex=False)

long_df["Sample_Run"] = (
    long_df[sample_col].astype(str)
    + "_"
    + long_df["Run"].astype(str)
)

long_df = long_df.dropna(subset=["NPX"])

print("\nLong-format rows:")
print(len(long_df))

print("\nSamples per run:")
print(long_df.groupby("Run")[sample_col].nunique())

print("\nTargets per run:")
print(long_df.groupby("Run")[target_col].nunique())

# =============================================================================
# Create PCA matrix
# Rows = Sample_Run
# Columns = Targets
# Values = NPX
# =============================================================================

matrix_raw = long_df.pivot_table(
    index="Sample_Run",
    columns=target_col,
    values="NPX",
    aggfunc="mean"
)

metadata = (
    long_df[[sample_col, "Run", "Sample_Run"] + ([group_col] if group_col in long_df.columns else [])]
    .drop_duplicates()
    .set_index("Sample_Run")
    .loc[matrix_raw.index]
)

missing_count = matrix_raw.isna().sum().sum()

matrix_df = matrix_raw.apply(lambda x: x.fillna(x.median()), axis=0)
matrix_df = matrix_df.loc[:, matrix_df.std(axis=0) > 0]

print(f"\nPCA matrix shape: {matrix_df.shape}")
print("Rows = sample-run combinations")
print("Columns = targets")
print(f"Missing values filled by target median: {missing_count}")

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

loading_df["Abs_PC1_Loading"] = loading_df["PC1_Loading"].abs()
loading_df["Abs_PC2_Loading"] = loading_df["PC2_Loading"].abs()

pca_scores_file = os.path.join(save_dir, f"{file_stem}_NPX_PCA_scores.csv")
loading_file = os.path.join(save_dir, f"{file_stem}_NPX_PCA_loadings.csv")

pca_df.to_csv(pca_scores_file)
loading_df.to_csv(loading_file)

print(f"\nPCA matrix saved to:\n{matrix_file}")
print(f"PCA scores saved to:\n{pca_scores_file}")
print(f"PCA loadings saved to:\n{loading_file}")

# =============================================================================
# Plot helpers
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

def save_show(fig_file):
    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to:\n{fig_file}")

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
    hue_order=run_order,
    s=80,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
ax.set_title(f"{file_stem} - PCA colored by Run")
ax.legend(frameon=False, title="Run")

format_plot(ax)
save_show(fig_file)

# =============================================================================
# Figure 2: PCA colored by Group, style by Run
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
        style_order=run_order,
        s=80,
        edgecolor="black",
        linewidth=0.6,
        ax=ax
    )

    ax.set_xlabel(f"PC1 ({explained_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_ratio[1] * 100:.1f}%)")
    ax.set_title(f"{file_stem} - PCA colored by Group")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")

    format_plot(ax)
    save_show(fig_file)

else:
    print("\nNo Group column found. Skipping PCA colored by Group.")

# =============================================================================
# Figure 3: PCA with same sample connected across runs
# =============================================================================

fig_file = os.path.join(save_dir, f"{file_stem}_PCA_sample_connections.png")

fig, ax = plt.subplots(figsize=(8, 7))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Run",
    hue_order=run_order,
    s=80,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

for sample_id, sub in pca_df.groupby(sample_col):
    sub = sub.copy()
    sub["Run"] = pd.Categorical(sub["Run"], categories=run_order, ordered=True)
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
save_show(fig_file)

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
save_show(fig_file)

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

    print(f"\nFigure saved to:\n{fig_file}")

else:
    print("\nSciPy is not installed. Creating non-clustered NPX heatmap instead.")

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
    save_show(fig_file)

# =============================================================================
# Figure 6: Mean NPX distribution by run
# =============================================================================

mean_npx_df = metadata.copy()
mean_npx_df["Mean_NPX_across_targets"] = matrix_df.mean(axis=1)
mean_npx_df["Median_NPX_across_targets"] = matrix_df.median(axis=1)

mean_npx_file = os.path.join(save_dir, f"{file_stem}_sample_run_mean_NPX.csv")
mean_npx_df.to_csv(mean_npx_file)

fig_file = os.path.join(save_dir, f"{file_stem}_mean_NPX_by_Run.png")

fig, ax = plt.subplots(figsize=(7, 6))

sns.boxplot(
    data=mean_npx_df,
    x="Run",
    y="Mean_NPX_across_targets",
    order=run_order,
    color="white",
    width=0.45,
    showfliers=False,
    boxprops=dict(edgecolor="black", linewidth=1.2),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
    ax=ax
)

sns.stripplot(
    data=mean_npx_df,
    x="Run",
    y="Mean_NPX_across_targets",
    order=run_order,
    color="#7B1FA2",
    size=4,
    alpha=0.8,
    jitter=0.18,
    ax=ax
)

ax.set_xlabel("Run")
ax.set_ylabel("Mean NPX across targets")
ax.set_title(f"{file_stem} - Mean NPX Distribution by Run")

format_plot(ax)
save_show(fig_file)

print(f"Mean NPX table saved to:\n{mean_npx_file}")

# =============================================================================
# Figure 7: Top PCA loading targets
# =============================================================================

top_n = 10

top_pc1 = loading_df.sort_values("Abs_PC1_Loading", ascending=False).head(top_n).copy()
top_pc2 = loading_df.sort_values("Abs_PC2_Loading", ascending=False).head(top_n).copy()

top_pc1_file = os.path.join(save_dir, f"{file_stem}_top_PC1_loadings.csv")
top_pc2_file = os.path.join(save_dir, f"{file_stem}_top_PC2_loadings.csv")

top_pc1.to_csv(top_pc1_file)
top_pc2.to_csv(top_pc2_file)

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
save_show(fig_file)

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
save_show(fig_file)

print(f"Top PC1 loading table saved to:\n{top_pc1_file}")
print(f"Top PC2 loading table saved to:\n{top_pc2_file}")

# =============================================================================
# Figure 8: Same-sample run-to-run correlations
# =============================================================================

pairwise_corr_rows = []

for sample_id, sub_meta in metadata.groupby(sample_col):

    sample_runs = sub_meta.index.tolist()

    available_runs = {
        metadata.loc[sr, "Run"]: sr
        for sr in sample_runs
        if sr in matrix_df.index
    }

    for r1, r2 in [("Run1", "Run2"), ("Run1", "Run3"), ("Run2", "Run3")]:

        if r1 in available_runs and r2 in available_runs:

            sr1 = available_runs[r1]
            sr2 = available_runs[r2]

            corr = matrix_df.loc[sr1].corr(matrix_df.loc[sr2])

            pairwise_corr_rows.append(
                {
                    sample_col: sample_id,
                    "Run_Pair": f"{r1}_vs_{r2}",
                    "Pearson_r": corr
                }
            )

pairwise_corr_df = pd.DataFrame(pairwise_corr_rows)

pairwise_corr_file = os.path.join(
    save_dir,
    f"{file_stem}_same_sample_run_to_run_correlations.csv"
)

pairwise_corr_df.to_csv(pairwise_corr_file, index=False)

fig_file = os.path.join(save_dir, f"{file_stem}_same_sample_run_to_run_correlations.png")

fig, ax = plt.subplots(figsize=(8, 6))

sns.boxplot(
    data=pairwise_corr_df,
    x="Run_Pair",
    y="Pearson_r",
    color="white",
    width=0.45,
    showfliers=False,
    boxprops=dict(edgecolor="black", linewidth=1.2),
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
    ax=ax
)

sns.stripplot(
    data=pairwise_corr_df,
    x="Run_Pair",
    y="Pearson_r",
    color="#7B1FA2",
    size=4,
    alpha=0.8,
    jitter=0.18,
    ax=ax
)

ax.set_ylim(0, 1.05)
ax.set_xlabel("Run pair")
ax.set_ylabel("Pearson r across targets")
ax.set_title(f"{file_stem} - Same-sample Run-to-Run Correlation")
ax.tick_params(axis="x", labelrotation=20)

format_plot(ax)
save_show(fig_file)

print(f"Same-sample run-to-run correlation table saved to:\n{pairwise_corr_file}")

print("\nAnalysis complete.")


