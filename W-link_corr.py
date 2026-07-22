import os
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\corr.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure\Correlation"

os.makedirs(save_dir, exist_ok=True)

file_stem = os.path.splitext(os.path.basename(data_file))[0]

pdf_file = os.path.join(
    save_dir,
    f"{file_stem}_correlation_figures.pdf"
)

# =============================================================================
# Read data
# =============================================================================

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

numeric_df = df.select_dtypes(include=[np.number]).copy()

print("\nNumeric columns found:")
print(numeric_df.columns.tolist())

# =============================================================================
# Pairwise correlation analysis + PDF output
# =============================================================================

results = []

with PdfPages(pdf_file) as pdf:

    for x_col, y_col in itertools.combinations(numeric_df.columns, 2):

        temp = numeric_df[[x_col, y_col]].dropna()

        x = temp[x_col].to_numpy()
        y = temp[y_col].to_numpy()

        if len(temp) < 3:
            print(f"Skipping {y_col} vs {x_col}: insufficient data")
            continue

        # ---------------------------------------------------------------------
        # Linear regression
        # ---------------------------------------------------------------------

        slope, intercept = np.polyfit(x, y, 1)

        y_pred = slope * x + intercept

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot)

        # ---------------------------------------------------------------------
        # Pearson correlation
        # ---------------------------------------------------------------------

        pearson_r = np.corrcoef(x, y)[0, 1]

        # ---------------------------------------------------------------------
        # Spearman correlation
        # ---------------------------------------------------------------------

        x_rank = pd.Series(x).rank().to_numpy()
        y_rank = pd.Series(y).rank().to_numpy()

        spearman_r = np.corrcoef(x_rank, y_rank)[0, 1]

        # ---------------------------------------------------------------------
        # Save statistics
        # ---------------------------------------------------------------------

        results.append({
            "X": x_col,
            "Y": y_col,
            "Slope": slope,
            "Intercept": intercept,
            "Equation": f"y = {slope:.3f}x + {intercept:.3f}",
            "R2": r2,
            "Pearson_r": pearson_r,
            "Spearman_r": spearman_r
        })

        # ---------------------------------------------------------------------
        # Plot
        # ---------------------------------------------------------------------

        fig, ax = plt.subplots(figsize=(7, 6))

        sns.regplot(
            x=x,
            y=y,
            scatter_kws={
                "s": 35,
                "alpha": 0.8,
                "color": "#7B1FA2"     # purple dots
            },
            line_kws={
                "linewidth": 2,
                "color": "black"       # black regression line
            },
            ax=ax
        )

        equation_text = (
            f"y = {slope:.3f}x + {intercept:.3f}\n"
            f"R² = {r2:.3f}\n"
            f"Pearson r = {pearson_r:.3f}\n"
            f"Spearman r = {spearman_r:.3f}"
        )

        ax.text(
            0.05,
            0.95,
            equation_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="none",
                edgecolor="black",
                linewidth=1
            )
        )

        ax.set_xlabel(x_col, fontsize=11)
        ax.set_ylabel(y_col, fontsize=11)
        ax.set_title(
            f"{file_stem}: {y_col} vs {x_col}",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()

        plot_file = os.path.join(
            save_dir,
            f"{file_stem}_{y_col}_vs_{x_col}_correlation.png"
        )

        fig.savefig(
            plot_file,
            dpi=300,
            bbox_inches="tight"
        )

        pdf.savefig(fig, bbox_inches="tight")

        plt.show()

        print(f"\nSaved plot:\n{plot_file}")

    # =========================================================================
    # Pearson correlation matrix
    # =========================================================================

    pearson_matrix = numeric_df.corr(method="pearson")

    pearson_matrix_file = os.path.join(
        save_dir,
        f"{file_stem}_pearson_correlation_matrix.csv"
    )

    pearson_matrix.to_csv(pearson_matrix_file)

    # =========================================================================
    # Heatmap
    # =========================================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        pearson_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        f"{file_stem}: Pearson Correlation Matrix",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    heatmap_file = os.path.join(
        save_dir,
        f"{file_stem}_pearson_correlation_heatmap.png"
    )

    fig.savefig(
        heatmap_file,
        dpi=300,
        bbox_inches="tight"
    )

    pdf.savefig(fig, bbox_inches="tight")

    plt.show()

# =============================================================================
# Save pairwise correlation results
# =============================================================================

results_df = pd.DataFrame(results)

stats_file = os.path.join(
    save_dir,
    f"{file_stem}_pairwise_correlation_results.csv"
)

results_df.to_csv(stats_file, index=False)

# =============================================================================
# Summary
# =============================================================================

print(f"\nPairwise correlation table saved:\n{stats_file}")
print(f"\nPearson matrix saved:\n{pearson_matrix_file}")
print(f"\nHeatmap saved:\n{heatmap_file}")
print(f"\nPDF with all figures saved:\n{pdf_file}")

print("\nAnalysis complete.")
#"color": "black"
#"color": "red"
#"color": "darkred"
#"color": "blue"
#"color": "navy"
#"color": "green"
#"color": "darkgreen"
#"color": "purple"
#"color": "orange"
#Use publication-quality hex colors
#"color": "#1F77B4"   # blue
#"color": "#D62728"   # red
#"color": "#2CA02C"   # green
#"color": "#9467BD"   # purple
#"color": "#000000"   # black

