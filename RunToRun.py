import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\Run1-3.csv"
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
        temp = temp[temp["N"] >= 3].copy()

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

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)

# =============================================================================
# Combined plot: all runs together
# =============================================================================

def plot_cv_by_run_and_range(data, range_col, range_order, x_label, value_type, save_prefix):
    fig_file = os.path.join(
        save_dir,
        f"{file_stem}_{save_prefix}_CV_by_run_and_mean_range.png"
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.boxplot(
        data=data,
        x=range_col,
        y="CV%",
        hue="Run",
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
        order=range_order,
        dodge=True,
        color="#7B1FA2",
        size=4,
        alpha=0.75,
        jitter=0.18,
        ax=ax
    )

    format_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
        title="Run",
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("CV%")
    ax.set_title(f"{file_stem} - {value_type} CV% per Target by Run")

    plt.tight_layout()
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Combined figure saved to:\n{fig_file}")

# =============================================================================
# Separate plot: one figure per run
# =============================================================================

def plot_cv_by_range_for_each_run(data, range_col, range_order, x_label, value_type, save_prefix):
    for run in ["Run1", "Run2", "Run3"]:

        run_df = data[data["Run"] == run].copy()

        if run_df.empty:
            print(f"Skipping {run}: no data")
            continue

        fig_file = os.path.join(
            save_dir,
            f"{file_stem}_{save_prefix}_CV_by_mean_range_{run}.png"
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        sns.boxplot(
            data=run_df,
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
            color="#7B1FA2",
            size=4,
            alpha=0.75,
            jitter=0.18,
            ax=ax
        )

        format_axis(ax)

        ax.legend(frameon=False)

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

