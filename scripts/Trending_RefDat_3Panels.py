import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ============================================================
# Default paths
# ============================================================
_HERE = Path(__file__).parent

CSV_NAME = "3Panels_Ref_MasterSheet_17Jul2026.csv"

# Search locations
SEARCH_LOCATIONS = [
    _HERE,
    _HERE / "datasheet",
    _HERE.parent,
    _HERE.parent / "datasheet",
]

_DEFAULT_INPUT = None

for folder in SEARCH_LOCATIONS:
    candidate = folder / CSV_NAME
    if candidate.exists():
        _DEFAULT_INPUT = candidate
        break

if _DEFAULT_INPUT is None:
    _DEFAULT_INPUT = SEARCH_LOCATIONS[0] / CSV_NAME

_DEFAULT_OUTPUT = _HERE / "Trending_3panels_Output"


# Tolerance settings for adjacent-version plots
ADJ_LOG2_ZERO_TOL = 1e-4   # values smaller than this in abs(log2 change) become 0
ADJ_PCT_ZERO_TOL = 0.1     # values smaller than this in abs(percent change) become 0
ADJ_LOG2_CLIP = 0.10       # fixed color scale for filtered log2 adjacent plots
ADJ_PCT_CLIP = 10.0        # fixed color scale for filtered percent adjacent plots

# ============================================================
# Helpers
# ============================================================
def ensure_outdir(path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_panel_version(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    try:
        f = float(s)
        if f.is_integer():
            return int(f)
        return f
    except ValueError:
        return s


def version_sort_key(v):
    if pd.isna(v):
        return (2, "")
    if isinstance(v, (int, np.integer)):
        return (0, float(v))
    if isinstance(v, (float, np.floating)):
        return (0, float(v))

    s = str(v)
    try:
        return (0, float(s))
    except Exception:
        return (1, s)


def version_str(v):
    if pd.isna(v):
        return "NA"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer():
            return str(int(v))
        return str(v)
    return str(v)


def panel_slug(panel_name):
    return (
        str(panel_name)
        .lower()
        .replace("olink target 48 ", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def parse_panel_versions(series):
    vals = series.dropna().unique().tolist()
    return sorted(vals, key=version_sort_key)


def savefig(path, dpi=200):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def make_matrix(df, value_col, version_order):
    mat = df.pivot_table(index="Assay", columns="PanelVersion", values=value_col, aggfunc="mean")
    mat = mat.reindex(columns=version_order)
    mat = mat.sort_index()
    return mat


def summarize_version_stats(df_panel, value_col):
    return (
        df_panel.groupby("PanelVersion")[value_col]
        .agg(["count", "median", "mean", "std", "min", "max"])
        .reset_index()
    )


def fold_change_matrix(mat):
    first_col = mat.columns[0]
    return mat.astype(float).divide(mat[first_col].astype(float), axis=0)


def log2_fold_change_matrix(mat):
    fc = fold_change_matrix(mat)
    return np.log2(fc)


def percent_change_matrix(mat):
    fc = fold_change_matrix(mat)
    return (fc - 1.0) * 100.0


def adjacent_log2_change_matrix(mat):
    out = pd.DataFrame(index=mat.index, columns=mat.columns, dtype=float)
    out.iloc[:, 0] = np.nan
    for i in range(1, len(mat.columns)):
        c0 = mat.columns[i - 1]
        c1 = mat.columns[i]
        out[c1] = np.log2(mat[c1].astype(float) / mat[c0].astype(float))
    return out


def adjacent_percent_change_matrix(mat):
    out = pd.DataFrame(index=mat.index, columns=mat.columns, dtype=float)
    out.iloc[:, 0] = np.nan
    for i in range(1, len(mat.columns)):
        c0 = mat.columns[i - 1]
        c1 = mat.columns[i]
        out[c1] = (mat[c1].astype(float) / mat[c0].astype(float) - 1.0) * 100.0
    return out


def apply_zero_tolerance(mat, zero_tol):
    out = mat.copy().astype(float)
    out[np.abs(out) < zero_tol] = 0.0
    return out


def ordered_cluster_rows(data):
    X = data.copy().astype(float)

    valid = np.isfinite(X.values).any(axis=1)
    X = X.loc[valid]

    if X.empty:
        return data.index.tolist()

    arr = X.to_numpy(copy=True)

    for i in range(arr.shape[0]):
        row = arr[i, :].copy()
        mask = np.isfinite(row)
        if not mask.any():
            row[:] = 0.0
        else:
            fill_value = np.nanmean(row[mask])
            row[~mask] = fill_value
        arr[i, :] = row

    row_means = np.mean(arr, axis=1, keepdims=True)
    row_stds = np.std(arr, axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1.0
    z = (arr - row_means) / row_stds

    if SCIPY_AVAILABLE and z.shape[0] > 2:
        try:
            d = pdist(z, metric="euclidean")
            Z = linkage(d, method="average")
            order = leaves_list(Z)
            return X.index[order].tolist()
        except Exception:
            pass

    scores = np.nansum(np.abs(arr), axis=1)
    return X.index[np.argsort(scores)].tolist()


def plot_heatmap(mat, title, cbar_label, outpath, use_log10=True, figsize=(12, 10)):
    plt.figure(figsize=figsize)

    data = mat.copy().astype(float)
    if use_log10:
        data = np.log10(data)

    arr = np.ma.masked_invalid(data.values.astype(float))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest")

    plt.xticks(range(len(data.columns)), [version_str(v) for v in data.columns], rotation=90)
    plt.yticks(range(len(data.index)), data.index)
    plt.xlabel("Panel Version")
    plt.ylabel("Assay")
    plt.title(title)

    cb = plt.colorbar(im)
    cb.set_label(cbar_label)

    savefig(outpath)


def plot_fold_change_heatmap(mat, title, outpath, figsize=(12, 10)):
    log2fc = log2_fold_change_matrix(mat)

    plt.figure(figsize=figsize)
    arr = np.ma.masked_invalid(log2fc.values.astype(float))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-3, vmax=3)

    plt.xticks(range(len(log2fc.columns)), [version_str(v) for v in log2fc.columns], rotation=90)
    plt.yticks(range(len(log2fc.index)), log2fc.index)
    plt.xlabel("Panel Version")
    plt.ylabel("Assay")
    plt.title(title)

    cb = plt.colorbar(im)
    cb.set_label("log2 fold-change vs first version")

    savefig(outpath)


def plot_percent_change_heatmap(mat, title, outpath, figsize=(12, 10)):
    pct = percent_change_matrix(mat)

    values = pct.values.astype(float)
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size > 0:
        vmax = np.nanpercentile(np.abs(finite_vals), 95)
        vmax = max(vmax, 50.0)
    else:
        vmax = 100.0

    plt.figure(figsize=figsize)
    arr = np.ma.masked_invalid(values)
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    plt.xticks(range(len(pct.columns)), [version_str(v) for v in pct.columns], rotation=90)
    plt.yticks(range(len(pct.index)), pct.index)
    plt.xlabel("Panel Version")
    plt.ylabel("Assay")
    plt.title(title)

    cb = plt.colorbar(im)
    cb.set_label("% change vs first version")

    savefig(outpath)


def plot_clustered_heatmap(data_mat, title, cbar_label, outpath, cmap="coolwarm", use_log10=False, figsize=(12, 10)):
    data = data_mat.copy().astype(float)

    if use_log10:
        data = np.log10(data)

    row_order = ordered_cluster_rows(data)
    data = data.loc[row_order]

    values = data.values.astype(float)
    finite_vals = values[np.isfinite(values)]

    if finite_vals.size > 0 and cmap == "coolwarm":
        vmax = np.nanpercentile(np.abs(finite_vals), 95)
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = None
        vmax = None

    plt.figure(figsize=figsize)
    arr = np.ma.masked_invalid(values)
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    plt.xticks(range(len(data.columns)), [version_str(v) for v in data.columns], rotation=90)
    plt.yticks(range(len(data.index)), data.index)
    plt.xlabel("Panel Version")
    plt.ylabel("Assay")
    plt.title(title)

    cb = plt.colorbar(im)
    cb.set_label(cbar_label)

    savefig(outpath)

def plot_boxplot(df_panel, value_col, title, outpath, show_points=False):
    versions = parse_panel_versions(df_panel["PanelVersion"])
    data = [df_panel.loc[df_panel["PanelVersion"] == v, value_col].dropna().values for v in versions]

    plt.figure(figsize=(max(10, 0.6 * len(versions)), 5))
    positions = np.arange(1, len(versions) + 1)

    plt.boxplot(
    data,
    positions=positions,
    tick_labels=[version_str(v) for v in versions],
    showfliers=False,
)
    

    if show_points:
        rng = np.random.default_rng(42)
        for i, v in enumerate(versions, start=1):
            vals = df_panel.loc[df_panel["PanelVersion"] == v, value_col].dropna().values
            if len(vals) == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            x = np.full(len(vals), i) + jitter
            plt.scatter(x, vals, s=14, alpha=0.45)

    plt.yscale("log")
    plt.xticks(rotation=90)
    plt.ylabel(value_col)
    plt.title(title)

    savefig(outpath)


def plot_assay_lines(mat, title, outpath, yscale="log10", figsize=(12, 8)):
    versions = mat.columns.tolist()

    if yscale == "log10":
        ylabel = title.split(" - ")[-1] + " (log10 scale)"
    else:
        ylabel = title.split(" - ")[-1]

    plt.figure(figsize=figsize)
    for assay in mat.index:
        y = mat.loc[assay].astype(float).values
        if yscale == "log10":
            y = np.log10(y)
        plt.plot(range(len(versions)), y, alpha=0.25, linewidth=1)

    plt.xticks(range(len(versions)), [version_str(v) for v in versions], rotation=90)
    plt.xlabel("Panel Version")
    plt.ylabel(ylabel)
    plt.title(title)

    savefig(outpath)


def plot_dynamic_range(df_panel, outpath):
    versions = parse_panel_versions(df_panel["PanelVersion"])
    rows = []

    for v in versions:
        tmp = df_panel[df_panel["PanelVersion"] == v].copy()
        tmp["DynamicRange"] = tmp["ULOQ"] / tmp["LLOQ"]
        rows.append(
            {
                "PanelVersion": v,
                "median": np.nanmedian(tmp["DynamicRange"].values),
                "mean": np.nanmean(tmp["DynamicRange"].values),
            }
        )

    summary = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(summary)), np.log10(summary["median"].astype(float).values), marker="o", label="Median")
    plt.plot(range(len(summary)), np.log10(summary["mean"].astype(float).values), marker="o", label="Mean")
    plt.xticks(range(len(summary)), [version_str(v) for v in summary["PanelVersion"]], rotation=90)
    plt.xlabel("Panel Version")
    plt.ylabel("log10(ULOQ / LLOQ)")
    plt.title("Dynamic range over versions")
    plt.legend()

    savefig(outpath)


def top_change_table(mat, metric_name, n=15):
    first_col = mat.columns[0]
    last_col = mat.columns[-1]

    first = mat[first_col].astype(float)
    last = mat[last_col].astype(float)

    out = pd.DataFrame(
        {
            "Assay": mat.index,
            f"{metric_name}_first": first.values,
            f"{metric_name}_last": last.values,
        }
    )

    out["fold_change"] = out[f"{metric_name}_last"] / out[f"{metric_name}_first"]
    out["log2_fold_change"] = np.log2(out["fold_change"])
    out["abs_log2_fold_change"] = out["log2_fold_change"].abs()
    out = out.sort_values("abs_log2_fold_change", ascending=False).head(n)

    return out


def plot_volcano(df_panel, outpath):
    versions = parse_panel_versions(df_panel["PanelVersion"])
    first_v = versions[0]
    last_v = versions[-1]

    base = (
        df_panel[df_panel["PanelVersion"] == first_v]
        .set_index("Assay")[["LLOQ", "ULOQ"]]
        .rename(columns={"LLOQ": "LLOQ_first", "ULOQ": "ULOQ_first"})
    )
    last = (
        df_panel[df_panel["PanelVersion"] == last_v]
        .set_index("Assay")[["LLOQ", "ULOQ"]]
        .rename(columns={"LLOQ": "LLOQ_last", "ULOQ": "ULOQ_last"})
    )

    merged = base.join(last, how="inner")
    merged["x_log2_ULOQ"] = np.log2(merged["ULOQ_last"] / merged["ULOQ_first"])
    merged["y_log2_LLOQ"] = np.log2(merged["LLOQ_last"] / merged["LLOQ_first"])

    plt.figure(figsize=(7, 7))
    plt.scatter(merged["x_log2_ULOQ"], merged["y_log2_LLOQ"], alpha=0.7)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.xlabel(f"log2(ULOQ {version_str(last_v)} / {version_str(first_v)})")
    plt.ylabel(f"log2(LLOQ {version_str(last_v)} / {version_str(first_v)})")
    plt.title("First vs last version change")

    savefig(outpath)

    return merged.reset_index()


def write_csv(df, path):
    df.to_csv(path, index=False)


def plot_panel_trend_summary(log2_mat, pct_mat, panel_name, panel_dir, slug, metric_name=""):
    suffix = f"_{metric_name.lower()}" if metric_name else ""
    versions = [version_str(v) for v in log2_mat.columns]
    x = np.arange(len(versions))

    def summarize(mat):
        med = np.nanmedian(mat.values.astype(float), axis=0)
        q25 = np.nanpercentile(mat.values.astype(float), 25, axis=0)
        q75 = np.nanpercentile(mat.values.astype(float), 75, axis=0)
        mean = np.nanmean(mat.values.astype(float), axis=0)
        return med, mean, q25, q75

    log2_med, log2_mean, log2_q25, log2_q75 = summarize(log2_mat)
    pct_med, pct_mean, pct_q25, pct_q75 = summarize(pct_mat)

    plt.figure(figsize=(12, 5))
    plt.plot(x, log2_med, marker="o", label="Median")
    plt.plot(x, log2_mean, marker="o", label="Mean")
    plt.fill_between(x, log2_q25, log2_q75, alpha=0.2)
    plt.axhline(0, linewidth=1, color="black")
    plt.xticks(x, versions, rotation=90)
    plt.ylabel("log2 fold-change vs first version")
    plt.xlabel("Panel Version")
    plt.title(f"{panel_name} - panel-level log2 change summary")
    plt.legend()
    savefig(panel_dir / f"{slug}{suffix}_panel_level_log2_summary.png")

    plt.figure(figsize=(12, 5))
    plt.plot(x, pct_med, marker="o", label="Median")
    plt.plot(x, pct_mean, marker="o", label="Mean")
    plt.fill_between(x, pct_q25, pct_q75, alpha=0.2)
    plt.axhline(0, linewidth=1, color="black")
    plt.xticks(x, versions, rotation=90)
    plt.ylabel("% change vs first version")
    plt.xlabel("Panel Version")
    plt.title(f"{panel_name} - panel-level percent change summary")
    plt.legend()
    savefig(panel_dir / f"{slug}{suffix}_panel_level_percent_summary.png")


def plot_cumulative_changed_assays(pct_mat, panel_name, panel_dir, slug, metric_name):
    versions = [version_str(v) for v in pct_mat.columns]
    thresholds = [10, 20, 50, 100]

    plt.figure(figsize=(12, 5))
    x = np.arange(len(versions))
    arr = pct_mat.values.astype(float)

    for thr in thresholds:
        counts = np.array([np.sum(np.abs(arr[:, i]) >= thr) for i in range(arr.shape[1])])
        plt.plot(x, counts, marker="o", label=f"|% change| >= {thr}%")

    plt.xticks(x, versions, rotation=90)
    plt.xlabel("Panel Version")
    plt.ylabel("Number of assays")
    plt.title(f"{panel_name} - cumulative changed assays ({metric_name})")
    plt.legend()
    savefig(panel_dir / f"{slug}_{metric_name.lower()}_cumulative_changed_assays.png")


def plot_adjacent_change_heatmap(adj_mat, title, outpath, label, vlim=None, figsize=(12, 10)):
    data = adj_mat.copy().astype(float)

    if vlim is None:
        finite_vals = data.values[np.isfinite(data.values)]
        if finite_vals.size > 0:
            vmax = np.nanpercentile(np.abs(finite_vals), 95)
            if vmax == 0:
                vmax = 1.0
        else:
            vmax = 1.0
    else:
        vmax = float(vlim)

    plt.figure(figsize=figsize)
    arr = np.ma.masked_invalid(data.values.astype(float))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    plt.xticks(range(len(data.columns)), [version_str(v) for v in data.columns], rotation=90)
    plt.yticks(range(len(data.index)), data.index)
    plt.xlabel("Panel Version")
    plt.ylabel("Assay")
    plt.title(title)

    cb = plt.colorbar(im)
    cb.set_label(label)

    savefig(outpath)


def plot_top_trajectory_assays(df_panel, value_col, panel_name, panel_dir, slug, metric_name, n=12):
    versions = parse_panel_versions(df_panel["PanelVersion"])
    mat = make_matrix(df_panel, value_col, versions)

    log2fc = log2_fold_change_matrix(mat)
    scores = log2fc.abs().max(axis=1).sort_values(ascending=False)
    top_assays = scores.head(n).index.tolist()

    plt.figure(figsize=(12, 7))
    for assay in top_assays:
        y = mat.loc[assay].astype(float).values
        plt.plot(range(len(versions)), np.log10(y), marker="o", linewidth=2, label=assay)

    plt.xticks(range(len(versions)), [version_str(v) for v in versions], rotation=90)
    plt.xlabel("Panel Version")
    plt.ylabel(f"log10({value_col})")
    plt.title(f"{panel_name} - top {n} assay trajectories ({metric_name})")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    savefig(panel_dir / f"{slug}_{metric_name.lower()}_top_trajectories.png")

    ncols = 3
    nrows = int(np.ceil(len(top_assays) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)

    for idx, assay in enumerate(top_assays):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        y = mat.loc[assay].astype(float).values
        ax.plot(range(len(versions)), np.log10(y), marker="o", linewidth=2)
        ax.set_xticks(range(len(versions)))
        ax.set_xticklabels([version_str(v) for v in versions], rotation=90)
        ax.set_title(assay)
        ax.set_xlabel("Panel Version")
        ax.set_ylabel(f"log10({value_col})")

    for idx in range(len(top_assays), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.suptitle(f"{panel_name} - top {n} assay trajectories ({metric_name})", y=1.02)
    plt.tight_layout()
    fig.savefig(panel_dir / f"{slug}_{metric_name.lower()}_top_trajectories_small_multiples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Per-panel analysis
# ============================================================
def analyze_panel(df_panel, panel_name, outdir):
    slug = panel_slug(panel_name)
    panel_dir = outdir / slug
    ensure_outdir(panel_dir)

    version_order = parse_panel_versions(df_panel["PanelVersion"])

    df_panel = df_panel.copy()
    df_panel["LLOQ"] = safe_numeric(df_panel["LLOQ"])
    df_panel["ULOQ"] = safe_numeric(df_panel["ULOQ"])
    df_panel = df_panel.dropna(subset=["LLOQ", "ULOQ"])

    lloq_summary = summarize_version_stats(df_panel, "LLOQ")
    uloq_summary = summarize_version_stats(df_panel, "ULOQ")
    write_csv(lloq_summary, panel_dir / f"{slug}_lloq_version_summary.csv")
    write_csv(uloq_summary, panel_dir / f"{slug}_uloq_version_summary.csv")

    lloq_mat = make_matrix(df_panel, "LLOQ", version_order)
    uloq_mat = make_matrix(df_panel, "ULOQ", version_order)

    lloq_log2 = log2_fold_change_matrix(lloq_mat)
    uloq_log2 = log2_fold_change_matrix(uloq_mat)
    lloq_pct = percent_change_matrix(lloq_mat)
    uloq_pct = percent_change_matrix(uloq_mat)

    lloq_adj_log2_raw = adjacent_log2_change_matrix(lloq_mat)
    uloq_adj_log2_raw = adjacent_log2_change_matrix(uloq_mat)
    lloq_adj_pct_raw = adjacent_percent_change_matrix(lloq_mat)
    uloq_adj_pct_raw = adjacent_percent_change_matrix(uloq_mat)

    lloq_adj_log2_filt = apply_zero_tolerance(lloq_adj_log2_raw, ADJ_LOG2_ZERO_TOL)
    uloq_adj_log2_filt = apply_zero_tolerance(uloq_adj_log2_raw, ADJ_LOG2_ZERO_TOL)
    lloq_adj_pct_filt = apply_zero_tolerance(lloq_adj_pct_raw, ADJ_PCT_ZERO_TOL)
    uloq_adj_pct_filt = apply_zero_tolerance(uloq_adj_pct_raw, ADJ_PCT_ZERO_TOL)

    # Existing raw all-target heatmaps
    plot_heatmap(
        lloq_mat,
        title=f"{panel_name} - LLOQ heatmap",
        cbar_label="log10(LLOQ)",
        outpath=panel_dir / f"{slug}_lloq_heatmap.png",
        use_log10=True,
        figsize=(12, max(8, 0.22 * len(lloq_mat.index))),
    )
    plot_heatmap(
        uloq_mat,
        title=f"{panel_name} - ULOQ heatmap",
        cbar_label="log10(ULOQ)",
        outpath=panel_dir / f"{slug}_uloq_heatmap.png",
        use_log10=True,
        figsize=(12, max(8, 0.22 * len(uloq_mat.index))),
    )

    # New clustered raw all-target heatmaps
    plot_clustered_heatmap(
        lloq_mat,
        title=f"{panel_name} - clustered all-target LLOQ heatmap",
        cbar_label="log10(LLOQ)",
        outpath=panel_dir / f"{slug}_lloq_all_targets_clustered_heatmap.png",
        cmap="viridis",
        use_log10=True,
        figsize=(12, max(8, 0.22 * len(lloq_mat.index))),
    )
    plot_clustered_heatmap(
        uloq_mat,
        title=f"{panel_name} - clustered all-target ULOQ heatmap",
        cbar_label="log10(ULOQ)",
        outpath=panel_dir / f"{slug}_uloq_all_targets_clustered_heatmap.png",
        cmap="viridis",
        use_log10=True,
        figsize=(12, max(8, 0.22 * len(uloq_mat.index))),
    )

    # Existing fold-change heatmaps
    plot_fold_change_heatmap(
        lloq_mat,
        title=f"{panel_name} - LLOQ log2 fold-change vs first version",
        outpath=panel_dir / f"{slug}_lloq_foldchange_heatmap.png",
        figsize=(12, max(8, 0.22 * len(lloq_mat.index))),
    )
    plot_fold_change_heatmap(
        uloq_mat,
        title=f"{panel_name} - ULOQ log2 fold-change vs first version",
        outpath=panel_dir / f"{slug}_uloq_foldchange_heatmap.png",
        figsize=(12, max(8, 0.22 * len(uloq_mat.index))),
    )

    # Existing percent-change heatmaps
    plot_percent_change_heatmap(
        lloq_mat,
        title=f"{panel_name} - LLOQ % change vs first version",
        outpath=panel_dir / f"{slug}_lloq_pctchange_heatmap.png",
        figsize=(12, max(8, 0.22 * len(lloq_mat.index))),
    )
    plot_percent_change_heatmap(
        uloq_mat,
        title=f"{panel_name} - ULOQ % change vs first version",
        outpath=panel_dir / f"{slug}_uloq_pctchange_heatmap.png",
        figsize=(12, max(8, 0.22 * len(uloq_mat.index))),
    )

    # Existing clustered change heatmaps
    plot_clustered_heatmap(
        lloq_log2,
        title=f"{panel_name} - clustered LLOQ log2 fold-change vs first version",
        cbar_label="log2 fold-change vs first version",
        outpath=panel_dir / f"{slug}_lloq_clustered_log2_heatmap.png",
        cmap="coolwarm",
        use_log10=False,
        figsize=(12, max(8, 0.22 * len(lloq_log2.index))),
    )
    plot_clustered_heatmap(
        uloq_log2,
        title=f"{panel_name} - clustered ULOQ log2 fold-change vs first version",
        cbar_label="log2 fold-change vs first version",
        outpath=panel_dir / f"{slug}_uloq_clustered_log2_heatmap.png",
        cmap="coolwarm",
        use_log10=False,
        figsize=(12, max(8, 0.22 * len(uloq_log2.index))),
    )
    plot_clustered_heatmap(
        lloq_pct,
        title=f"{panel_name} - clustered LLOQ % change vs first version",
        cbar_label="% change vs first version",
        outpath=panel_dir / f"{slug}_lloq_clustered_pct_heatmap.png",
        cmap="coolwarm",
        use_log10=False,
        figsize=(12, max(8, 0.22 * len(lloq_pct.index))),
    )
    plot_clustered_heatmap(
        uloq_pct,
        title=f"{panel_name} - clustered ULOQ % change vs first version",
        cbar_label="% change vs first version",
        outpath=panel_dir / f"{slug}_uloq_clustered_pct_heatmap.png",
        cmap="coolwarm",
        use_log10=False,
        figsize=(12, max(8, 0.22 * len(uloq_pct.index))),
    )

    # Panel-level summaries
    plot_panel_trend_summary(lloq_log2, lloq_pct, panel_name, panel_dir, slug, metric_name="LLOQ")
    plot_panel_trend_summary(uloq_log2, uloq_pct, panel_name, panel_dir, slug, metric_name="ULOQ")

    # Raw adjacent-version heatmaps
    plot_adjacent_change_heatmap(
        lloq_adj_log2_raw,
        title=f"{panel_name} - LLOQ adjacent-version log2 change",
        outpath=panel_dir / f"{slug}_lloq_adjacent_log2_heatmap.png",
        label="log2 change vs previous version",
        vlim=None,
        figsize=(12, max(8, 0.22 * len(lloq_adj_log2_raw.index))),
    )
    plot_adjacent_change_heatmap(
        uloq_adj_log2_raw,
        title=f"{panel_name} - ULOQ adjacent-version log2 change",
        outpath=panel_dir / f"{slug}_uloq_adjacent_log2_heatmap.png",
        label="log2 change vs previous version",
        vlim=None,
        figsize=(12, max(8, 0.22 * len(uloq_adj_log2_raw.index))),
    )
    plot_adjacent_change_heatmap(
        lloq_adj_pct_raw,
        title=f"{panel_name} - LLOQ adjacent-version % change",
        outpath=panel_dir / f"{slug}_lloq_adjacent_pct_heatmap.png",
        label="% change vs previous version",
        vlim=None,
        figsize=(12, max(8, 0.22 * len(lloq_adj_pct_raw.index))),
    )
    plot_adjacent_change_heatmap(
        uloq_adj_pct_raw,
        title=f"{panel_name} - ULOQ adjacent-version % change",
        outpath=panel_dir / f"{slug}_uloq_adjacent_pct_heatmap.png",
        label="% change vs previous version",
        vlim=None,
        figsize=(12, max(8, 0.22 * len(uloq_adj_pct_raw.index))),
    )

    # Filtered adjacent-version heatmaps with fixed symmetric color scales
    plot_adjacent_change_heatmap(
        lloq_adj_log2_filt,
        title=f"{panel_name} - LLOQ adjacent-version log2 change (filtered)",
        outpath=panel_dir / f"{slug}_lloq_adjacent_log2_heatmap_filtered.png",
        label="log2 change vs previous version",
        vlim=ADJ_LOG2_CLIP,
        figsize=(12, max(8, 0.22 * len(lloq_adj_log2_filt.index))),
    )
    plot_adjacent_change_heatmap(
        uloq_adj_log2_filt,
        title=f"{panel_name} - ULOQ adjacent-version log2 change (filtered)",
        outpath=panel_dir / f"{slug}_uloq_adjacent_log2_heatmap_filtered.png",
        label="log2 change vs previous version",
        vlim=ADJ_LOG2_CLIP,
        figsize=(12, max(8, 0.22 * len(uloq_adj_log2_filt.index))),
    )
    plot_adjacent_change_heatmap(
        lloq_adj_pct_filt,
        title=f"{panel_name} - LLOQ adjacent-version % change (filtered)",
        outpath=panel_dir / f"{slug}_lloq_adjacent_pct_heatmap_filtered.png",
        label="% change vs previous version",
        vlim=ADJ_PCT_CLIP,
        figsize=(12, max(8, 0.22 * len(lloq_adj_pct_filt.index))),
    )
    plot_adjacent_change_heatmap(
        uloq_adj_pct_filt,
        title=f"{panel_name} - ULOQ adjacent-version % change (filtered)",
        outpath=panel_dir / f"{slug}_uloq_adjacent_pct_heatmap_filtered.png",
        label="% change vs previous version",
        vlim=ADJ_PCT_CLIP,
        figsize=(12, max(8, 0.22 * len(uloq_adj_pct_filt.index))),
    )

    # Save adjacent matrices so you can inspect values directly
    write_csv(lloq_adj_log2_raw.reset_index(), panel_dir / f"{slug}_lloq_adjacent_log2_raw.csv")
    write_csv(uloq_adj_log2_raw.reset_index(), panel_dir / f"{slug}_uloq_adjacent_log2_raw.csv")
    write_csv(lloq_adj_pct_raw.reset_index(), panel_dir / f"{slug}_lloq_adjacent_pct_raw.csv")
    write_csv(uloq_adj_pct_raw.reset_index(), panel_dir / f"{slug}_uloq_adjacent_pct_raw.csv")
    write_csv(lloq_adj_log2_filt.reset_index(), panel_dir / f"{slug}_lloq_adjacent_log2_filtered.csv")
    write_csv(uloq_adj_log2_filt.reset_index(), panel_dir / f"{slug}_uloq_adjacent_log2_filtered.csv")
    write_csv(lloq_adj_pct_filt.reset_index(), panel_dir / f"{slug}_lloq_adjacent_pct_filtered.csv")
    write_csv(uloq_adj_pct_filt.reset_index(), panel_dir / f"{slug}_uloq_adjacent_pct_filtered.csv")

    # Boxplots
    plot_boxplot(
        df_panel,
        "LLOQ",
        title=f"{panel_name} - LLOQ distribution by version",
        outpath=panel_dir / f"{slug}_lloq_boxplot.png",
        show_points=False,
    )
    plot_boxplot(
        df_panel,
        "ULOQ",
        title=f"{panel_name} - ULOQ distribution by version",
        outpath=panel_dir / f"{slug}_uloq_boxplot.png",
        show_points=True,
    )

    # Assay-level line plots
    plot_assay_lines(
        lloq_mat,
        title=f"{panel_name} - LLOQ across versions",
        outpath=panel_dir / f"{slug}_lloq_assay_lines.png",
        yscale="log10",
        figsize=(12, max(8, 0.14 * len(lloq_mat.index))),
    )
    plot_assay_lines(
        uloq_mat,
        title=f"{panel_name} - ULOQ across versions",
        outpath=panel_dir / f"{slug}_uloq_assay_lines.png",
        yscale="log10",
        figsize=(12, max(8, 0.14 * len(uloq_mat.index))),
    )

    # Dynamic range
    plot_dynamic_range(
        df_panel,
        outpath=panel_dir / f"{slug}_dynamic_range.png",
    )

    # Top changes tables
    top_lloq = top_change_table(lloq_mat, "LLOQ", n=15)
    top_uloq = top_change_table(uloq_mat, "ULOQ", n=15)
    write_csv(top_lloq, panel_dir / f"{slug}_top_changes_lloq.csv")
    write_csv(top_uloq, panel_dir / f"{slug}_top_changes_uloq.csv")

    # Volcano plot
    volcano = plot_volcano(
        df_panel,
        outpath=panel_dir / f"{slug}_first_vs_last_volcano.png",
    )
    write_csv(volcano, panel_dir / f"{slug}_first_vs_last_volcano_data.csv")

    # Cumulative changed assay plots
    plot_cumulative_changed_assays(
        lloq_pct,
        panel_name,
        panel_dir,
        slug,
        metric_name="LLOQ",
    )
    plot_cumulative_changed_assays(
        uloq_pct,
        panel_name,
        panel_dir,
        slug,
        metric_name="ULOQ",
    )

    # Top trajectory plots
    plot_top_trajectory_assays(
        df_panel,
        value_col="LLOQ",
        panel_name=panel_name,
        panel_dir=panel_dir,
        slug=slug,
        metric_name="LLOQ",
        n=12,
    )
    plot_top_trajectory_assays(
        df_panel,
        value_col="ULOQ",
        panel_name=panel_name,
        panel_dir=panel_dir,
        slug=slug,
        metric_name="ULOQ",
        n=12,
    )

    # Tidy derived metrics
    tidy = df_panel.copy()
    tidy["DynamicRange"] = tidy["ULOQ"] / tidy["LLOQ"]
    tidy["log10_LLOQ"] = np.log10(tidy["LLOQ"])
    tidy["log10_ULOQ"] = np.log10(tidy["ULOQ"])
    tidy["log10_DynamicRange"] = np.log10(tidy["DynamicRange"])
    write_csv(tidy, panel_dir / f"{slug}_tidy_with_derived_metrics.csv")

    # Text summary
    summary_txt = panel_dir / f"{slug}_analysis_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Panel: {panel_name}\n")
        f.write(f"Rows: {len(df_panel)}\n")
        f.write(f"Unique assays: {df_panel['Assay'].nunique()}\n")
        f.write(f"Versions: {len(version_order)}\n")
        f.write(f"Version order: {', '.join(map(version_str, version_order))}\n")
        f.write(f"Scipy available for clustering: {SCIPY_AVAILABLE}\n")
        f.write(f"Adjacent log2 zero tolerance: {ADJ_LOG2_ZERO_TOL}\n")
        f.write(f"Adjacent percent zero tolerance: {ADJ_PCT_ZERO_TOL}\n")
        f.write(f"Adjacent log2 fixed scale: +/-{ADJ_LOG2_CLIP}\n")
        f.write(f"Adjacent percent fixed scale: +/-{ADJ_PCT_CLIP}\n")
        f.write("\nLLOQ version summary:\n")
        f.write(lloq_summary.to_string(index=False))
        f.write("\n\nULOQ version summary:\n")
        f.write(uloq_summary.to_string(index=False))
        f.write("\n")


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Trending analysis for 3-panel Olink data.")
    parser.add_argument("--input",  type=Path, default=_DEFAULT_INPUT,  help="Path to input CSV (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Directory for output files (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    inpath = args.input
    output_dir = args.output
    ensure_outdir(output_dir)

    if not inpath.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{inpath}\n\n"
            "Pass the correct path with --input or update _DEFAULT_INPUT at the top of the script."
        )

    df = pd.read_csv(inpath)

    required = {"Panel", "PanelVersion", "Assay", "LLOQ", "ULOQ"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["Panel"] = df["Panel"].astype(str)
    df["Assay"] = df["Assay"].astype(str)
    df["PanelVersion"] = df["PanelVersion"].apply(normalize_panel_version)
    df["LLOQ"] = safe_numeric(df["LLOQ"])
    df["ULOQ"] = safe_numeric(df["ULOQ"])
    df = df.dropna(subset=["LLOQ", "ULOQ"])

    overall_summary = (
        df.groupby(["Panel", "PanelVersion"])
        .agg(
            n_assays=("Assay", "nunique"),
            n_rows=("Assay", "size"),
            LLOQ_median=("LLOQ", "median"),
            ULOQ_median=("ULOQ", "median"),
            LLOQ_mean=("LLOQ", "mean"),
            ULOQ_mean=("ULOQ", "mean"),
        )
        .reset_index()
    )
    overall_summary.to_csv(output_dir / "overall_version_summary.csv", index=False)

    for panel_name, df_panel in df.groupby("Panel", sort=False):
        analyze_panel(df_panel, panel_name, output_dir)

    files = []
    for root, _, filenames in os.walk(output_dir):
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(root, fn), output_dir))

    pd.DataFrame({"output_files": sorted(files)}).to_csv(output_dir / "output_manifest.csv", index=False)

    print(f"Done. Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
