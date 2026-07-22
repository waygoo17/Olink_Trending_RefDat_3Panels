import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================
INPUT_CSV = r"C:\Users\wei.guo2\Python\datasheet\3Panels_Ref_MasterSheet_17Jul2026.csv"
OUTPUT_DIR = Path(r"C:\Users\wei.guo2\Python\figures\olink_t48_results")


# ============================================================
# Helpers
# ============================================================
def ensure_outdir(path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_panel_version(x):
    """
    Normalize PanelVersion so sorting works well.
    Examples:
        "1"   -> 1
        "1.0" -> 1
        "2.5" -> 2.5
        "v3"  -> "v3"
    """
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
    """
    Sort numeric versions before text versions.
    """
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


def plot_log2_fold_change_heatmap(mat, title, outpath, figsize=(12, 10)):
    first_col = mat.columns[0]
    base = mat[first_col].astype(float)
    fc = mat.astype(float).divide(base, axis=0)
    log2fc = np.log2(fc)

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
    first_col = mat.columns[0]
    base = mat[first_col].astype(float)
    pct = (mat.astype(float).divide(base, axis=0) - 1.0) * 100.0

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


def plot_boxplot(df_panel, value_col, title, outpath, show_points=False):
    versions = parse_panel_versions(df_panel["PanelVersion"])
    data = [df_panel.loc[df_panel["PanelVersion"] == v, value_col].dropna().values for v in versions]

    plt.figure(figsize=(max(10, 0.6 * len(versions)), 5))
    positions = np.arange(1, len(versions) + 1)

    plt.boxplot(data, positions=positions, labels=[version_str(v) for v in versions], showfliers=False)

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

    plt.figure(figsize=figsize)
    for assay in mat.index:
        y = mat.loc[assay].astype(float).values
        if yscale == "log10":
            y = np.log10(y)
            ylabel = title.split(" - ")[-1] + " (log10 scale)"
        else:
            ylabel = title.split(" - ")[-1]

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

    # Summary tables
    lloq_summary = summarize_version_stats(df_panel, "LLOQ")
    uloq_summary = summarize_version_stats(df_panel, "ULOQ")
    write_csv(lloq_summary, panel_dir / f"{slug}_lloq_version_summary.csv")
    write_csv(uloq_summary, panel_dir / f"{slug}_uloq_version_summary.csv")

    # Matrices
    lloq_mat = make_matrix(df_panel, "LLOQ", version_order)
    uloq_mat = make_matrix(df_panel, "ULOQ", version_order)

    # Heatmaps
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

    # Log2 fold-change heatmaps
    plot_log2_fold_change_heatmap(
        lloq_mat,
        title=f"{panel_name} - LLOQ log2 fold-change vs first version",
        outpath=panel_dir / f"{slug}_lloq_foldchange_heatmap.png",
        figsize=(12, max(8, 0.22 * len(lloq_mat.index))),
    )
    plot_log2_fold_change_heatmap(
        uloq_mat,
        title=f"{panel_name} - ULOQ log2 fold-change vs first version",
        outpath=panel_dir / f"{slug}_uloq_foldchange_heatmap.png",
        figsize=(12, max(8, 0.22 * len(uloq_mat.index))),
    )

    # Percent-change heatmaps
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

    # Top changes
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
        f.write("\nLLOQ version summary:\n")
        f.write(lloq_summary.to_string(index=False))
        f.write("\n\nULOQ version summary:\n")
        f.write(uloq_summary.to_string(index=False))
        f.write("\n")


# ============================================================
# Main
# ============================================================
def main():
    inpath = Path(INPUT_CSV)
    ensure_outdir(OUTPUT_DIR)

    if not inpath.exists():
        raise FileNotFoundError(
            f"Input CSV not found:\n{inpath}\n\n"
            "Check INPUT_CSV at the top of the script and make sure the file name is correct."
        )

    df = pd.read_csv(inpath)

    required = {"Panel", "PanelVersion", "Assay", "LLOQ", "ULOQ"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Clean data
    df = df.copy()
    df["Panel"] = df["Panel"].astype(str)
    df["Assay"] = df["Assay"].astype(str)
    df["PanelVersion"] = df["PanelVersion"].apply(normalize_panel_version)
    df["LLOQ"] = safe_numeric(df["LLOQ"])
    df["ULOQ"] = safe_numeric(df["ULOQ"])
    df = df.dropna(subset=["LLOQ", "ULOQ"])

    # Overall summary across panel-version groups
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
    overall_summary.to_csv(OUTPUT_DIR / "overall_version_summary.csv", index=False)

    # Per-panel analysis
    for panel_name, df_panel in df.groupby("Panel", sort=False):
        analyze_panel(df_panel, panel_name, OUTPUT_DIR)

    # Output manifest
    files = []
    for root, _, filenames in os.walk(OUTPUT_DIR):
        for fn in filenames:
            files.append(os.path.relpath(os.path.join(root, fn), OUTPUT_DIR))

    pd.DataFrame({"output_files": sorted(files)}).to_csv(OUTPUT_DIR / "output_manifest.csv", index=False)

    print(f"Done. Results written to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()