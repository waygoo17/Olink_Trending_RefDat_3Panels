import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Input / Output
# =============================================================================

data_file = r"C:\Users\wei.guo2\Python\datasheet\QOFG_T48Immune_re_vitreouhumor_30Jun2026_Extended_2026-06-30.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure"

os.makedirs(save_dir, exist_ok=True)

# =============================================================================
# File names
# =============================================================================

folder = os.path.dirname(data_file)
base_name = os.path.basename(data_file)
file_stem, file_ext = os.path.splitext(base_name)

clean_file = os.path.join(folder, f"{file_stem}_clean{file_ext}")
save_file = os.path.join(save_dir, f"{file_stem}_NPX_by_SampleID_boxplot.png")

# =============================================================================
# Create clean CSV
# =============================================================================

df_clean = pd.read_csv(data_file, sep=";")
df_clean.columns = df_clean.columns.str.strip()

df_clean.to_csv(clean_file, index=False)

print(f"Clean CSV saved to:\n{clean_file}")
# =============================================================================
# Use clean CSV for plotting
# =============================================================================

df = pd.read_csv(clean_file)
df.columns = df.columns.str.strip()

df["NPX"] = pd.to_numeric(df["NPX"], errors="coerce")
df = df.dropna(subset=["SampleID", "NPX"])

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 6))

sns.boxplot(
    data=df,
    x="SampleID",
    y="NPX",
    color="#7B1FA2",
    width=0.6,
    showfliers=False,
    ax=ax
)

sns.stripplot(
    data=df,
    x="SampleID",
    y="NPX",
    color="black",
    size=2.5,
    alpha=0.8,
    jitter=True,
    ax=ax
)

ax.set_xlabel("SampleID")
ax.set_ylabel("NPX")
ax.set_title(f"{file_stem} - NPX by SampleID")
ax.tick_params(axis="x", labelrotation=45)

plt.tight_layout()
plt.savefig(save_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to:\n{save_file}")


