
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%
data_file = r"C:\Users\wei.guo2\Python\datasheet\ref_T48Cytokine_formatted_clean.csv"
save_dir = r"C:\Users\wei.guo2\Python\Figure"
save_file = os.path.join(save_dir, "NPX_by_SampleID_boxplot.png")

os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()

# %%
fig, ax = plt.subplots(figsize=(14, 6))

sns.boxplot(data=df, x="SampleID", y="NPX", color="#7B1FA2", width=0.6, showfliers=False, ax=ax)
sns.stripplot(data=df, x="SampleID", y="NPX", color="black", size=2.5, alpha=1, jitter=True, ax=ax)

ax.set_xlabel("SampleID")
ax.set_ylabel("NPX")
ax.set_title("NPX by SampleID")
ax.tick_params(axis="x", labelrotation=45)

plt.tight_layout()
plt.savefig(save_file, dpi=300, bbox_inches="tight")

plt.show()








