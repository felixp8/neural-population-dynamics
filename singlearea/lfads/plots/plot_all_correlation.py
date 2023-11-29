import numpy as np
import pandas as pd
import itertools
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_STR = "murray_rec_pfc"
RUN_TAG = "231120_lfads_frozen"

RESULT_DIR = Path("../results") / DATASET_STR / RUN_TAG
PLOT_DIR = Path("./") / DATASET_STR
df_list = []
for result_file in sorted(RESULT_DIR.glob("*.csv")):
    df = pd.read_csv(result_file)
    df.set_index("name", inplace=True)
    df_list.append(df)
df = pd.concat(df_list, axis=1, ignore_index=False)
df.reset_index(drop=False, inplace=True)
df = df.loc[:,~df.columns.duplicated()]

print(f"dropping {(df['rate_r2'] <= 0.5).sum()} models")
df = df[df["rate_r2"] > 0.5]

# group_fields = ["model", "model.gen_dim"]
# group_fields = ["model", "model.gen_cell.rank"]
group_fields = ["model.co_dim", "model.gen_dim"]
# group_fields = ["model.co_dim", "model.gen_cell.rank"]
unique_groups = sorted(df.set_index(group_fields).index.unique().tolist())
unique_colors = plt.cm.hsv(np.arange(0, 1, 1/len(unique_groups)))
unique_labels = ["_".join([str(g) for g in group]) for group in unique_groups]

group_idx = np.full((df.shape[0],), np.nan)
for i, group_vals in enumerate(unique_groups):
    group_idx[np.all(df[group_fields] == group_vals, axis=1)] = i
assert not np.any(np.isnan(group_idx))
group_idx = group_idx.astype(int)

metrics = [
    "rate_r2",
    "fwd_latent_r2",
    "bwd_latent_r2",
    "fwd_input_r2",
    "bwd_input_r2",
    "latent_procrustes_rot",
    "input_procrustes_rot"
]
prefix = "st_"
metrics = [(prefix + m) if (prefix + m) in df.columns else m for m in metrics]

fig, axs = plt.subplots(len(metrics), len(metrics), figsize=(len(metrics)*4,len(metrics)*4))

for i, field1 in enumerate(metrics):
    axs[i][0].set_ylabel(field1)
    for j, field2 in enumerate(metrics):
        if i == 0:
            axs[0][j].set_title(field2)
        if i == j:
            axs[i][j].spines['right'].set_visible(False)
            axs[i][j].spines['top'].set_visible(False)
            axs[i][j].spines['bottom'].set_visible(False)
            axs[i][j].spines['left'].set_visible(False)
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
        else:
            axs[i][j].scatter(
                df[field2], df[field1], 
                c=np.array([unique_colors[n] for n in group_idx]),
            )
handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(unique_colors, unique_labels)]
plt.legend(handles=handles, title="_".join([f"[{g.split('.')[-1]}]" for g in group_fields]))
plt.tight_layout()

PLOT_DIR.mkdir(exist_ok=True)
plt.savefig(PLOT_DIR / "correlation" / f'{RUN_TAG}_all_{prefix}correlation.png')