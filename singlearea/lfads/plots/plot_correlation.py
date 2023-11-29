import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

RUN_TAG = "231112_lfads_odinlin_rs"
result_file = f"{RUN_TAG}.csv"
df = pd.read_csv(result_file)

group_fields = ["model", "model.co_dim", "model.gen_cell.rank"] # ["model", "model.co_dim", "model.gen_dim"]
unique_groups = sorted(df.set_index(group_fields).index.unique().tolist())
unique_colors = plt.cm.hsv(np.arange(0, 1, 1/len(unique_groups)))
unique_labels = ["_".join([str(g) for g in group]) for group in unique_groups]

group_idx = np.full((df.shape[0],), np.nan)
for i, group_vals in enumerate(unique_groups):
    group_idx[np.all(df[group_fields] == group_vals, axis=1)] = i
assert not np.any(np.isnan(group_idx))
group_idx = group_idx.astype(int)

group_names = [unique_labels[i] for i in group_idx]
df['group'] = group_names

metric1 = "fwd_input_r2"
metric2 = "rate_r2"

plot = sns.jointplot(data=df, x=metric1, y=metric2, hue="group")
# plt.xlim(0.7, 1.2)
# plt.ylim(0.8, 1.1)
plt.tight_layout()
plt.savefig(f'./plots/{RUN_TAG}_{metric1}_{metric2}_correlation.png')