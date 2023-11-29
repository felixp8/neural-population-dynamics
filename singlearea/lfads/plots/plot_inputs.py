import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

DATASET_STR = "murray_rec_pfc"
RUN_TAG = "231118_lfads_lr"
RUN_DIR = Path("/home/fpei2/interp/neural-population-dynamics/singlearea/lfads/runs/") / DATASET_STR / RUN_TAG

region = "ppc" if ("ppc" in DATASET_STR) else "pfc"

RESULT_DIR = Path("../results") / DATASET_STR / RUN_TAG
PLOT_DIR = Path("./") / DATASET_STR

ref_metric = "fwd_input_r2" # "input_procrustes_rot"

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

sorted_df = df.sort_values(ref_metric, ascending=(False if "procrustes" in ref_metric else True))

ok_score = (df[ref_metric].max() + df[ref_metric].min()) / 2
ok_idx = (sorted_df[ref_metric] - ok_score).abs().argmin()
model_names = sorted_df.name.iloc[[0, ok_idx, len(sorted_df) - 1]]
model_scores = sorted_df.iloc[[0, ok_idx, len(sorted_df) - 1]][["rate_r2", "fwd_input_r2", "bwd_input_r2", "input_procrustes_rot"]]
model_labels = ["bad", "ok", "good"]

file_paths = [RUN_DIR / name / "lfads_output_sess0.h5" for name in model_names]
save_path = PLOT_DIR / "inputs " / f"{RUN_TAG}.png"

for i, file_path in enumerate(file_paths):
    with h5py.File(file_path, 'r') as h5f:
        train_inds = h5f['train_inds'][()]
        valid_inds = h5f['valid_inds'][()]
        def merge_tv(field):
            arr_dims = h5f[f"train_{field}"].shape[1:]
            full_arr = np.empty((len(train_inds) + len(valid_inds),) + arr_dims)
            full_arr[train_inds] = h5f[f"train_{field}"][()]
            full_arr[valid_inds] = h5f[f"valid_{field}"][()]
            return full_arr

        if region == "pfc":
            true_inputs = merge_tv("eff_pfc_inputs")
        else:
            true_inputs = merge_tv("eff_ppc_inputs") + merge_tv("true_stimulus")[:, :, :2]
        inf_inputs = merge_tv("gen_inputs") # h5f['train_gen_inputs'][()]

    fig, axs = plt.subplots(3, 4, figsize=(14, 10))

    inf_input_fltn = inf_inputs.reshape(-1, inf_inputs.shape[-1])
    inf_input_fltn = np.concatenate([np.ones((inf_input_fltn.shape[0], 1)), inf_input_fltn], axis=1)
    true_input_fltn = true_inputs.reshape(-1, true_inputs.shape[-1])
    weights = np.linalg.inv(inf_input_fltn.T @ inf_input_fltn) @ (inf_input_fltn.T @ true_input_fltn)
    pred_input = inf_input_fltn @ weights
    pred_input = pred_input.reshape(true_inputs.shape)

    axs[0,0].plot(true_inputs[0])
    axs[0,1].plot(true_inputs[1])
    axs[0,2].plot(true_inputs[2])
    axs[0,3].plot(true_inputs[3])

    axs[1,0].plot(pred_input[0])
    axs[1,1].plot(pred_input[1])
    axs[1,2].plot(pred_input[2])
    axs[1,3].plot(pred_input[3])

    axs[2,0].plot(inf_inputs[0])
    axs[2,1].plot(inf_inputs[1])
    axs[2,2].plot(inf_inputs[2])
    axs[2,3].plot(inf_inputs[3])

    axs[0,0].set_ylabel("true input")
    axs[1,0].set_ylabel("inf->true input")
    axs[2,0].set_ylabel("inferred input")

    plt.suptitle(
        f"rate r2={model_scores.iloc[i]['rate_r2']:.3f}, "
        f"fwd r2={model_scores.iloc[i]['fwd_input_r2']:.3f}, "
        f"bwd r2={model_scores.iloc[i]['bwd_input_r2']:.3f}, "
        f"proc={model_scores.iloc[i]['input_procrustes_rot']:.3f}"
    )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "inputs" / f"{RUN_TAG}_{model_labels[i]}.png")
    plt.close()

