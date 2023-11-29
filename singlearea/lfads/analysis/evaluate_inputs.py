import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


PROJECT_STR = "singlearea"
DATASET_STR = "murray_rec_ppc"
RUN_TAG = "231115_lfads_lr_rs_ppc"
RUN_DIR = Path("/home/fpei2/interp/neural-population-dynamics/singlearea/lfads/runs/") / DATASET_STR / RUN_TAG

fltn = lambda x: x.reshape(-1, x.shape[-1])

# metrics to compute:
# rate R2
# forward/reverse latent R2
# latent procrustes alpha = 1.0
# latent procrustes alpha = 0.0?
# latent dsa
# forward/reverse input R2
# input procrustes alpha = 1.0
# input procrustes alpha = 0.0?

scores = []

for model_dir in tqdm(sorted(RUN_DIR.glob("run_model_*/"))):
    assert (model_dir / "lfads_output_sess0.h5").exists()
    assert (model_dir / "params.json").exists()

    with h5py.File((model_dir / "lfads_output_sess0.h5"), "r") as h5f:
        train_inds = h5f['train_inds'][()]
        valid_inds = h5f['valid_inds'][()]
        def merge_tv(field):
            arr_dims = h5f[f"train_{field}"].shape[1:]
            full_arr = np.empty((len(train_inds) + len(valid_inds),) + arr_dims)
            full_arr[train_inds] = h5f[f"train_{field}"][()]
            full_arr[valid_inds] = h5f[f"valid_{field}"][()]
            return full_arr

        true_ext_inputs = merge_tv("inputs") # h5f['train_inputs'][()]
        true_latents = merge_tv("latents") # h5f['train_latents'][()]
        inf_latents = merge_tv("factors") # h5f['valid_factors'][()]

        # true_inputs = merge_tv("eff_pfc_inputs") # h5f['train_eff_inputs'][()] # true_latents[:,:,:2] @ communication_matrix
        true_inputs = merge_tv("eff_ppc_inputs") + merge_tv("true_stimulus")[:, :, :2]
        inf_inputs = merge_tv("gen_inputs") # h5f['train_gen_inputs'][()]

        # if 'truth' in h5f.keys():
        true_rates = merge_tv("truth") # h5f['valid_truth'][()]
        inf_rates = merge_tv("output_params") # h5f['valid_output_params'][()]

    def eval_r2(inputs, targets, metric_name="metric"):
        if len(inputs.shape) > 2:
            inputs_flt = fltn(inputs)
        if len(targets.shape) > 2:
            targets_flt = fltn(targets)
        lr = LinearRegression()
        lr.fit(inputs_flt, targets_flt)
        return {
            metric_name: lr.score(inputs_flt, targets_flt), 
            "st_" + metric_name: np.mean([lr.score(it, tt) for it, tt in zip(inputs, targets)])
        }
    
    model_info = {"name": model_dir.name}
    model_info.update(eval_r2(inf_rates, true_rates, 'rate_r2'))
    model_info.update(eval_r2(inf_rates[valid_inds], true_rates[valid_inds], 'val_rate_r2'))
    model_info.update(eval_r2(inf_latents, true_latents, 'fwd_latent_r2'))
    model_info.update(eval_r2(inf_latents[valid_inds], true_latents[valid_inds], 'val_fwd_latent_r2'))
    model_info.update(eval_r2(true_latents, inf_latents, 'bwd_latent_r2'))
    model_info.update(eval_r2(true_latents[valid_inds], inf_latents[valid_inds], 'val_bwd_latent_r2'))
    model_info.update(eval_r2(inf_inputs, true_inputs, 'fwd_input_r2'))
    model_info.update(eval_r2(inf_inputs[valid_inds], true_inputs[valid_inds], 'val_fwd_input_r2'))
    model_info.update(eval_r2(true_inputs, inf_inputs, 'bwd_input_r2'))
    model_info.update(eval_r2(true_inputs[valid_inds], inf_inputs[valid_inds], 'val_bwd_input_r2'))
    model_info.update(eval_r2(inf_inputs, true_ext_inputs, 'fwd_ext_r2'))
    model_info.update(eval_r2(inf_inputs[valid_inds], true_ext_inputs[valid_inds], 'val_fwd_ext_r2'))
    model_info.update(eval_r2(true_ext_inputs, inf_inputs, 'bwd_ext_r2'))
    model_info.update(eval_r2(true_ext_inputs[valid_inds], inf_inputs[valid_inds], 'val_bwd_ext_r2'))

    with open(model_dir / "params.json", "r") as f:
        params = json.load(f)
    model_info.update(params)

    scores.append(model_info)

scores = pd.DataFrame(scores)
scores.to_csv(RUN_TAG + ".csv", index=False)


