import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from netrep.metrics import LinearMetric
from tqdm import tqdm


PROJECT_STR = "singlearea"
DATASET_STR = "murray_rec_pfc"
RUN_TAG = "231120_lfads_frozen"
RUN_DIR = Path("/home/fpei2/interp/neural-population-dynamics/singlearea/lfads/runs/") / DATASET_STR / RUN_TAG

region = "ppc" if ("ppc" in DATASET_STR) else "pfc"

fltn = lambda x: x.reshape(-1, x.shape[-1])

scores = []

for model_dir in tqdm(sorted(RUN_DIR.glob("run_model_*/"))):
    assert (model_dir / "lfads_output_sess0.h5").exists()
    assert (model_dir / "params.json").exists()
    
    with open(model_dir / "params.json", "r") as f:
        params = json.load(f)

    with h5py.File((model_dir / "lfads_output_sess0.h5"), "r") as h5f:
        train_inds = h5f['train_inds'][()]
        valid_inds = h5f['valid_inds'][()]
        def merge_tv(field):
            arr_dims = h5f[f"train_{field}"].shape[1:]
            full_arr = np.empty((len(train_inds) + len(valid_inds),) + arr_dims)
            full_arr[train_inds] = h5f[f"train_{field}"][()]
            full_arr[valid_inds] = h5f[f"valid_{field}"][()]
            return full_arr

        true_latents = merge_tv("latents") # h5f['train_latents'][()]
        inf_latents = merge_tv("factors") # h5f['valid_factors'][()]

        if params.get("model.co_dim", 4) <= 0:
            true_inputs = None
            inf_inputs = None
        else:
            if region == "pfc":
                true_inputs = merge_tv("eff_pfc_inputs")
            else:
                true_inputs = merge_tv("eff_ppc_inputs") + merge_tv("true_stimulus")[:, :, :2]
            inf_inputs = merge_tv("gen_inputs") # h5f['train_gen_inputs'][()]

    def eval_procrustes(inputs, targets, metric_name="metric"):
        if inputs is None or targets is None:
            return {metric_name + "_lin": np.nan, metric_name + "_rot": np.nan}
        if len(inputs.shape) > 2:
            inputs_flt = fltn(inputs)
        if len(targets.shape) > 2:
            targets_flt = fltn(targets)
        lm = LinearMetric(alpha=1.0e-6)
        rm = LinearMetric(alpha=1.0)
        lm.fit(inputs_flt, targets_flt)
        rm.fit(inputs_flt, targets_flt)
        return {
            metric_name + "_lin": lm.score(inputs_flt, targets_flt), 
            metric_name + "_rot": rm.score(inputs_flt, targets_flt),
        }
    
    model_info = {"name": model_dir.name}
    model_info.update(eval_procrustes(inf_latents, true_latents, 'latent_procrustes'))
    model_info.update(eval_procrustes(inf_inputs, true_inputs, 'input_procrustes'))

    model_info.update(params)

    scores.append(model_info)

scores = pd.DataFrame(scores)

result_dir = Path("../results/") / DATASET_STR / RUN_TAG
result_dir.mkdir(exist_ok=True)
scores.to_csv(result_dir / "procrustes.csv", index=False)
