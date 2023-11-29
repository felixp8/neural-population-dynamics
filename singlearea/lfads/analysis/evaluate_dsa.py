import json
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from DSA import DSA
from sklearn.decomposition import PCA
from tqdm import tqdm


PROJECT_STR = "singlearea"
DATASET_STR = "murray_rec_ppc"
RUN_TAG = "231115_lfads_lr_rs_ppc"
RUN_DIR = Path("/home/fpei2/interp/neural-population-dynamics/singlearea/lfads/runs/") / DATASET_STR / RUN_TAG

scores = []

fltn = lambda x: x.reshape(-1, x.shape[-1])

def apply_pca(arr, explained_variance=0.99):
    pca = PCA()
    arr = pca.fit_transform(fltn(arr)).reshape(arr.shape)
    trunc = np.nonzero(np.cumsum(pca.explained_variance_ratio_) >= explained_variance)[0][0]
    arr = arr[:,:,:trunc]
    return arr

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

        true_latents = merge_tv("latents") # h5f['train_latents'][()]
        inf_latents = merge_tv("factors") # h5f['valid_factors'][()]

    def eval_dsa(inputs, targets, metric_name="metric"):
        inputs = apply_pca(inputs, explained_variance=0.99)
        targets = apply_pca(targets, explained_variance=0.99)
        dsa = DSA(
            X=inputs,
            Y=targets,
            n_delays=30,
            delay_interval=1,
            rank=None,
            rank_thresh=None,
            rank_explained_variance=0.99,
            lamb=0.0,
            send_to_cpu=True,
            iters=1000,
            score_method="angular",
            lr=0.01,
            group="O(n)",
            zero_pad=True,
            device='cuda:0', # 'cpu',
            verbose=False,
            threaded=0,
        )
        score = dsa.fit_score().item()
        return {
            metric_name: score, 
        }
    
    model_info = {"name": model_dir.name}
    model_info.update(eval_dsa(inf_latents, true_latents, 'latent_dsa'))

    with open(model_dir / "params.json", "r") as f:
        params = json.load(f)
    model_info.update(params)

    scores.append(model_info)

scores = pd.DataFrame(scores)

result_dir = Path("../results/") / DATASET_STR / RUN_TAG
result_dir.mkdir(exist_ok=True)
scores.to_csv(result_dir / "dsa.csv", index=False)
