import os
import shutil
import inspect
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "singlearea"
DATASET_STR = "murray_rec_pfc"
MODEL_STR = "arprior_10d"
RUN_TAG = "231106_arprior_10d_rs"
RUN_DIR = Path("/home/fpei2/interp/neural-population-dynamics/singlearea/lfads/runs/") / DATASET_STR / RUN_TAG
RUN_TYPE = "multi" # "pbt"
# ------------------------------

# weird path handling so we can have configs separate from lfads-torch
func_path = Path(inspect.getfile(run_model)).resolve().parent
config_path = Path(f'./configs/{RUN_TYPE}.yaml').resolve()
rel_config_path = os.path.relpath(config_path, func_path)


# Function to keep dropout and CD rates in-bounds
def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": MODEL_STR,
    # "logger.wandb_logger.project": PROJECT_STR,
    # "logger.wandb_logger.tags.1": DATASET_STR,
    # "logger.wandb_logger.tags.2": RUN_TAG,
}
for model_dir in RUN_DIR.glob("run_model_*/"):
    # Switch working directory to this folder (usually handled by tune)
    os.chdir(model_dir)
    # Load the best model and run posterior sampling (skip training)
    if RUN_TYPE == "pbt":
        ckpt_dir = sorted(model_dir.glob("checkpoint_*/"))[-1] # choose last one b/c idk
    elif RUN_TYPE == "multi":
        print("don't run this")
        exit(0)
    run_model(
        overrides=mandatory_overrides,
        checkpoint_dir=ckpt_dir,
        config_path=rel_config_path,
        do_train=False,
    )
