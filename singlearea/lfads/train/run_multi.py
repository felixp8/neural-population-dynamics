import shutil
import os
import inspect
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "singlearea"
DATASET_STR = "murray_rec_pfc"
# MODEL_STR = "lr_mvnprior"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_lfads_frozen"
RUN_DIR = Path("/home/fpei2/interp/neural-population-dynamics/singlearea/lfads/runs/") / DATASET_STR / RUN_TAG
# ------------------------------

# weird path handling so we can have configs separate from lfads-torch
func_path = Path(inspect.getfile(run_model)).resolve().parent
config_path = Path('../configs/multi.yaml').resolve()
rel_config_path = os.path.relpath(config_path, func_path)

# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    # "model": "odinlin_mvnprior",
    # # emable ext input
    # "model.ext_input_dim": 2, 
    # # disable controller
    # "model.ci_enc_dim": 0, 
    # "model.con_dim": 0, 
    # "model.co_dim": 0, 
    # # extras
    # "model.gen_cell.scale": 0.05,
    # "logger.wandb_logger.project": PROJECT_STR,
    # "logger.wandb_logger.tags.1": DATASET_STR,
    # "logger.wandb_logger.tags.2": RUN_TAG,
}
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
tune.run(
    tune.with_parameters(
        run_model,
        config_path=rel_config_path,
        freeze_generator=True,
    ),
    metric="valid/recon_smth",
    mode="min",
    name=RUN_DIR.name,
    config={
        **mandatory_overrides,
        "model": tune.choice(["base_mvnprior", "base_arprior"]),
        # "model": tune.choice(["lr_mvnprior", "lr_arprior"]),
        # "model": tune.choice(["odin_mvnprior", "odin_arprior"]),
        # "model": tune.choice(["odinlin_mvnprior", "odinlin_arprior"]),
        "model.co_dim": tune.choice([1, 2, 4]),
        "model.gen_dim": tune.choice([10, 20, 40]),
        # "model.gen_dim": tune.choice([5, 10, 20]),
        # "model.gen_cell.rank": tune.choice([2, 5, 8]),
        "model.dropout_rate": tune.uniform(0.2, 0.6),
        "model.kl_co_scale": tune.loguniform(1e-6, 1e-4),
        "model.kl_ic_scale": tune.loguniform(1e-6, 1e-3),
        "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
        "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
    },
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=20,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=FIFOScheduler(),
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=["valid/recon_smth", "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)

# import hydra
# from omegaconf import OmegaConf
# overrides = [f"datamodule={DATASET_STR}", f"model={MODEL_STR}"]
# with hydra.initialize(
#     config_path=os.path.relpath(config_path.parent, os.getcwd()),
#     version_base="1.1",
# ):
#     config = hydra.compose(config_name=config_path.name, overrides=overrides)

# OmegaConf.save(config, RUN_DIR / "run_config.yaml")
