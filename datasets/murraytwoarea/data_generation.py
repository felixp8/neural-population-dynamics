import numpy as np
import pandas as pd
import h5py
import torch
from typing import Optional, Any, Union

from synergen.core import NeuralDataGenerator
from synergen.systems.base import UncoupledSystem
from synergen.systems.models.base import Model
from synergen.systems.envs.neurogym import NeurogymEnvironment
from synergen.embedding.subsample import Subsample
from synergen.neural_data.lin_nonlin import LinearNonlinearPoisson
from synergen.utils.data_io import dict_to_h5

from task import PulseDiscriminationTask
from model import MurrayTwoArea

seed = 0
region = "pfc"
sanity_check = True
dataset_name = "murray_recnf_pfc_ext"

env = PulseDiscriminationTask(
    dt=10,
    frequencies=[4, 8, 12, 16, 20, 24, 28, 32],
    rewards=None,
    timing=dict(fixation=50, stimulus=900, decision=50),
    noise_type="ornstein-uhlenbeck",
    noise_scale=0.009,
    pulse_height=0.012,
    pulse_width=50,
    pulse_kernel="box",
    extra_channels=2,
)
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

# define model class
class MurrayTwoAreaModel(Model):
    def __init__(self, model: MurrayTwoArea, seed=None):
        super().__init__(
            n_dim=4,
            seed=seed,
        )
        self.model = model
        self.dtype = list(model.parameters())[0].dtype

    def simulate(
        self,
        ics: np.ndarray,
        inputs: np.ndarray,
    ):
        with torch.no_grad():
            ics = torch.from_numpy(ics).to(self.dtype)
            inputs = torch.from_numpy(inputs).to(self.dtype)
            ics = ics.unsqueeze(dim=1)
            (states, rates), last_state = self.model(inputs, ics, steps_per_bin=10)
            eff_pfc_inputs = states[:,:,0:2] @ self.model.J[0:2, 2:4]
            eff_ppc_inputs = states[:,:,2:4] @ self.model.J[2:4, 0:2]
            rates = rates.numpy()
            outputs = (rates[:,:,-1] - rates[:,:,-2])
            hidden = states.numpy()
        temporal_data = dict(
            eff_pfc_inputs=eff_pfc_inputs,
            eff_ppc_inputs=eff_ppc_inputs,
            pop_rates=rates,
        )
        return hidden, outputs, None, temporal_data
    

# env = NeurogymEnvironment(env=env, seed=seed)
# class PulseTaskEnvironment(NeurogymEnvironment):
#     def sample_inputs(self, *args, **kwargs):
#         trial_info, inputs, temporal_data = super().sample_inputs(*args, **kwargs)
#         if hasattr(self.env, 'true_stimulus'):
#             temporal_data["true_ext_inputs"] = self.env.true_stimulus
#         else:
#             print(f"failed to get true stimulus")
#         return trial_info, inputs, temporal_data


# load model from checkpoint
net = MurrayTwoArea(
    Js21=-0.04,
    Jt21=-0.08,
    dt=0.01,
)
net.eval()

# instantiate objects
model = MurrayTwoAreaModel(model=net, seed=seed)
env = NeurogymEnvironment(env=env, seed=seed)
system = UncoupledSystem(model=model, env=env, seed=seed)

data_sampler = LinearNonlinearPoisson(
    output_dim=30,
    proj_weight_dist="uniform",
    proj_weight_params=dict(
        low=-1.0,
        high=1.0,
    ),
    normalize_proj=False,
    nonlinearity="exp",
    nonlinearity_params=dict(
        offset=-3.0,
    ),
    mean_center="all",
    rescale_variance="all",
    target_variance=1.0,
    clip_val=5,
)

embedding = Subsample(
    n_dim=(4 if region == "all" else 2),
    subsample_method=("last" if region == "pfc" else "first"),
    seed=seed,
)

datagen = NeuralDataGenerator(
    system=system,
    data_sampler=data_sampler,
    embedding=embedding,
    seed=seed,
)

# kwargs
trajectory_kwargs = dict(
    n_traj=1000,
    ic_kwargs=dict(
        dist="exponential",
        dist_params=dict(scale=0.04)
    ),
    simulation_kwargs=dict(),
    trial_kwargs=dict(
        temporal_data_fields=["true_stimulus"],
    )
)

export_kwargs = dict(
    file_format="lfads",
    file_path=f"./data/{dataset_name}.h5",
    data_field="spikes",
    truth_field="rates",
    latent_field="temporal_data.embedded_states",
    ext_input_field="temporal_data.eff_pfc_inputs",
    overwrite=True,
    trial_info_as_csv=True,
    extra_fields=[
        "temporal_data.eff_ppc_inputs", 
        "temporal_data.eff_pfc_inputs", 
        "temporal_data.pop_rates", 
        "temporal_data.true_stimulus",
        "states",
    ],
)

# generate data
output = datagen.generate_dataset(
    trajectory_kwargs=trajectory_kwargs,
    export_kwargs=export_kwargs,
)

with h5py.File(f"./data/{dataset_name}_params.h5", "w") as h5f:
    dict_to_h5(output.general_data, h5f)

if sanity_check:

    from sklearn.decomposition import PCA

    states_all = output.states.reshape(-1, output.states.shape[-1]).copy()
    states_all -= states_all.mean(axis=0, keepdims=True)
    pca1 = PCA()
    pca1.fit(states_all)
    orig_participation_ratio = np.square(np.sum(pca1.explained_variance_)) / np.sum(
        np.square(pca1.explained_variance_)
    )
    print(orig_participation_ratio)

    rates_all = (
        output.neural_data["rates"]
        .reshape(-1, output.neural_data["rates"].shape[-1])
        .copy()
    )
    rates_all -= rates_all.mean(axis=0, keepdims=True)
    pca2 = PCA()
    pca2.fit(rates_all)
    rate_participation_ratio = np.square(np.sum(pca2.explained_variance_)) / np.sum(
        np.square(pca2.explained_variance_)
    )
    print(rate_participation_ratio)

    logrates_all = np.log(
        output.neural_data["rates"].reshape(-1, output.neural_data["rates"].shape[-1])
    )
    logrates_all -= logrates_all.mean(axis=0, keepdims=True)
    pca3 = PCA()
    pca3.fit(logrates_all)
    lograte_participation_ratio = np.square(np.sum(pca3.explained_variance_)) / np.sum(
        np.square(pca3.explained_variance_)
    )
    print(lograte_participation_ratio)

    mean_spikes = output.neural_data["spikes"].mean(axis=(0,1))
    print(mean_spikes / 0.01)

    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'sans-serif'
    fig, axs = plt.subplots(1, 4, figsize=(14,4))
    axs[0].plot(output.states[:,:,0].T)
    axs[1].plot(output.states[:,:,1].T)
    axs[2].plot(output.states[:,:,2].T)
    axs[3].plot(output.states[:,:,3].T)
    plt.savefig('temp.png')
