import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# for true dyn_model

import sys
sys.path.insert(1, "../../../datasets/murraytwoarea/")
from model import MurraySingleArea, MurrayTwoArea

dyn_model = MurraySingleArea(
    Js=0.4182,
    Jt=0.28387,
    dt=0.01,
    I_0=0.334,
    tau_ndma=0.06,
    gamma=0.641,
)

with h5py.File("../../../datasets/murraytwoarea/data/murray_rec_pfc.h5", "r") as h5f:
    latents = h5f["train_latents"][:, :, 2:] # TODO: CHANGE
    inputs = h5f["train_eff_inputs"][()] # eff_pfc_inputs

ics = latents[:, 0, :]
mean_inputs = np.mean(np.abs(inputs), axis=(0,1))

# for data dyn_model

# dyn_model = ...
# ics = ...
# mean_inputs = ...

# set up simulation

device = next(iter(dyn_model.parameters())).device
dtype = next(iter(dyn_model.parameters())).dtype

num_simulations = 20
simulation_size = 500
stim_steps = 100
rest_steps = 100
min_k = 1
max_k = 10

clusterer_class = GaussianMixture # KMeans
clusterer_kwargs = dict() # dict(n_init='auto')

# clusterer_class = AgglomerativeClustering
# clusterer_kwargs = dict()

best_ks = []

# run simulations

noise_strengths = np.linspace(0, 1, num_simulations + 1)[1:]

for std in noise_strengths:
    sim_ics = ics[np.random.choice(ics.shape[0], size=simulation_size)][None, :, :]
    inputs = np.random.normal(
        loc=0.0, scale=(mean_inputs[None, None, :] * std), 
        size=(simulation_size, stim_steps, mean_inputs.shape[-1]))
    inputs = np.concatenate([
        inputs, np.zeros((simulation_size, rest_steps, mean_inputs.shape[-1]))], axis=1)

    sim_ics = torch.from_numpy(sim_ics).to(device).to(dtype)
    inputs = torch.from_numpy(inputs).to(device).to(dtype)

    _, last_states = dyn_model(inputs, sim_ics)

    last_states = last_states.detach().cpu().numpy()[0]

    scores = []
    for k in range(min_k, max_k + 1):
        clusterer = clusterer_class(k, **clusterer_kwargs)
        labels = clusterer.fit_predict(last_states)
        # if len(np.unique(labels)) <= 1:
        #     score = np.nan
        # else:
        #     score = silhouette_score(last_states, labels)
        score = clusterer.bic(last_states)
        scores.append(score)

    # if np.max(scores) < 0.5:
    #     best_k = 1
    # else:
    #     best_k = list(range(min_k, max_k + 1))[np.argmax(scores)]
    best_k = list(range(min_k, max_k + 1))[np.argmin(scores)]
    best_ks.append(best_k)
    clusterer = clusterer_class(best_k, **clusterer_kwargs)
    labels = clusterer.fit_predict(last_states)

    # clusterer = clusterer_class(**clusterer_kwargs)
    # labels = clusterer.fit_predict(last_states)
    # best_ks.append(clusterer.n_clusters_)

    plt.scatter(last_states[:, 0], last_states[:, 1], c=labels)
    plt.savefig(f'{int(std*1000):04d}.png')
    plt.clf()

# save and plot results

plt.plot(noise_strengths, best_ks)
plt.savefig('temp.png')

# utils
