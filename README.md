# neural-population-dynamics
some little experiments in modeling neural population dynamics

## datasets

currently only one dataset, as galgali adversarial thing has not been implemented. `murraytwoarea` is a simple two area PPC-PFC model.

### murraytwoarea

* `notebooks/` contains a simple explanation of how the two-area model works in `analysis.ipynb`. I also attempt some basic data checks in `data_validation.ipynb`
* `data/` contains the simulated neural data from PFC, made using `data_generation.py`
* `model.py` contains a PyTorch implementation of single-area and two-area models described in the Murray paper
* `task.py` contains `neurogym` implementations of simple decision making task
* `data_generation.py` contains a script for simulating neural data with [synergen](https://github.com/felixp8/synergen)

## singlearea

these are just experiments in how well LFADS-like models applied to single areas within multi-area networks can reconstruct single-area dynamics. it's probably impossible - see Galgali et al.'s work with the same Murray model - but doesn't hurt to try

in the `lfads/` directory we have:
* `train/` which contains training scripts for LFADS models (currently using my fork of `lfads-torch`)
* `configs/` contains run configs, unsurprisingly
* `analysis/` contains scripts for evaluating model performance on various metrics
* `results/` contains csv files with results of above evaluations
* `plots/` contains plots visualizing above results. `plots/old/` also has some interesting and similar plots but I no longer have those runs saved

there are three(-ish) dataset variants I use
* `murray_rec_pfc` - vanilla two-area model with recurrent feedback, data from PFC
* `murray_rec_pfc_ext` - same model as above, but the dynamics models get the true PPC input instead of inferring it
* `murray_recnf_pfc` - modified Murray model with negative feedback from PFC, so the PFC activity isn't super different but the inputs from PPC return to 0 after the decision is made

there are also a few model variants I use
* `lfads_base` - pretty much standard LFADS, but without the factor bottleneck. I just make the generator smaller in general
* `lfads_lr` - standard LFADS but with a low-rank RNN generator
* `lfads_odin` - LFADS with an MLPRNN generator and a flow readout. note that inputs are treated linearly and not concatenated as input to the MLP. in old runs (`plots/old/`) I tried both, which is why there's `odin` and `odinlin` there
* `lfads_frozen` - only done for one dataset, but this is LFADS with a frozen generator that is not trained at all. the IC encoder is also overridden so ICs are sampled from a zero-mean gaussian. this is intended to force the controller to do all of the work, basically, guiding whatever dynamics it's given to follow the desired path

lastly there are a few metrics I use
* `rate_r2` - $R^2$ of model predicted firing rates and true firing rates
* `fwd_latent_r2` - $R^2$ of linear regression from model latents to true latents
* `bwd_latent_r2` - $R^2$ of linear regression from true latents to model latents
* `latent_procrustes_lin` - non-regularized procrustes angular distance metric between model and true latents. lower is better. see [netrep](https://github.com/ahwillia/netrep)
* `latent_procrustes_rot` - regularized procrustes angular distance metric between model and true latents. lower is better
* `fwd_input_r2` - $R^2$ of linear regression from model inputs to true inputs
* `bwd_input_r2` - $R^2$ of linear regression from true inputs to model inputs
* `input_procrustes_lin` - non-regularized procrustes angular distance metric between model and true inputs. lower is better
* `input_procrustes_rot` - regularized procrustes angular distance metric between model and true inputs. lower is better

## TODOS
* basic dynamics analyses, like fixed points comparisons or DSA (doable)
* galgali data - ideally want to show that single-area dynamics models can capture the same differences revealed by residual dynamics but still fail to capture the true dynamics (maybe doable)
* perturbations? multi-area recordings? do they work? (not doable)
