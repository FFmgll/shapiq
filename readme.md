## 📄 SHAP-IQ: Unified Approximation of any-order Shapley Interactions
This repository holds the supplement material for the contribution _SHAP-IQ: Unified Approximation of any-order Shapley Interactions_.

### 🚀 Quickstart
For a quick introduction, we refer to the `main.ipynb` notebook:
Install the dependencies via `pip install -r requirements.txt` and run the notebook.

#### Setup
```python
from games import ParameterizedSparseLinearModel
from approximators import SHAPIQEstimator

game = ParameterizedSparseLinearModel(
    n=30, # n of players
    weighting_scheme="uniform", # how the interactions should be distributed over the subset sizes
    min_interaction_size=1, # min size of interactions in the model
    max_interaction_size=20, # max size of interactions in the model
    n_interactions=100, # number of interactions in the model
    n_non_important_features=3 # number of dummy (zero weight) features, which will also not be part of the interactions
)

game_name = game.game_name
game_fun = game.set_call
n_players = game.n
player_set = set(range(n_players))

interaction_order = 2
```

#### SHAP-IQ to approximate the Shapley Interaction Index

```python

shapiq_sii = SHAPIQEstimator(
    interaction_type="SII", N=player_set, order=interaction_order,  top_order=True
)

sii_scores = shapiq_sii.compute_interactions_from_budget(
    game=game_fun, budget=budget,
    pairing=False, sampling_kernel="ksh", only_sampling=False, only_expicit=False, stratification=False
)
```
#### SHAP-IQ to approximate the Shapley Taylor Index

```python
shapiq_sti = SHAPIQEstimator(
    interaction_type="STI", N=player_set, order=interaction_order,  top_order=True
)

sti_scores = shapiq_sti.compute_interactions_from_budget(
    game=game_fun, budget=budget,
    pairing=False, sampling_kernel="ksh", only_sampling=False, only_expicit=False, stratification=False
) 
```
#### SHAP-IQ to approximate the Shapley Faith Index

```python
shapiq_FSI = SHAPIQEstimator(
    interaction_type="FSI", N=player_set, order=interaction_order,  top_order=True
)

FSI_scores = shapiq_FSI.compute_interactions_from_budget(
    game=game_fun, budget=budget,
    pairing=False, sampling_kernel="ksh", only_sampling=False, only_expicit=False, stratification=False
)
```

### ✅ Validate Experiments

To run and validate the same experiments as in the paper we refer to `run_experiment.py`, `run_experiment_sln.py`, and `unbiased_vs_shapx.py`.

#### Sum of unanimity models (SOUM)
To run the experiemtns on the synthetic model functions (synthetic game functions) we refer to `experiment_run_soum.py`. 
There you can specify the complexity of the model functions.

#### Language Model (LM)
To run the experiments conducted on the language model we refer to `experiment_run_look_up.py`.
We compute interaction indices for a sentiment analysis model. 
The underlying language model is a finetuned `DistilBert` version: [dhlee347/distilbert-imdb](https://huggingface.co/dhlee347/distilbert-imdb)
For more information on the movie reviews and value function we refer to `precompute_lm.py`.

#### Image Classification Model (ICM)
To run the experiments conducted on the image classification model we refer to `experiment_run_look_up.py`.
The underlying image classifier is a ResNet trained on ImageNet as provided by `torchvision.models.resnet18`.
For more information on the superpixels and value function we refer to `precompute_icm.py`.

#### SHAP-IQ = UnbiasedKernelSHAP:
To validate the experiment, that `UnbiasedKernelSHAP` and `SHAP-IQ` indeed are the same method, we refer to `test_unbiased_vs_shapiq.py`.

#### n-SII efficiency throught sampling:
To validate the claim that SHAP-IQ preserves the efficiency of n-SII throughout the sampling process, we refer to `test_n_sii_efficiency.py`.

#### FSI is not s-efficient:
To validate the claim that FSI is not s-efficient, we refer to `test_s_efficiency.py`.
