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
    N=player_set, max_order=interaction_order, min_order=interaction_order,
    interaction_type="SII"
)

sii_scores = shapiq_sii.compute_interactions_from_budget(
    game=game_fun, budget=budget,
    pairing=False, sampling_kernel="ksh", only_sampling=False, stratification=False
)
```
#### SHAP-IQ to approximate the Shapley Taylor Index

```python
shapiq_sti = SHAPIQEstimator(
    N=player_set, max_order=interaction_order, min_order=interaction_order,
    interaction_type="STI"
)

sti_scores = shapiq_sti.compute_interactions_from_budget(
    game=game_fun, budget=budget,
    pairing=False, sampling_kernel="ksh", only_sampling=False, stratification=False
) 
```
#### SHAP-IQ to approximate the Shapley Faith Index

```python
shapiq_FSI = SHAPIQEstimator(
    N=player_set, max_order=interaction_order, min_order=interaction_order,
    interaction_type="FSI"
)

FSI_scores = shapiq_FSI.compute_interactions_from_budget(
    game=game_fun, budget=budget,
    pairing=False, sampling_kernel="ksh", only_sampling=False, stratification=False
)
```

### ✅ Validate Experiments

To run and validate the same experiments as in the paper we refer to `run_experiment.py`, `run_experiment_sln.py`, and `unbiased_vs_shapx.py`.

#### Sum of unanimity models (SOUM)
To run the experiemtns on the synthetic model functions (synthetic game functions) we refer to `run_experiment_sln.py`. 
There you can specify the complexity of the model functions.

#### Language Model (LM)
To run the experiments conducted on the language model we refer to `run_experiment_sln.py`.
We compute interaction indices for a sentiment analysis model. 
The underlying language model is a finetuned `DistilBert` version: [dhlee347/distilbert-imdb](https://huggingface.co/dhlee347/distilbert-imdb)

#### SHAP-IQ =  UnbiasedKernelSHAP:
To validate the experiment, that `UnbiasedKernelSHAP` and `SHAP-IQ` indeed are the same method, we refer to `unbiased_vs_shapx.py`.
