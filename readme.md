## 📄 SHAP-IQ: Unified Approximation of any-order Shapley Interactions
This repository holds the supplement material for the contribution _SHAP-IQ: Unified Approximation of any-order Shapley Interactions_.

### Validate Experiments

To run and validate the same experiments as in the paper we refer to `run_experiment.py` and `run_experiment_sln.py`.

#### Sum of unanimity models (SOUM)
To run the experiemtns on the synthetic model functions (synthetic game functions) we refer to `run_experiment_sln.py`. 
There you can specify the complexity of the model functions.

#### Language Model (LM)
To run the experiments conducted on the language model we refer to `run_experiment_sln.py`.
We compute interaction indices for a sentiment analysis model. 
The underlying language model is a finetuned `DistilBert` version: [dhlee347/distilbert-imdb](https://huggingface.co/dhlee347/distilbert-imdb)
