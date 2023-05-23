"""This module is used to run experiments to investigate weather FSI is s-efficient."""
import numpy as np

from games import ParameterizedSparseLinearModel
from approximators import SHAPIQEstimator


if __name__ == "__main__":

    # game settings
    N_FEATURES: int = 10  # number of players
    N_INTERACTIONS: int = 30  # number of interactions in the model
    N_NON_IMPORTANT_FEATURES: int = 0  # percentage of dummy players (zero value players)

    # SHAP-IQ settings
    s_0 = 2  # interaction order to compute values for

    # Game Function --------------------------------------------------------------------------------
    game = ParameterizedSparseLinearModel(
        n=N_FEATURES,
        weighting_scheme="uniform",
        min_interaction_size=1,
        max_interaction_size=N_FEATURES - 1,
        n_interactions=N_INTERACTIONS,
        n_non_important_features=N_NON_IMPORTANT_FEATURES
    )
    game_name = game.game_name
    game_fun = game.set_call
    n = game.n
    N = set(range(n))

    # Estimator ------------------------------------------------------------------------------------
    shapley_extractor_FSI = SHAPIQEstimator(N, order=s_0, interaction_type="FSI")
    shapley_extractor_sti = SHAPIQEstimator(N, order=s_0, interaction_type="STI")
    shapley_extractor_sii = SHAPIQEstimator(N, order=s_0, interaction_type="SII")

    shap_iq_estimators = {
        "SII": shapley_extractor_sii,
        "STI": shapley_extractor_sti,
        "FSI": shapley_extractor_FSI
    }

    # Compute s-efficiency -------------------------------------------------------------------------
    for interaction_index, approximator in shap_iq_estimators.items():
        print(f"Running SHAP-IQ for {interaction_index} d = {n} with s_0 = {s_0} and k in range "
              f"[{s_0}, {n - s_0}]:")
        s_efficiency = np.zeros(shape=(n, n))
        for k in range(s_0, n - s_0 + 1):
            s_efficiency += approximator._compute_interactions_complete_k(game=game_fun, k=k)[s_0]
        sum_s_efficiency = np.sum(s_efficiency)
        print(f"Sum of terms with set sizes k: {sum_s_efficiency}")
        print()
