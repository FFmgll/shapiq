import copy

import numpy as np
from scipy.special import binom

from games import LinearModelWithCrosses,ParameterizedSparseLinearModel
from approximators import SHAPIQEstimator
from approximators.regression import RegressionEstimator


if __name__ == "__main__":

    # setup the game function (here we use a
    game = LinearModelWithCrosses(n=3)
    #game = ParameterizedSparseLinearModel(
    #    n=10,
    #    weighting_scheme="uniform",
    #    min_interaction_size=2,
    #    max_interaction_size=7,
    #    n_interactions=100,
    #    n_non_important_features=0
    #)
    game_name = game.game_name
    game_fun = game.set_call
    n = game.n
    N = set(range(n))

    interaction_order = 2



    # SHAP-IQ to approximate the Shapley Interaction Index
    shapley_extractor_sii = SHAPIQEstimator(
        N=N,
        max_order=interaction_order,
        min_order=1,
        interaction_type="SII"
    )

    # SHAP-IQ to approximate the Shapley Taylor Index
    shapley_extractor_sti = SHAPIQEstimator(
        N=N,
        max_order=interaction_order,
        min_order=1,
        interaction_type="STI"
    )

    # SHAP-IQ to approximate the Shapley Faith Index
    shapley_extractor_sfi = SHAPIQEstimator(
        N=N,
        max_order=interaction_order,
        min_order=interaction_order,
        interaction_type="SFI"
    )

    approximators = {
        "SII": shapley_extractor_sii,
        "STI": shapley_extractor_sti,
        "SFI": shapley_extractor_sfi
    }

    print("Starting exact computations")
    shapx_exact_values = {}

    for interaction_type, approximator in approximators.items():
        print("Exact values are calculated via brute force.")
        if interaction_type in ("STI","SII"):
            shapx_exact_values[interaction_type] = copy.deepcopy(
                approximator.compute_interactions_complete(game_fun)
            )
        if interaction_type == "SFI":
            shapley_extractor_sfi_regression = RegressionEstimator(
                N, interaction_order)
            shapx_exact_values[interaction_type] = shapley_extractor_sfi_regression.compute_exact_values(game_fun)

    shapx_exact_values["n_shapley"] = approximators["SII"].transform_interactions_in_n_shapley(shapx_exact_values["SII"])