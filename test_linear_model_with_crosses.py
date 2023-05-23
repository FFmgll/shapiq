import copy

import numpy as np

from games.all import LinearModelWithCrosses
from approximators import SHAPIQEstimator
from approximators.regression import RegressionEstimator

if __name__ == "__main__":

    # setup the game function (here we use a
    game_list = [
        LinearModelWithCrosses(n=3, order=2, variant="STI_example", c=0),
        LinearModelWithCrosses(n=3, order=2, variant="STI_example", c=1),
        LinearModelWithCrosses(n=7, order=3, variant="STI_example", c=1),
        LinearModelWithCrosses(n=7, order=2, variant="STI_example", c=1, l=3)]

    for game in game_list:

        game_name = game.game_name
        game_fun = game.set_call
        n = game.n
        N = set(range(n))

        interaction_order = game.order

        # SHAP-IQ to approximate the Shapley Interaction Index
        shapley_extractor_sii = SHAPIQEstimator(
            N=N,
            order=interaction_order,
            interaction_type="SII",
            top_order=False
        )

        # SHAP-IQ to approximate the Shapley Taylor Index
        shapley_extractor_sti = SHAPIQEstimator(
            N=N,
            order=interaction_order,
            interaction_type="STI",
            top_order=False
        )

        approximators = {
            "SII": shapley_extractor_sii,
            "STI": shapley_extractor_sti,
        }

        # print("Starting exact computations")
        shapx_exact_values = {}

        for interaction_type, approximator in approximators.items():
            # print("Exact values are calculated via brute force.")
            shapx_exact_values[interaction_type] = copy.deepcopy(
                approximator.compute_interactions_complete(game_fun)
            )

        # FSI values
        shapley_extractor_FSI_regression = RegressionEstimator(
            N, interaction_order)
        shapx_exact_values["FSI"] = shapley_extractor_FSI_regression.compute_exact_values(game_fun)

        # n-Shapley
        shapx_exact_values["n_shapley"] = approximators["SII"].transform_interactions_in_n_shapley(
            shapx_exact_values["SII"])

        print("---------------")
        print(game.n, " features: ", game.variant)
        for vals in shapx_exact_values:
            print("computed by: ", vals)
            results = {}
            for key in shapx_exact_values[vals]:
                results[key] = np.unique(
                    np.round(
                        shapx_exact_values[vals][key][np.nonzero(shapx_exact_values[vals][key])],
                        6))
            print(results)
