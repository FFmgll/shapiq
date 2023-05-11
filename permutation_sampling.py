from games.all import LinearModelWithCrosses
from approximators.permutation import PermutationSampling
import copy
from approximators.shapiq import SHAPIQEstimator

if __name__=="__main__":
    game = LinearModelWithCrosses(n=10, variant="STI_example")
    game_name = game.game_name
    game_fun = game.set_call
    n = game.n
    N = set(range(n))

    budget = 10000
    interaction_order = 2

    shapley_extractor_sii_permutation = PermutationSampling(N, interaction_order,
                        interaction_type="SII", top_order=False)
    shapley_extractor_sti_permutation = PermutationSampling(N, interaction_order,
                        interaction_type="STI", top_order=False)

    baselines = {
        "SII": shapley_extractor_sii_permutation,
        "STI": shapley_extractor_sti_permutation,
    }

    results = {}

    for interaction_type in baselines:
        results[interaction_type] = baselines[interaction_type].approximate_with_budget(game_fun, budget)

    #Exact values
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

        print("Starting exact computations")
        shapx_exact_values = {}

        for interaction_type, approximator in approximators.items():
            print("Exact values are calculated via brute force.")
            shapx_exact_values[interaction_type] = copy.deepcopy(
                approximator.compute_interactions_complete(game_fun)
            )
