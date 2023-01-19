import copy

import numpy as np
import itertools
import random

from evaluation import draw_shapley_values
from shapx.interaction import ShapleyInteractionsEstimator
from shapx.base import determine_complete_subsets
from shapx.unbiased import CovertRegression, calculate_uksh_from_samples, get_weights



class GameWrapper:

    def __init__(self, game):
        self.game = game
        self.players = game.n

    def __call__(self, S):
        result = np.empty(S.shape[0])
        for i in range(S.shape[0]):
            sample = S[i]
            s_set = set()
            for j, value in enumerate(sample):
                if value == 1:
                    s_set.add(j)
            result[i] = self.game.set_call(s_set)
        return result

    def grand(self):
        '''Get grand coalition value.'''
        return self.__call__(np.ones((1, self.players), dtype=bool))[0]

    def null(self):
        '''Get null coalition value.'''
        return self.__call__(np.zeros((1, self.players), dtype=bool))[0]


def get_S_and_game(budget, num_players, weight_vector, N, pairing, game_fun):
    complete_subsets = []
    incomplete_subsets = list(N)
    incomplete_subsets.remove(0)

    all_subsets_to_sample = []

    for complete_subset in complete_subsets:
        combinations = itertools.combinations(N, complete_subset)
        for subset in combinations:
            subset = set(subset)
            all_subsets_to_sample.append(subset)

    remaining_weight = weight_vector[incomplete_subsets] / sum(weight_vector[incomplete_subsets])

    if len(incomplete_subsets) > 0:
        sampled_subsets = []
        while len(sampled_subsets) < budget:
            subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
            ids = np.random.choice(num_players, size=subset_size, replace=False)
            sampled_subsets.append(tuple(sorted(ids)))
            if pairing:
                if len(sampled_subsets) < budget:
                    sampled_subsets.append(tuple(N - set(ids)))
        for subset in sampled_subsets:
            all_subsets_to_sample.append(set(subset))

    game_values = [game_fun(subset) for subset in all_subsets_to_sample]

    return all_subsets_to_sample, game_values


def compare_unbiasedksh_and_shapx(
        game,
        budget: int,
        pairing: bool = True,
        u_ksh_sample_size: int = 5_000
):
    num_players = game.n
    game_fun = game.set_call
    N = set(range(num_players))
    empty_value = game_fun({})
    full_value = game_fun(N)

    weights = get_weights(num_players)
    weight_vector = (np.asarray([0] + [*weights] + [0])) / sum(weights)
    all_subsets_to_sample, game_values = get_S_and_game(
        budget, num_players, weight_vector, N, pairing, game_fun)

    # SII
    interaction_estimator = ShapleyInteractionsEstimator(
        N=N, max_order=1, min_order=1, interaction_type='SII'
    )
    values_shapx_sfi = interaction_estimator.compute_from_samples(
        S_list=all_subsets_to_sample, game_values=game_values,
        val_empty=empty_value, val_full=full_value)
    values_shapx_sii = copy.deepcopy(values_shapx_sfi[1])

    # STI
    interaction_estimator = ShapleyInteractionsEstimator(
        N=N, max_order=1, min_order=1, interaction_type='STI'
    )
    values_shapx_sfi = interaction_estimator.compute_from_samples(
        S_list=all_subsets_to_sample, game_values=game_values,
        val_empty=empty_value, val_full=full_value)
    values_shapx_sti = copy.deepcopy(values_shapx_sfi[1])

    # SFI
    interaction_estimator = ShapleyInteractionsEstimator(
        N=N, max_order=1, min_order=1, interaction_type='SFI'
    )
    values_shapx_sfi = interaction_estimator.compute_from_samples(
        S_list=all_subsets_to_sample, game_values=game_values,
        val_empty=empty_value, val_full=full_value)
    values_shapx_sfi = copy.deepcopy(values_shapx_sfi[1])

    values_ksh = calculate_uksh_from_samples(
        game=GameWrapper(game),
        game_values=game_values,
        S_list=all_subsets_to_sample
    )

    # Original Unbiased Kernel SHAP
    u_ksh_covert, _, _, _ = CovertRegression(
        game=GameWrapper(game),
        batch_size=1,
        detect_convergence=False,
        n_samples=u_ksh_sample_size,
        paired_sampling=pairing
    )

    values_shapx_sii = [round(value, 5) for value in values_shapx_sii]
    values_shapx_sti = [round(value, 5) for value in values_shapx_sti]
    values_shapx_sfi = [round(value, 5) for value in values_shapx_sfi]
    values_ksh = [round(value, 5) for value in values_ksh]
    u_ksh_covert = [round(value, 5) for value in u_ksh_covert]

    print(f"shapx-defined-samples (sii): {values_shapx_sii} (n: {budget})\n"
          f"shapx-defined-samples (sti): {values_shapx_sti} (n: {budget})\n"
          f"shapx-defined-samples (sfi): {values_shapx_sfi} (n: {budget})\n"
          f"u-ksh-defined-samples: {values_ksh} (n: {budget})\n"
          f"u-ksh-sampling:        {u_ksh_covert} (n: {u_ksh_sample_size})")

    return values_ksh, values_shapx_sii, values_shapx_sti, values_shapx_sfi, u_ksh_covert


if __name__ == "__main__":
    from games import SparseLinearModel, NLPLookupGame, SyntheticNeuralNetwork, NLPGame, ParameterizedSparseLinearModel
    from interaction import ShapleyInteractionsEstimator
    n = 14
    #game = NLPLookupGame(n=n, sentence_id=172, set_zero=True)
    #game = NLPGame("I like this movie a lot")
    game = SyntheticNeuralNetwork(n=n,set_zero=True)
    #game = ParameterizedSparseLinearModel(n=n,weighting_scheme="uniform",n_interactions=20,max_interaction_size=5)
    N = set(range(n))

    shap = ShapleyInteractionsEstimator(N,1,1,"SII")
    #exact_values = game.exact_values(shap.weights,1,1)[1]
    exact_values = shap.compute_interactions_complete(game.set_call)[1]
    game_fun = game.set_call
    values_ksh, values_shapx_sii, values_shapx_sti, values_shapx_sfi, u_ksh_covert = compare_unbiasedksh_and_shapx(
        game=game, budget=500, pairing=False, u_ksh_sample_size=2000)
    #feature_names = game.input_sentence.split(" ")
    #print(feature_names)
    #draw_shapley_values(
    #    values_ksh, u_ksh_covert, values_shapx_sii, values_shapx_sti, values_shapx_sfi,
     #   labels=feature_names, figsize=(8, 3.5), save_name="plots/shap_comparison.png")

    print(np.sum((exact_values-values_ksh)**2))
    print(np.sum((exact_values-values_shapx_sii)**2))
    assert values_ksh==values_shapx_sii
