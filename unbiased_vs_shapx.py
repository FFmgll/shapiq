import numpy as np
import itertools
import random

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


def get_S_and_game(budget, num_players, weight_vector, N, pairing):
    complete_subsets, incomplete_subsets, budget = determine_complete_subsets(
        budget=budget, n=num_players, s=1, q=weight_vector)

    all_subsets_to_sample = []

    for complete_subset in complete_subsets:
        combinations = itertools.combinations(N, complete_subset)
        for subset in combinations:
            subset = set(subset)
            all_subsets_to_sample.append(subset)

    remaining_weight = weight_vector[incomplete_subsets] / sum(weight_vector[incomplete_subsets])

    if len(incomplete_subsets) > 0:
        sampled_subsets = set()
        while len(sampled_subsets) < budget:
            subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
            ids = np.random.choice(num_players, size=subset_size, replace=False)
            sampled_subsets.add(tuple(sorted(ids)))
            if pairing:
                if len(sampled_subsets) < budget:
                    sampled_subsets.add(tuple(N - set(ids)))
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
        budget, num_players, weight_vector, N, pairing)

    # Interaction estimator
    interaction_estimator = ShapleyInteractionsEstimator(
        N=N, max_order=1, min_order=1, interaction_type='SFI'
    )

    values_shapx = interaction_estimator.compute_from_samples(
        S_list=all_subsets_to_sample, game_values=game_values,
        val_empty=empty_value, val_full=full_value)
    values_shapx = values_shapx[1]

    values_ksh = calculate_uksh_from_samples(
        game=GameWrapper(game),
        game_values=game_values,
        S_list=all_subsets_to_sample
    )

    # Original Unbiased Kernel SHAP
    u_ksh_covert, _, _, _ = CovertRegression(
        game=GameWrapper(game),
        batch_size=100,
        detect_convergence=False,
        n_samples=u_ksh_sample_size,
        paired_sampling=pairing
    )

    values_shapx = [round(value, 5) for value in values_shapx]
    values_ksh = [round(value, 5) for value in values_ksh]
    u_ksh_covert = [round(value, 5) for value in u_ksh_covert]

    print(f"Shapx:          {values_shapx}\n"
          f"unbiased-ksh:   {values_ksh}\n"
          f"covert-u-ksh:   {u_ksh_covert}")

    return values_ksh, values_shapx


if __name__ == "__main__":
    from games import SparseLinearModel
    game = SparseLinearModel(n=10, n_interactions_per_order=None, n_non_important_features=3)
    game_fun = game.set_call
    values_ksh, values_shapx = compare_unbiasedksh_and_shapx(
        game=game, budget=500, pairing=True, u_ksh_sample_size=5_000)
    game_values = game.interaction_matrices[1]
