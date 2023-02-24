import copy
import itertools
import random

import numpy as np
from scipy.special import binom

from games import NLPLookupGame
from approximators import BaseShapleyInteractions
from approximators.base import determine_complete_subsets, powerset


def get_weights(num_players):
    weights = np.arange(1, num_players)
    weights = 1 / (weights * (num_players - weights))
    weights = weights / np.sum(weights)
    return weights


class RegressionEstimator(BaseShapleyInteractions):
    """ Estimates the SI (for FSI) using the weighted least square approach """
    def __init__(self, N, max_order):
        min_order = max_order
        super().__init__(N, max_order, min_order=min_order)
        self._big_M = float(10_000_000)
        self.interaction_type = 'SFI'

    def _init_sampling_weights(self):
        weight_vector = np.zeros(shape=self.n - 1)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (binom(self.n, subset_size) * subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0])) / sum(weight_vector)
        regression_weights = (np.asarray([self._big_M] + [*weight_vector] + [self._big_M]))
        return sampling_weight, regression_weights

    @staticmethod
    def get_S_and_values(budget, num_players, weight_vector, N, pairing, game_fun):
        complete_subsets, incomplete_subsets, budget = determine_complete_subsets(
            budget=budget, n=num_players, s=1, q=weight_vector)

        all_subsets_to_sample = []

        for complete_subset in complete_subsets:
            combinations = itertools.combinations(N, complete_subset)
            for subset in combinations:
                subset = set(subset)
                all_subsets_to_sample.append(subset)

        remaining_weight = weight_vector[incomplete_subsets] / sum(
            weight_vector[incomplete_subsets])

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

    def approximate_with_budget(self, game_fun, budget, pairing: bool = True):
        sampling_weight, regression_weights = self._init_sampling_weights()

        S_list, game_values = self.get_S_and_values(
            budget, self.n, sampling_weight, self.N, pairing, game_fun)

        empty_value = game_fun({})
        full_value = game_fun(self.N)

        S_list.append(set())
        S_list.append(self.N)

        game_values.append(empty_value)
        game_values.append(full_value)

        # transform s and v into np.ndarrays
        all_S = np.zeros(shape=(len(S_list), self.n), dtype=bool)
        for i, subset in enumerate(S_list):
            if len(subset) == 0:
                continue
            subset = np.asarray(list(subset))
            all_S[i, subset] = 1
        game_values = np.asarray(game_values)
        game_values = game_values - empty_value

        num_players: int = 0
        for s in range(1, self.s + 1):
            num_players += int(binom(self.n, s))

        i = 0
        player_indices = {}
        player_indices_inv = {}
        for combination in powerset(self.N, max_size=self.s, min_size=1):
            player_indices[combination] = i
            player_indices_inv[i] = combination
            i += 1

        N_arr = np.arange(0, self.n)
        W = np.zeros(shape=game_values.shape, dtype=float)
        new_S = np.zeros(shape=(len(S_list), num_players), dtype=bool)
        for i, S in enumerate(all_S):
            S = N_arr[S]
            W[i] = regression_weights[len(S)]
            for s in range(1, self.s + 1):
                for combination in itertools.combinations(S, s):
                    index = player_indices[combination]
                    new_S[i, index] = 1

        A = new_S
        B = game_values
        W = np.sqrt(np.diag(W))
        Aw = np.dot(W, A)
        Bw = np.dot(B, W)
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)

        result = np.zeros(np.repeat(self.n, self.s), dtype=float)

        for i in range(len(phi)):
            combination = player_indices_inv[i]
            if len(combination) != self.s:
                continue
            result[combination] = phi[i]

        return copy.deepcopy(self._smooth_with_epsilon(result))


if __name__ == "__main__":
    from approximators.shapiq import SHAPIQEstimator

    n = 14
    #game = ParameterizedSparseLinearModel(n=n, weighting_scheme="uniform", n_interactions=4, max_interaction_size=2, min_interaction_size=2)
    #game_fun = game.set_call

    game = NLPLookupGame(n=n)
    game_fun = game.set_call


    N = set(range(0, n))

    est_1 = SHAPIQEstimator(N, 2, 2, "SFI")

    exact_values = copy.deepcopy(
        est_1.compute_interactions_complete(game_fun)
    )


    est = RegressionEstimator(N, 2)
    phi = est.approximate_with_budget(game_fun=game_fun, budget=2**10)

    #exact_values = game.exact_values(est_1.weights, 2, 2)

    np.sum((exact_values[2] - phi) ** 2)