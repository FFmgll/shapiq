"""This module contains the RegressionEstimator class, which is used to approximate the FSI interaction index using the weighted least square approach."""
import copy
import itertools
import random

import numpy as np
from scipy.special import binom

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
        min_order = 1
        super().__init__(N, max_order, min_order=min_order)
        self._big_M = float(10_000_000)
        self.interaction_type = 'FSI'

    def _init_sampling_weights(self):
        weight_vector = np.zeros(shape=self.n - 1)
        for subset_size in range(1, self.n):
            weight_vector[subset_size - 1] = (self.n - 1) / (subset_size * (self.n - subset_size))
        sampling_weight = (np.asarray([0] + [*weight_vector] + [0]))/ sum(weight_vector)
        return sampling_weight

    @staticmethod
    def get_S_and_values(budget, num_players, weight_vector, N, pairing, game_fun):
        complete_subsets, incomplete_subsets, budget = determine_complete_subsets(
            budget=budget, n=num_players, s=1, q=weight_vector)

        all_subsets_to_sample = []
        kernel_weights = {}

        for complete_subset in complete_subsets:
            combinations = itertools.combinations(N, complete_subset)
            for subset in combinations:
                subset = set(subset)
                all_subsets_to_sample.append(subset)
                kernel_weights[tuple(sorted(subset))] = weight_vector[len(subset)] / binom(num_players, len(subset))

        remaining_weight = weight_vector[incomplete_subsets] / sum(
            weight_vector[incomplete_subsets])
        kernel_weights_sampling = {}

        if len(incomplete_subsets) > 0:
            sampled_subsets = set()
            n_sampled_subsets = 0
            while len(sampled_subsets) < budget:
                subset_size = random.choices(incomplete_subsets, remaining_weight, k=1)
                ids = np.random.choice(num_players, size=subset_size, replace=False)
                sampled_subset = tuple(sorted(ids))
                if sampled_subset not in sampled_subsets:
                    sampled_subsets.add(sampled_subset)
                    kernel_weights_sampling[sampled_subset] = 1.
                else:
                    kernel_weights_sampling[sampled_subset] += 1.
                n_sampled_subsets += 1
                if pairing:
                    if len(sampled_subsets) < budget:
                        sampled_subset_paired = tuple(sorted(set(N) - set(ids)))
                        if sampled_subset_paired not in sampled_subsets:
                            sampled_subsets.add(sampled_subset_paired)
                            kernel_weights_sampling[sampled_subset_paired] = 1.
                        else:
                            kernel_weights_sampling[sampled_subset_paired] += 1.
                        n_sampled_subsets += 1
            for subset in sampled_subsets:
                all_subsets_to_sample.append(set(subset))

            # re-normalize kernel weights
            weight_left = np.sum(weight_vector[incomplete_subsets])
            kernel_weights_sampling = {subset: weight * (weight_left / n_sampled_subsets) for
                                       subset, weight in kernel_weights_sampling.items()}
            kernel_weights.update(kernel_weights_sampling)

        game_values = [game_fun(subset) for subset in all_subsets_to_sample]
        return all_subsets_to_sample, game_values, kernel_weights

    def approximate_with_budget(self, game_fun, budget, pairing: bool = True):
        sampling_weight = self._init_sampling_weights()

        S_list, game_values, kernel_weights = self.get_S_and_values(
            budget, self.n, sampling_weight, self.N, pairing, game_fun)

        empty_value = game_fun({})
        full_value = game_fun(self.N)
        S_list.append(set())
        S_list.append(self.N)
        game_values.append(empty_value)
        game_values.append(full_value)
        kernel_weights[()] = self._big_M
        kernel_weights[tuple(self.N)] = self._big_M

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
        for s in range(1, self.s_0 + 1):
            num_players += int(binom(self.n, s))

        i = 0
        player_indices = {}
        player_indices_inv = {}
        for combination in powerset(self.N, max_size=self.s_0, min_size=1):
            player_indices[combination] = i
            player_indices_inv[i] = combination
            i += 1

        N_arr = np.arange(0, self.n)
        W = np.zeros(shape=game_values.shape, dtype=float)
        new_S = np.zeros(shape=(len(S_list), num_players), dtype=bool)
        for i, S in enumerate(all_S):
            S = N_arr[S]
            W[i] = kernel_weights[tuple(S)]
            for s in range(1, self.s_0 + 1):
                for combination in itertools.combinations(S, s):
                    index = player_indices[combination]
                    new_S[i, index] = 1

        A = new_S
        B = game_values
        W = np.sqrt(np.diag(W))
        Aw = np.dot(W, A)
        Bw = np.dot(B, W)
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)

        #result = np.zeros(np.repeat(self.n, self.s_0), dtype=float)

        #for i in range(len(phi)):
        #    combination = player_indices_inv[i]
        #    if len(combination) != self.s_0:
        #        continue
        #   result[combination] = phi[i]

        result = self.init_results()

        for i in range(len(phi)):
            combination = player_indices_inv[i]
            result[len(combination)][combination] = phi[i]

        return copy.deepcopy(self._smooth_with_epsilon(result))


    def compute_exact_values(self, game_fun):
        S_list = []
        game_values = []
        kernel_weights = {}

        sampling_weight = self._init_sampling_weights()
        #scale sampling weights to kernel_weights
        kernel_size_weights = np.zeros(self.n+1)
        for i in range(1,self.n):
            kernel_size_weights[i] = sampling_weight[i] /binom(self.n,i)

        for T in powerset(self.N,1,self.n-1):
            S_list.append(set(T))
            game_values.append(game_fun(T))
            kernel_weights[T] = kernel_size_weights[len(T)]

        empty_value = game_fun({})
        full_value = game_fun(self.N)
        S_list.append(set())
        S_list.append(self.N)
        game_values.append(empty_value)
        game_values.append(full_value)
        kernel_weights[()] = self._big_M
        kernel_weights[tuple(self.N)] = self._big_M

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
        for s in range(1, self.s_0 + 1):
            num_players += int(binom(self.n, s))

        i = 0
        player_indices = {}
        player_indices_inv = {}
        for combination in powerset(self.N, max_size=self.s_0, min_size=1):
            player_indices[combination] = i
            player_indices_inv[i] = combination
            i += 1

        N_arr = np.arange(0, self.n)
        W = np.zeros(shape=game_values.shape, dtype=float)
        new_S = np.zeros(shape=(len(S_list), num_players), dtype=bool)
        for i, S in enumerate(all_S):
            S = N_arr[S]
            W[i] = kernel_weights[tuple(S)]
            for s in range(1, self.s_0 + 1):
                for combination in itertools.combinations(S, s):
                    index = player_indices[combination]
                    new_S[i, index] = 1

        A = new_S
        B = game_values
        W = np.sqrt(np.diag(W))
        Aw = np.dot(W, A)
        Bw = np.dot(B, W)
        phi, residuals, rank, singular_values = np.linalg.lstsq(Aw, Bw, rcond=None)

        #result = np.zeros(np.repeat(self.n, self.s_0), dtype=float)
        result = self.init_results()

        for i in range(len(phi)):
            combination = player_indices_inv[i]
            result[len(combination)][combination] = phi[i]

        return copy.deepcopy(self._smooth_with_epsilon(result))

