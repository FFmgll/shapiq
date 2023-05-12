import copy
import random

import numpy as np
from scipy.special import binom
from typing import Dict

from approximators import BaseShapleyInteractions
from approximators.base import determine_complete_subsets, powerset


def _init_sampling_weights(n_players: int, interaction_order: int):
    weights_q: np.ndarray = np.zeros(n_players + 1)
    weights_p: np.ndarray = np.zeros(n_players + 1)
    for t in range(n_players + 1):
        weights_q[t] = _get_ksh_kernel_value(t, n_players, interaction_order)
        weights_p[t] = weights_q[t] * binom(n_players, t)
    return weights_q, weights_p


def _get_ksh_kernel_value(subset_size: int, n_players: int, interaction_order: int,
                          big_m: int = 1_000_000):
    if interaction_order <= subset_size <= n_players - interaction_order:
        size_weight = np.math.factorial(n_players - subset_size - interaction_order)
        size_weight *= np.math.factorial(subset_size - interaction_order)
        size_weight /= np.math.factorial(n_players - interaction_order + 1)
    else:
        size_weight = big_m
    return size_weight


class SHAPIQEstimator(BaseShapleyInteractions):

    def __init__(self, player_names: set, max_order: int, min_order: int = 1,
                 interaction_type: str = "SII"):
        assert interaction_type in ["SII", "STI", "FSI"], ValueError("interaction_type must be "
                                                                     "one of 'SII', 'STI', 'FSI'")
        self.interaction_type = "SII"

        super().__init__(N=player_names, max_order=max_order, min_order=min_order)
        self.s = self.s_0
        weights_q, weights_p = _init_sampling_weights(n_players=self.n, interaction_order=self.s)
        self.weights_q = weights_q
        self.weights_p = weights_p

        self.weights = np.zeros((self.n + 1, self.s + 1))
        for t in range(0, self.n + 1):
            for k in range(max(0, self.s + t - self.n), min(self.s, t) + 1):
                self.weights[t, k] = (-1) ** (self.s - k) * self._kernel_m(t - k)

        self.interaction_values: Dict[int, np.ndarray] = self.init_results()

    def _kernel_m(self, t):
        """ Returns the weight for each interaction type for a subset of size t """
        if self.interaction_type == "SII":
            weight_value = np.math.factorial(self.n - t - self.s)
            weight_value *= np.math.factorial(t)
            weight_value /= np.math.factorial(self.n - self.s + 1)
        elif self.interaction_type == "STI":
            weight_value = np.math.factorial(self.n - t - 1)
            weight_value *= np.math.factorial(t)
            weight_value /= np.math.factorial(self.n)
        elif self.interaction_type == "FSI":
            weight_value = np.math.factorial(2 * self.s - 1)
            weight_value /= np.math.factorial(self.s - 1) ** 2
            weight_value *= np.math.factorial(self.n - t - 1)
            weight_value *= np.math.factorial(t + self.s - 1)
            weight_value /= np.math.factorial(self.n + self.s - 1)
        else:
            raise ValueError("interaction_type must be one of 'SII', 'STI', 'FSI'")
        return weight_value

    def _compute_interactions_complete_k(self, game, k):
        """ Computes the SI values for all interactions over all subsets of size k for a given game """
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            interaction_subset_iterator = powerset(self.N, self.min_order, self.s)
            for S in interaction_subset_iterator:
                s_t = len(set(S).intersection(T))
                self.interaction_values[len(S)][S] += game_eval * self.weights[t, s_t]

    def _compute_bernoulli_numbers(self, n: int):
        """ Computes the Bernoulli numbers up to order n and returns a list of length n+1"""
        bernoulli_numbers = np.zeros(n + 1)
        bernoulli_numbers[0] = 1
        for k in range(1, n):
            bernoulli_number = -1 / (k + 1) * np.sum(
                [binom(k + 1, j) * bernoulli_numbers[j] for j in range(k)])
            bernoulli_numbers[k] = bernoulli_number
        return bernoulli_numbers

    def compute_interactions_from_budget(self, game, budget: int):
        # get sampling order
        subsets = determine_complete_subsets(s=0, n=self.n, budget=budget, q=self.weights_p)
        complete_subsets, incomplete_subsets, budget = subsets

        # compute all deterministic subset sizes
        for k in complete_subsets:
            self._compute_interactions_complete_k(game=game, k=k)

        # if no sampling required (rarely the case)
        if len(incomplete_subsets) == 0:
            return copy.deepcopy(self._smooth_with_epsilon(self.interaction_values))

        # re-weighting and normalization of the probability vector
        subset_weight_vector = np.zeros(self.n + 1)
        for k in incomplete_subsets:
            subset_weight_vector[k] = self.weights_q[k] * binom(self.n, k)
        subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
        sampled_subset_sizes = random.choices(
            incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])

        r = np.zeros(self.n + 1)
        for k in incomplete_subsets:
            r[k] = subset_weight_vector[k] / binom(self.n, k)

        sampled_interactions = self.init_results()
        n_subsets_sampled = 0
        for subset_size in sampled_subset_sizes:

            T = set(np.random.choice(self.n, subset_size, replace=False))
            n_subsets_sampled += 1

            game_eval = game(T)
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s):
                size_intersection = len(set(S).intersection(T))
                interaction_update = game_eval * self.weights[t, size_intersection] / r[subset_size]
                sampled_interactions[len(S)][S] += interaction_update

        for interaction_order in range(self.min_order, self.s + 1):
            sampled_interactions[interaction_order] /= n_subsets_sampled
            self.interaction_values[interaction_order] += sampled_interactions[interaction_order]

        return copy.deepcopy(self._smooth_with_epsilon(self.interaction_values))

    def transform_interactions_in_n_shapley(self, n: int = None,
                                            reduce_one_dimension: bool = False):
        """Computes the n-Shapley values from the interaction values

        Args:
            n (int, optional): The order of the Shapley values. Defaults to None.
            reduce_one_dimension (bool, optional): If True, the n-Shapley values are reduced to one dimension. Defaults to False.

        Returns:
            dict: A dictionary containing the n-Shapley values
        """
        if n is None:
            n = self.s
        bernoulli_numbers = self._compute_bernoulli_numbers(n=n + 1)
        result = {}
        # all subsets S with 1 <= |S| <= n
        for S in powerset(self.N, min_size=1, max_size=n):
            # get un-normalized interaction value (delta_S(x))
            S_effect = self.interaction_values[len(S)][S]
            subset_size = len(S)
            # go over all subsets T of length |S| + 1, ..., n that contain S
            for T in powerset(self.N, min_size=subset_size + 1, max_size=n):
                if not set(S).issubset(T):
                    continue
                # get the effect of T
                T_effect = self.interaction_values[len(T)][T]
                # normalization with bernoulli numbers
                S_effect = S_effect + bernoulli_numbers[len(T) - subset_size] * T_effect
            result[S] = S_effect
        if not reduce_one_dimension:
            return result
        return self._convert_n_shapley_values_to_one_dimension(result, n=n)

    def _convert_n_shapley_values_to_one_dimension(self, n_shapley_values, n: int = None):
        """Converts the n-Shapley values to one dimension"""
        if n is None:
            n = max((len(k), _) for k, v in n_shapley_values.iteritems())
        result_pos = {order: {player: 0. for player in range(self.n)} for order in range(1, n + 1)}
        result_neg = {order: {player: 0. for player in range(self.n)} for order in range(1, n + 1)}
        for S, n_shap_value in n_shapley_values.items():
            for player in S:
                if n_shap_value > 0:
                    result_pos[len(S)][player] += n_shap_value / len(S)
                if n_shap_value < 0:
                    result_neg[len(S)][player] += n_shap_value / len(S)
        return result_pos, result_neg
