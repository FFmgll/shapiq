import copy
import random
import typing

import numpy as np
from scipy.special import binom

from .base import BaseShapleyInteractions, powerset, determine_complete_subsets


class ShapleyInteractionsEstimator(BaseShapleyInteractions):
    def __init__(self, N, max_order, min_order=1, interaction_type="SII"):
        super().__init__(N, max_order, min_order)
        self.interaction_type = interaction_type
        for t in range(0, self.n + 1):
            for k in range(max(0, self.s + t - self.n), min(self.s, t) + 1):
                self.weights[t, k] = (-1) ** (self.s - k) * self._kernel_m(t - k)
        self.last_const_complete = self.init_results()

    def compute_interactions_from_budget(self, game, budget, pairing=True, sampling_kernel="ksh"):
        """Computes the Shapley interactions given a game and budget."""
        q = self._init_sampling_weights(sampling_kernel)

        complete_subsets, incomplete_subsets, budget = determine_complete_subsets(self.s, self.n,
                                                                                  budget, q)
        result_complete = self.init_results()
        result_constant = self._constant_c(game)

        # Calculate the subsets at the edge completely and store it away
        for k in complete_subsets:
            result_complete = self.update_results(result_complete, self._compute_interactions_complete_k(game, k))
        constant_complete = self.update_results(result_constant, result_complete)
        self.last_const_complete = copy.deepcopy(self._smooth_with_epsilon(constant_complete))
        result_complete = copy.deepcopy(constant_complete)

        # Update weights for samplings for the remaining subsets
        #subset_weight_vector = np.asarray(
        #    [q[subset_size] for subset_size in incomplete_subsets])

        # Split the budget in case of pairing
        if pairing:
            total_budget = 2 * int(budget / 2)
        else:
            total_budget = budget

        # Sample the remaining budget and update the approximations
        if len(incomplete_subsets) > 0:
            #subset_weight_vector[incomplete_subsets] contains probabilities for subset sizes in incomplete_subsets
            subset_weight_vector = q/np.sum(q[incomplete_subsets])
            subset_sizes_samples = random.choices(incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])
            r = np.zeros(self.n+1)
            for k in incomplete_subsets:
                r[k] = total_budget/binom(self.n,k)
            for k in subset_sizes_samples:
                result_sample = self.init_results()
                #n_samples = int(budget * subset_weight_vector[i])
                #p[k] = subset_weight_vector / (binom(self.n, k))
                #for j in range(n_samples):  # TODO add counter here with proper weighting
                T = set(np.random.choice(self.n, k, replace=False))
                result_sample = self.update_results(result_sample, self._evaluate_subset(game, T, r[k]))
                if pairing:
                    T_c = self.N - T
                    k_c = len(T_c)
                    result_sample = self.update_results(result_sample, self._evaluate_subset(game, T_c, r[k_c]))
            result_complete = self.update_results(result_complete, result_sample)
        results_out = self._smooth_with_epsilon(result_complete)
        return copy.deepcopy(results_out)

    def compute_interactions_complete(self, game):
        """Computes the Exact Shapley interactions given a game (becomes computationally challenging around n = 15)."""
        results = self.init_results()
        for T in powerset(self.N):
            game_eval = game(T)
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        result_out = copy.deepcopy(self._smooth_with_epsilon(results))
        return result_out

    def compute_efficiency(self, game):
        result = 0
        for t in range(self.s):
            factor = binom(self.n - t, self.s - t - 1)
            sign = (-1) ** t
            for S in powerset(self.N, t, t):
                result += factor * sign * ((-1) ** self.s * game(S) + game(set(self.N) - set(S)))
        return result / self.s

    def _evaluate_subset(self, game, T, r):
        tmp = self.init_results()
        game_eval = game(T)
        t = len(T)
        for S in powerset(self.N, self.min_order, self.s):
            size_intersection = len(set(S).intersection(T))
            tmp[len(S)][S] += game_eval * self.weights[t, size_intersection] / r
        return tmp

    def _constant_c(self, game):
        rslt = self.init_results()
        for T in powerset(self.N, 0, self.s - 1):
            game_val = game(T)
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s):
                rslt[len(S)][S] += game_val * self.weights[t, len(set(S).intersection(T))]

        for T in powerset(self.N, self.n - self.s + 1, self.n):
            game_val = game(T)
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s):
                rslt[len(S)][S] += game_val * self.weights[t, len(set(S).intersection(T))]
        return rslt

    def _init_sampling_weights(self, sampling_kernel):
        q = np.zeros(self.n + 1)
        for t in range(self.s, self.n - self.s + 1):
            q[t] = self._kernel_q(t, sampling_kernel)
        return q

    def _kernel_q(self, t, sampling_kernel):
        if sampling_kernel == "ksh":
            return np.math.factorial(self.n - t - self.s) * np.math.factorial(t - self.s) / np.math.factorial(
                self.n - self.s + 1)
        if sampling_kernel == "faith":
            return np.math.factorial(self.n - t - 1) * np.math.factorial(t - 1) / np.math.factorial(self.n - 1)
        if sampling_kernel == "unif-size":
            return 1
        if sampling_kernel == "unif-set":
            return binom(self.n, t)

    def _kernel_m(self, t):
        if self.interaction_type == "SII":
            return np.math.factorial(self.n - t - self.s) * np.math.factorial(t) / np.math.factorial(
                self.n - self.s + 1)
        if self.interaction_type == "STI":
            return self.s * np.math.factorial(self.n - t - 1) * np.math.factorial(t) / np.math.factorial(self.n)
        if self.interaction_type == "SFI":
            return np.math.factorial(2 * self.s - 1) / np.math.factorial(self.s - 1) ** 2 * np.math.factorial(
                self.n - t - 1) * np.math.factorial(t + self.s - 1) / np.math.factorial(self.n + self.s - 1)

    def _compute_interactions_complete_k(self, game, k):
        results = self.init_results()
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        return results

    def compute_from_samples(
            self,
            S_list: typing.List[set],
            game_values: typing.List,
            val_empty,
            val_full
    ):
        subsets = copy.deepcopy(S_list)
        S_game_mapping = {tuple(S): game_value for S, game_value in zip(subsets, game_values)}
        S_game_mapping[tuple(set())] = val_empty
        S_game_mapping[tuple(self.N)] = val_full
        subsets.append(set())
        subsets.append(set(self.N))

        results = self.init_results()
        for T in subsets:
            game_eval = S_game_mapping[tuple(T)]
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        result_out = copy.deepcopy(self._smooth_with_epsilon(results))
        return result_out
