import copy
import random
import typing

import numpy as np
from scipy.special import binom

from .base import BaseShapleyInteractions, powerset, determine_complete_subsets


class SHAPIQEstimator(BaseShapleyInteractions):
    def __init__(self, N, max_order, min_order=1, interaction_type="SII"):
        super().__init__(N, max_order, min_order)
        self.interaction_type = interaction_type
        for t in range(0, self.n + 1):
            for s in range(min_order,max_order+1):
                for k in range(max(0, s + t - self.n), min(s, t) + 1):
                    self.weights[s][t, k] = (-1) ** (s - k) * self._kernel_m(t - k,s)
        self.inf = 1000000

    def compute_interactions_from_budget_one(self, game, budget, interaction, pairing=False, sampling_kernel="ksh", sampling_only=False):
        """Estimates the Shapley interactions given a game and budget for a single interaction

        Parameters
        ---------------------------------------
        game: set functions with real-valued outputs
        budget: budget of model evaluations and sampld subsets
        interaction: interaction for which the estimate should be computed
        pairing: if True, then subsets are sampled pairwise
        sampling_kernel: determines the subset weights for sampling
        sampling_only: if True, then sampling is enabled for all subset sizes

        """
        q, p = self._init_sampling_weights(sampling_kernel)
        result_complete = {interaction:0}

        if budget > 0:
            if sampling_only:
                #sampling for all subset sizes
                incomplete_subsets = list(range(self.n + 1))
            else:
                #sampling dependent on weights p
                complete_subsets, incomplete_subsets, budget = determine_complete_subsets(0, self.n, budget, p)

            for k in complete_subsets:
                #compute all deterministic subset sizes
                result_complete = self.update_results(result_complete, self._compute_interactions_complete_k_one(game, k, interaction))

            #Adjust budget, if pairwise sampling is used
            if pairing:
                budget = 2 * int(budget / 2)
            else:
                budget = budget

            # Sample the remaining budget and update the approximations
            if len(incomplete_subsets) > 0:
                subset_weight_vector = np.zeros(self.n+1)
                n_samples = 0
                for k in incomplete_subsets:
                    subset_weight_vector[k] = q[k]*binom(self.n, k)

                subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
                subset_sizes_samples = random.choices(incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])
                r = np.zeros(self.n+1)
                for k in incomplete_subsets:
                    r[k] = subset_weight_vector[k]/binom(self.n,k)

                result_sample_mean = {interaction:0}
                result_sample_s2 = {interaction:0}
                for k in subset_sizes_samples:
                    T = set(np.random.choice(self.n, k, replace=False))
                    result_sample_update = self._evaluate_subset_one(game, T, r[k],interaction)
                    result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                    if pairing:
                        T_c = self.N - T
                        k_c = len(T_c)
                        result_sample_update = self._evaluate_subset_one(game, T_c, r[k_c],interaction)
                        result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
        return copy.deepcopy(result_complete)

    def compute_interactions_from_budget(self, game, budget, pairing=False, sampling_kernel="ksh", sampling_only=False, stratification=False):
        """Estimates the Shapley interactions given a game and budget for all top-order interactions

        Parameters
        ---------------------------------------
        game: set functions with real-valued outputs
        budget: budget of model evaluations and sampld subsets
        pairing: if True, then subsets are sampled pairwise
        sampling_kernel: determines the subset weights for sampling
        sampling_only: if True, then sampling is enabled for all subset sizes
        stratification: if True, then stratification is used for estimation

        """
        q, p = self._init_sampling_weights(sampling_kernel)
        result_complete = self.init_results()

        if budget > 0:
            if sampling_only:
                #sampling for all subset sizes
                incomplete_subsets = list(range(self.n + 1))
            else:
                #sampling dependent on weights p
                complete_subsets, incomplete_subsets, budget = determine_complete_subsets(0, self.n, budget, p)

            for k in complete_subsets:
                #compute all deterministic subset sizes
                result_complete = self.update_results(result_complete, self._compute_interactions_complete_k(game, k))

            #Adjust budget, if pairwise sampling is used
            if pairing:
                budget = 2 * int(budget / 2)
            else:
                budget = budget

            # Sample the remaining budget and update the approximations
            if len(incomplete_subsets) > 0:
                subset_weight_vector = np.zeros(self.n+1)
                for k in incomplete_subsets:
                    subset_weight_vector[k] = q[k]*binom(self.n, k)

                subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
                subset_sizes_samples = random.choices(incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])

                if stratification:
                    r = np.zeros(self.n + 1)
                    for k in incomplete_subsets:
                        r[k] = 1 / (binom(self.n, k))

                    n_samples = {}
                    result_sample_mean = {}
                    result_sample_s2 = {}
                    self.result_sample_variance ={}

                    for k in incomplete_subsets:
                        n_samples[k] = 0
                        result_sample_mean[k] = self.init_results()
                        result_sample_s2[k] = self.init_results()

                    for k in subset_sizes_samples:
                        T = set(np.random.choice(self.n, k, replace=False))
                        result_sample_update = self._evaluate_subset(game, T, r[k])
                        result_sample_mean[k], result_sample_s2[k], n_samples[k] = self.update_mean_variance(result_sample_mean[k],
                                                                                                    result_sample_s2[k],
                                                                                                    n_samples[k],
                                                                                                    result_sample_update)
                        if pairing:
                            T_c = self.N - T
                            k_c = len(T_c)
                            result_sample_update = self._evaluate_subset(game, T_c, r[k_c])
                            result_sample_mean[k], result_sample_s2[k], n_samples[k] = self.update_mean_variance(
                                result_sample_mean[k], result_sample_s2[k], n_samples[k], result_sample_update)
                        if n_samples[k] > 1:
                            self.result_sample_variance[k] = self.scale_results(result_sample_s2[k], 1 / (n_samples[k] - 1))

                    for k in incomplete_subsets:
                        result_complete = self.update_results(result_complete,result_sample_mean[k])

                else:
                    r = np.zeros(self.n + 1)
                    for k in incomplete_subsets:
                        r[k] = subset_weight_vector[k] / binom(self.n, k)

                    n_samples = 0
                    result_sample_mean = self.init_results()
                    result_sample_s2 = self.init_results()
                    for k in subset_sizes_samples:
                        T = set(np.random.choice(self.n, k, replace=False))
                        result_sample_update = self._evaluate_subset(game, T, r[k])
                        result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                        if pairing:
                            T_c = self.N - T
                            k_c = len(T_c)
                            result_sample_update = self._evaluate_subset(game, T_c, r[k_c])
                            result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                        if n_samples>1:
                            self.result_sample_variance = self.scale_results(result_sample_s2, 1/(n_samples-1))

                    result_complete = self.update_results(result_complete, result_sample_mean)

        results_out = self._smooth_with_epsilon(result_complete)
        return copy.deepcopy(results_out)

    def compute_interactions_complete(self, game,interaction_subsets={}):
        """Computes the Exact Shapley interactions given a game (becomes computationally challenging around n = 15)."""
        results = self.init_results()
        for T in powerset(self.N):
            game_eval = game(T)
            t = len(T)
            if len(interaction_subsets) == 0:
                interaction_subsets_iterator = powerset(self.N, self.min_order, self.s_0)
            for S in interaction_subsets_iterator:
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[len(S)][t, s_t]
        result_out = copy.deepcopy(self._smooth_with_epsilon(results))
        return result_out

    def compute_efficiency(self, game):
        """ Computes efficiency value for SII """
        result = 0
        for t in range(self.s_0):
            factor = binom(self.n - t, self.s_0 - t - 1)
            sign = (-1) ** t
            for S in powerset(self.N, t, t):
                result += factor * sign * ((-1) ** self.s_0 * game(S) + game(set(self.N) - set(S)))
        return result / self.s

    def _evaluate_subset(self, game, T, p):
        """ Evaluates and weights a single subset evaluation of T sampled with probability p(T) for all interactions """
        tmp = self.init_results()
        game_eval = game(T)
        t = len(T)
        for S in powerset(self.N, self.min_order, self.s_0):
            size_intersection = len(set(S).intersection(T))
            tmp[len(S)][S] += game_eval * self.weights[len(S)][t, size_intersection] / p
        return tmp

    def _evaluate_subset_one(self, game, T, p, interaction):
        """ Evaluates and weights a single subset evaluation of T sampled with probability p(T) for a single interaction """
        tmp = {interaction:0}
        game_eval = game(T)
        t = len(T)
        size_intersection = len(set(interaction).intersection(T))
        tmp[interaction] += game_eval * self.weights[len(interaction)][t, size_intersection] / p
        return tmp


    def _init_sampling_weights(self, sampling_kernel):
        """ Initializes the subset weights for sampling for one subset of that size (q) and any subset of that size (p) """
        q = np.zeros(self.n + 1)
        p = np.zeros(self.n + 1)
        for t in range(self.n+1):
            q[t],p[t] = self._kernel_q(t, sampling_kernel)
        return q,p

    def _kernel_q(self, t, sampling_kernel):
        """ Determines the sampling weights for a subset of size t (size_weight) and the weight for any subset of size t (size_weight*binom(n,t)) """
        if sampling_kernel == "ksh":
            if t>=self.s_0 and t<=self.n-self.s_0:
                size_weight = np.math.factorial(self.n - t - self.s_0) * np.math.factorial(t - self.s_0) / np.math.factorial(
                self.n - self.s_0 + 1)
            else:
                size_weight = self.inf
        if sampling_kernel == "faith":
            if t>=1 and t<=self.n-1:
                size_weight = np.math.factorial(self.n - t - 1) * np.math.factorial(t - 1) / np.math.factorial(self.n - 1)
            else:
                size_weight = self.inf
        if sampling_kernel == "unif-size":
            size_weight = 1
        if sampling_kernel == "unif-set":
            size_weight = 1/binom(self.n, t)
        return size_weight, size_weight*binom(self.n,t)

    def _kernel_m(self, t, s):
        """ Returns the weight for each interaction type for a subset of size t and interaction of size s"""
        if self.interaction_type == "SII":
            return np.math.factorial(self.n - t - s) * np.math.factorial(t) / np.math.factorial(
                self.n - s + 1)
        if self.interaction_type == "STI":
            if s==self.s_0:
                return self.s_0 * np.math.factorial(self.n - t - 1) * np.math.factorial(t) / np.math.factorial(self.n)
            else:
                return 1.0*(t==0)
        if self.interaction_type == "SFI":
            if s == self.s_0:
                return np.math.factorial(2 * s - 1) / np.math.factorial(s - 1) ** 2 * np.math.factorial(
                    self.n - t - 1) * np.math.factorial(t + s - 1) / np.math.factorial(self.n + s - 1)
            else:
                raise ValueError("Lower order interactions are not supported.")

    def _compute_interactions_complete_k(self, game, k):
        """ Computes the SI values for all interactions over all subsets of size k for a given game """
        results = self.init_results()
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            interaction_subset_iterator = powerset(self.N, self.min_order, self.s_0)
            for S in interaction_subset_iterator:
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[len(S)][t, s_t]
        return results

    def _compute_interactions_complete_k_one(self, game, k, interaction):
        """ Computes the SI values for a single interaction over all subsets of size k for a given game """

        results = {interaction:0}
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            s_t = len(set(interaction).intersection(T))
            results[interaction] += game_eval * self.weights[len(interaction)][t, s_t]
        return results


    def compute_from_samples(
            self,
            S_list: typing.List[set],
            game_values: typing.List,
            val_empty,
            val_full,
            epsilon_correction = True
    ):
        """ Estimates all Shapley values using a list of sampled subsets for a given game

        Parameters
        ------------------------------------
        S_list: List of sampled subsets
        game_values: List of game evaluations for elements in S_list
        val_empty: model evaluated for the empty set
        val_full: model evaluated for the full set (self.N)
        epsilon_correction: if True, then yields the exact values of Unbiased KernelSHAP (only differs for sets with game({}) <> 0

        """
        subsets = copy.deepcopy(S_list)
        S_game_mapping = {tuple(S): game_value for S, game_value in zip(subsets, game_values)}
        S_game_mapping[tuple(set())] = val_empty
        S_game_mapping[tuple(self.N)] = val_full
        #Constant c_1
        baseline = (val_full-val_empty)/self.n
        #Harmonic number
        h = 0
        for k in range(1,self.n):
            h += 1/k

        epsilons = self.init_results()
        results_sample = self.init_results()
        results = self.init_results()
        for T in subsets:
            game_eval = S_game_mapping[tuple(T)]
            t = len(T)
            for S in powerset(self.N,self.min_order,self.s_0):
                s_t = len(set(S).intersection(T))
                results_sample[len(S)][S] += 2*h*game_eval*(s_t - t/self.n)
                epsilons[len(S)][S] += 2*h*val_empty*(s_t-t/self.n)

        results_sample[len(S)] = results_sample[len(S)]/len(subsets)
        epsilons[len(S)] = epsilons[len(S)]/len(subsets)
        results[len(S)] = baseline + results_sample[len(S)]

        if epsilon_correction:
            results[len(S)] = results[len(S)] - epsilons[len(S)]

        result_out = copy.deepcopy(self._smooth_with_epsilon(results))
        return result_out
