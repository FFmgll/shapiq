import copy
import random
import typing

import numpy as np
import setuptools
from scipy.special import binom


from .base import BaseShapleyInteractions, powerset, determine_complete_subsets


class ShapleyInteractionsEstimator(BaseShapleyInteractions):
    def __init__(self, N, max_order, min_order=1, interaction_type="SII"):
        super().__init__(N, max_order, min_order)
        self.epsilon_sampling = 1
        self.interaction_type = interaction_type
        for t in range(0, self.n + 1):
            for k in range(max(0, self.s + t - self.n), min(self.s, t) + 1):
                self.weights[t, k] = (-1) ** (self.s - k) * self._kernel_m(t - k)
        self.inf = 1000000

    def compute_interactions_from_budget_one(self, game, budget, interaction, pairing=False, sampling_kernel="ksh", sampling_only=False):
        """Computes the Shapley interactions given a game and budget for selected interactions"""
        q, p = self._init_sampling_weights(sampling_kernel)

        self.last_sampling_params = {}
        self.last_sampling_params["q"] = q
        self.last_sampling_params["p"] = p
        self.last_sampling_params["sampling"] = False
        self.last_sampling_params["average_std"] = 0
        self.last_sampling_params["std_threshold"] = 0

        result_complete = {interaction:0}

        if budget > 0:
            if sampling_only:
                #sampling for all subset sizes
                incomplete_subsets = list(range(self.n + 1))
            else:
                #sampling dependent on weights p
                complete_subsets, incomplete_subsets, budget = determine_complete_subsets(0, self.n, budget, p)

            self.last_sampling_params["complete_subsets"] = complete_subsets
            self.last_sampling_params["incomplete_subsets"] = incomplete_subsets

            for k in complete_subsets:
                #compute all deterministic subset sizes
                result_complete = self.update_results(result_complete, self._compute_interactions_complete_k_one(game, k, interaction))

            self.result_no_sampling = copy.deepcopy(result_complete)
            #Adjust budget, if pairwise sampling is used
            if pairing:
                budget = 2 * int(budget / 2)
            else:
                budget = budget

            self.last_sampling_params["sampling_budget"] = budget
            # Sample the remaining budget and update the approximations
            if len(incomplete_subsets) > 0:
                subset_weight_vector = np.zeros(self.n+1)
                n_samples = 0
                for k in incomplete_subsets:
                    subset_weight_vector[k] = q[k]*binom(self.n, k)

                subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
                subset_sizes_samples = random.choices(incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])
                r = np.zeros(self.n+1)
                self.last_sampling_params["subset_weight_vector"] = subset_weight_vector
                for k in incomplete_subsets:
                    r[k] = subset_weight_vector[k]/binom(self.n,k)

                self.last_sampling_params["p"] = p
                result_sample_mean = {interaction:0}
                result_sample_s2 = {interaction:0}
                for k in subset_sizes_samples:
                    T = set(np.random.choice(self.n, k, replace=False))
                    result_sample_update = self._evaluate_subset_one(game, T, r[k],interaction)
                    self.last_sampling_params["update"] = result_sample_update
                    result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                    if pairing:
                        T_c = self.N - T
                        k_c = len(T_c)
                        result_sample_update = self._evaluate_subset_one(game, T_c, r[k_c],interaction)
                        result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                    self.result_sample_mean = result_sample_mean
        return copy.deepcopy(result_complete)

    def compute_interactions_from_budget(self, game, budget, pairing=False, sampling_kernel="ksh", sampling_only=False):
        """Computes the Shapley interactions given a game and budget."""
        q, p = self._init_sampling_weights(sampling_kernel)

        self.last_sampling_params = {}
        self.last_sampling_params["q"] = q
        self.last_sampling_params["p"] = p
        self.last_sampling_params["sampling"] = True
        self.last_sampling_params["average_std"] = 0
        self.last_sampling_params["std_threshold"] = 0

        result_complete = self.init_results()
        self.result_no_sampling = self.init_results()
        self.average_std_S = 0

        if budget > 0:
            if sampling_only:
                #sampling for all subset sizes
                incomplete_subsets = list(range(self.n + 1))
            else:
                #sampling dependent on weights p
                complete_subsets, incomplete_subsets, budget = determine_complete_subsets(0, self.n, budget, p)

            self.last_sampling_params["complete_subsets"] = complete_subsets
            self.last_sampling_params["incomplete_subsets"] = incomplete_subsets

            for k in complete_subsets:
                #compute all deterministic subset sizes
                result_complete = self.update_results(result_complete, self._compute_interactions_complete_k(game, k))
            self.result_no_sampling = copy.deepcopy(result_complete)

            #Adjust budget, if pairwise sampling is used
            if pairing:
                budget = 2 * int(budget / 2)
            else:
                budget = budget

            self.last_sampling_params["sampling_budget"] = budget
            # Sample the remaining budget and update the approximations
            if len(incomplete_subsets) > 0:
                subset_weight_vector = np.zeros(self.n+1)
                n_samples = 0
                for k in incomplete_subsets:
                    subset_weight_vector[k] = q[k]*binom(self.n, k)

                subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
                subset_sizes_samples = random.choices(incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])
                r = np.zeros(self.n+1)
                self.last_sampling_params["subset_weight_vector"] = subset_weight_vector
                for k in incomplete_subsets:
                    r[k] = subset_weight_vector[k]/binom(self.n,k)

                self.last_sampling_params["p"] = p
                result_sample_mean = self.init_results()
                result_sample_s2 = self.init_results()
                for k in subset_sizes_samples:
                    T = set(np.random.choice(self.n, k, replace=False))
                    result_sample_update = self._evaluate_subset(game, T, r[k])
                    self.last_sampling_params["update"] = result_sample_update
                    result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                    if pairing:
                        T_c = self.N - T
                        k_c = len(T_c)
                        result_sample_update = self._evaluate_subset(game, T_c, r[k_c])
                        result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(result_sample_mean,result_sample_s2,n_samples,result_sample_update)
                    if n_samples>1:
                        self.result_sample_variance = self.scale_results(result_sample_s2, 1/(n_samples-1))
                    self.result_sample_mean = result_sample_mean

                #result_complete = self.update_results(result_complete, result_sample_mean)
                self.average_std_S = np.sum(np.sqrt(self.result_sample_variance[self.s]))/binom(self.n,self.s)
                self.average_variance = (np.sum(self.result_sample_variance[self.s])/binom(self.n,self.s))/n_samples
                self.average_squared_mean = (np.sum(self.result_sample_mean[self.s]**2)/binom(self.n,2))
                self.max_value = np.max(self.result_sample_mean[self.s])
                self.min_value = np.min(self.result_sample_mean[self.s])
                self.std_threshold = np.abs(self.max_value - self.min_value)
                self.last_sampling_params["std_threshold"] = self.std_threshold
                self.average_std = np.sqrt(self.average_variance)
                self.last_sampling_params["average_std"] = self.average_std

                result_complete = self.update_results(result_complete, result_sample_mean)

                #if self.average_std < self.std_threshold*self.epsilon_sampling:
                    #result_complete = self.update_results(result_complete, result_sample_mean)
                    #print("sampling used")
                    #self.last_sampling_params["sampling"] = True
                #else:
                    #self.last_sampling_params["sampling"] = False
                    #print("sampling not used")



            self.last_sampling_params["complete"] = result_complete
        results_out = self._smooth_with_epsilon(result_complete)
        return copy.deepcopy(results_out)

    def compute_interactions_complete(self, game,interaction_subsets={}):
        """Computes the Exact Shapley interactions given a game (becomes computationally challenging around n = 15)."""
        results = self.init_results()
        for T in powerset(self.N):
            game_eval = game(T)
            t = len(T)
            if len(interaction_subsets) == 0:
                interaction_subsets_iterator = powerset(self.N, self.min_order, self.s)
            for S in interaction_subsets_iterator:
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

    def _evaluate_subset(self, game, T, p):
        tmp = self.init_results()
        game_eval = game(T)
        t = len(T)
        for S in powerset(self.N,self.min_order,self.s):
            size_intersection = len(set(S).intersection(T))
            tmp[len(S)][S] += game_eval * self.weights[t, size_intersection] / p
        return tmp

    def _evaluate_subset_one(self, game, T, p, interaction):
        tmp = {interaction:0}
        game_eval = game(T)
        t = len(T)
        size_intersection = len(set(interaction).intersection(T))
        tmp[interaction] += game_eval * self.weights[t, size_intersection] / p
        return tmp


    def _init_sampling_weights(self, sampling_kernel):
        q = np.zeros(self.n + 1)
        p = np.zeros(self.n + 1)
        for t in range(self.n+1):
            q[t],p[t] = self._kernel_q(t, sampling_kernel)
        return q,p

    def _kernel_q(self, t, sampling_kernel):
        if sampling_kernel == "ksh":
            if t>=self.s and t<=self.n-self.s:
                size_weight = np.math.factorial(self.n - t - self.s) * np.math.factorial(t - self.s) / np.math.factorial(
                self.n - self.s + 1)
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
            interaction_subset_iterator = powerset(self.N, self.min_order, self.s)
            for S in interaction_subset_iterator:
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        return results

    def _compute_interactions_complete_k_one(self, game, k, interaction):
        results = {interaction:0}
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            s_t = len(set(interaction).intersection(T))
            results[interaction] += game_eval * self.weights[t, s_t]
        return results


    def compute_from_samples(
            self,
            S_list: typing.List[set],
            game_values: typing.List,
            val_empty,
            val_full,
            epsilon_correction = True
    ):
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
            for S in powerset(self.N,self.min_order,self.s):
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
