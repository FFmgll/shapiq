"""This module implements the SHAPIQ estimator for Shapley interactions for any SI interaction index."""
import copy
import random

import numpy as np
from scipy.special import binom, bernoulli
from typing import Dict, Tuple, List, Union, Callable
from tqdm import tqdm

from .base import BaseShapleyInteractions, powerset, determine_complete_subsets


class SHAPIQEstimator(BaseShapleyInteractions):
    """Estimates the Shapley interactions using the SHAPIQ estimator"""

    def __init__(self, N, order, interaction_type="SII", top_order: bool = True):
        min_order = order if top_order else 1
        super().__init__(N, order, min_order)
        self.interaction_type = interaction_type
        for t in range(0, self.n + 1):
            for s in range(min_order, order + 1):
                for k in range(max(0, s + t - self.n), min(s, t) + 1):
                    self.weights[s][t, k] = (-1) ** (s - k) * self._kernel_m(t - k, s)
        self.inf = 1000000
        self.interaction_values = self.init_results()

    def compute_interactions_from_budget(
            self,
            game: Callable[[set], float],
            budget: int,
            pairing: bool = False,
            sampling_kernel: str = "ksh",
            only_sampling: bool = False,
            stratification: bool = False,
            show_pbar: bool = False,
            only_expicit: bool = False
    ) -> Dict[int, np.ndarray]:
        """Estimates the Shapley interactions given a game and budget for all top-order interactions

        Args:
            game (Callable[[set], float]): set functions with real-valued outputs
            budget (int): budget of model evaluations to use
            pairing (bool, optional): if True, then subsets are sampled pairwise. Defaults to False.
            sampling_kernel (str, optional): determines the subset weights for sampling.
                Defaults to "ksh".
            only_sampling (bool, optional): if True, then sampling is enabled for all subset sizes
                complete subsets are not calculated explicitly. Defaults to False.
            stratification (bool, optional): if True, then stratification is used for sampling.
                Defaults to False.
            show_pbar (bool, optional): if True, then a progress bar is shown. Defaults to False.
            only_expicit (bool, optional): if True, then only explicit interactions are calculated.
                Defaults to False.

        Returns:
            Dict[int, np.ndarray]: dictionary of interactions and their estimates
        """
        start_budget = budget
        pbar = None
        if show_pbar:
            pbar = tqdm(total=start_budget)

        q, p = self._init_sampling_weights(sampling_kernel)
        result_complete = self.init_results()

        if budget > 0:
            if only_sampling:
                # sampling for all subset sizes
                incomplete_subsets = list(range(self.n + 1))
            else:
                # sampling dependent on weights p
                complete_subsets, incomplete_subsets, budget = determine_complete_subsets(0, self.n,
                                                                                          budget, p)

            for k in complete_subsets:
                # compute all deterministic subset sizes
                result_complete = self.update_results(
                    result_complete, self._compute_interactions_complete_k(game, k, pbar=pbar))

            # Adjust budget, if pairwise sampling is used
            if pairing:
                budget = int(budget / 2)
            else:
                budget = budget

            if only_expicit:
                incomplete_subsets = []
            # Sample the remaining budget and update the approximations
            if len(incomplete_subsets) > 0:
                subset_weight_vector = np.zeros(self.n + 1)
                for k in incomplete_subsets:
                    subset_weight_vector[k] = q[k] * binom(self.n, k)

                subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
                subset_sizes_samples = random.choices(incomplete_subsets, k=budget,
                                                      weights=subset_weight_vector[
                                                          incomplete_subsets])

                if stratification:
                    r = np.zeros(self.n + 1)
                    for k in incomplete_subsets:
                        r[k] = 1 / (binom(self.n, k))

                    n_samples = {}
                    result_sample_mean = {}
                    result_sample_s2 = {}
                    self.result_sample_variance = {}

                    for k in incomplete_subsets:
                        n_samples[k] = 0
                        result_sample_mean[k] = self.init_results()
                        result_sample_s2[k] = self.init_results()

                    for k in subset_sizes_samples:
                        T = set(np.random.choice(self.n, k, replace=False))
                        result_sample_update = self._evaluate_subset(game, T, r[k])
                        result_sample_mean[k], result_sample_s2[k], n_samples[
                            k] = self.update_mean_variance(result_sample_mean[k],
                                                           result_sample_s2[k],
                                                           n_samples[k],
                                                           result_sample_update)
                        if show_pbar:
                            pbar.update(1)
                        if pairing:
                            T_c = self.N - T
                            k_c = len(T_c)
                            result_sample_update = self._evaluate_subset(game, T_c, r[k_c])
                            result_sample_mean[k], result_sample_s2[k], n_samples[
                                k] = self.update_mean_variance(
                                result_sample_mean[k], result_sample_s2[k], n_samples[k],
                                result_sample_update)
                            if show_pbar:
                                pbar.update(1)
                        if n_samples[k] > 1:
                            self.result_sample_variance[k] = self.scale_results(result_sample_s2[k],
                                                                                1 / (n_samples[
                                                                                         k] - 1))


                    for k in incomplete_subsets:
                        result_complete = self.update_results(result_complete,
                                                              result_sample_mean[k])

                else:
                    r = np.zeros(self.n + 1)
                    for k in incomplete_subsets:
                        r[k] = subset_weight_vector[k] / binom(self.n, k)

                    n_samples = 0
                    result_sample_mean = self.init_results()
                    result_sample_s2 = self.init_results()
                    for k in subset_sizes_samples:
                        if show_pbar:
                            pbar.update(2) if pairing else pbar.update(1)
                        T = set(np.random.choice(self.n, k, replace=False))
                        result_sample_update = self._evaluate_subset(game, T, r[k])
                        result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(
                            result_sample_mean, result_sample_s2, n_samples, result_sample_update)
                        if pairing:
                            T_c = self.N - T
                            k_c = len(T_c)
                            result_sample_update = self._evaluate_subset(game, T_c, r[k_c])
                            result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(
                                result_sample_mean, result_sample_s2, n_samples,
                                result_sample_update)
                        if n_samples > 1:
                            self.result_sample_variance = self.scale_results(result_sample_s2,
                                                                             1 / (n_samples - 1))

                    result_complete = self.update_results(result_complete, result_sample_mean)

        results_out = self._smooth_with_epsilon(result_complete)
        self.interaction_values = result_complete
        return copy.deepcopy(results_out)

    def compute_interactions_complete(
            self,
            game: Callable[[set], float],
            interaction_subsets: dict = None
    ) -> Dict[int, np.ndarray]:
        """Computes the Exact Shapley interactions given a game (becomes computationally challenging
        around n = 15).

        Args:
            game (Callable[[set], float]): The game to be evaluated.
            interaction_subsets (dict, optional): A dictionary with the interaction subsets to be
                evaluated. Defaults to None.

        Returns:
            Dict[int, np.ndarray]: A dictionary with the interaction values for each subset size.
        """
        if interaction_subsets is None:
            interaction_subsets = {}
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
        self.interaction_values = results
        return result_out

    def compute_efficiency(self, game):
        """Computes efficiency value for SII by comparing the full game (grand coalition) to the
        empty game.

        Args:
            game (function): game function

        Returns:
            float: efficiency value (full game vs. empty game)
        """
        result = 0
        for t in range(self.s_0):
            factor = binom(self.n - t, self.s_0 - t - 1)
            sign = (-1) ** t
            for S in powerset(self.N, t, t):
                result += factor * sign * ((-1) ** self.s_0 * game(S) + game(set(self.N) - set(S)))
        return result / self.s

    def _evaluate_subset(self, game, T, p) -> dict:
        """Evaluates and weights a single subset evaluation of T sampled with probability p(T) for
        all interactions.

        Args:
            game (function): game function
            T (set): subset
            p (float): probability of sampling T

        Returns:
            tmp (dict): dictionary with interaction as key and interaction score update as value
        """
        tmp = self.init_results()
        game_eval = game(T)
        t = len(T)
        for S in powerset(self.N, self.min_order, self.s_0):
            size_intersection = len(set(S).intersection(T))
            tmp[len(S)][S] += game_eval * self.weights[len(S)][t, size_intersection] / p
        return tmp

    def _evaluate_subset_one(self, game, T, p, interaction) -> dict:
        """Evaluates and weights a single subset evaluation of T sampled with probability p(T) for
        a single interaction.

        Args:
            game (function): game function
            T (set): subset
            p (float): probability of sampling T
            interaction (tuple): interaction to evaluate

        Returns:
            tmp (dict): dictionary with interaction as key and interaction score update as value
        """
        tmp = {interaction: 0}
        game_eval = game(T)
        t = len(T)
        size_intersection = len(set(interaction).intersection(T))
        tmp[interaction] += game_eval * self.weights[len(interaction)][t, size_intersection] / p
        return tmp

    def _init_sampling_weights(self, sampling_kernel) -> tuple:
        """Initializes the subset weights for sampling for one subset of that size (q) and any
        subset of that size (p).

        Args:
            sampling_kernel (str): sampling kernel to use

        Returns:
            q (np.array): subset weights for sampling
            p (np.array): subset probabilities for sampling any subset of that size
        """
        q = np.zeros(self.n + 1)
        p = np.zeros(self.n + 1)
        for t in range(self.n + 1):
            q[t], p[t] = self._kernel_q(t, sampling_kernel)
        return q, p

    def _kernel_q(self, t, sampling_kernel) -> tuple:
        """Determines the sampling weights for a subset of size t (size_weight) and the weight for
        any subset of size t (size_weight*binom(n,t))

        Args:
            t (int): size of subset
            sampling_kernel (str): sampling kernel to use

        Returns:
            size_weight (float): weight for subset of size t
            size_weight*binom(n,t) (float): weight for any subset of size t
        """
        if sampling_kernel == "ksh":
            if t >= self.s_0 and t <= self.n - self.s_0:
                size_weight = np.math.factorial(self.n - t - self.s_0) * np.math.factorial(
                    t - self.s_0) / np.math.factorial(
                    self.n - self.s_0 + 1)
            else:
                size_weight = self.inf
        if sampling_kernel == "faith":
            if t >= 1 and t <= self.n - 1:
                size_weight = np.math.factorial(self.n - t - 1) * np.math.factorial(
                    t - 1) / np.math.factorial(self.n - 1)
            else:
                size_weight = self.inf
        if sampling_kernel == "unif-size":
            size_weight = 1
        if sampling_kernel == "unif-set":
            size_weight = 1 / binom(self.n, t)
        return size_weight, size_weight * binom(self.n, t)

    def _kernel_m(self, t, s) -> float:
        """Returns the weight for each interaction type for a subset of size t and interaction of
        size s.

        Args:
            t (int): size of subset
            s (int): size of interaction

        Returns:
            float: weight for interaction type for subset of size t and interaction of size s
                specific to the interaction type
        """
        if self.interaction_type == "SII":
            return np.math.factorial(self.n - t - s) * np.math.factorial(t) / np.math.factorial(
                self.n - s + 1)
        if self.interaction_type == "STI":
            if s == self.s_0:
                return self.s_0 * np.math.factorial(self.n - t - 1) * np.math.factorial(
                    t) / np.math.factorial(self.n)
            else:
                return 1.0 * (t == 0)
        if self.interaction_type == "FSI":
            if s == self.s_0:
                return np.math.factorial(2 * s - 1) / np.math.factorial(
                    s - 1) ** 2 * np.math.factorial(
                    self.n - t - 1) * np.math.factorial(t + s - 1) / np.math.factorial(
                    self.n + s - 1)
            else:
                raise ValueError("Lower order interactions are not supported.")

    def _compute_interactions_complete_k(
            self,
            game: Callable,
            k: int,
            pbar: tqdm = None
    ) -> Dict[int, np.ndarray]:
        """ Computes the SI values for all interactions over all subsets of size k for a given game.

        Args:
            game (function): function that returns the game value for a given subset
            k (int): size of the subset
            pbar (tqdm, optional): progress bar to track progress. Defaults to None.

        Returns:
            Dict[int, np.ndarray]: dictionary with the interaction scores for each order.
        """
        results = self.init_results()
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            if pbar is not None:
                pbar.update(1)
            t = len(T)
            interaction_subset_iterator = powerset(self.N, self.min_order, self.s_0)
            for S in interaction_subset_iterator:
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[len(S)][t, s_t]
        return results

    def _compute_interactions_complete_k_one(
            self,
            game: Callable,
            k: int,
            interaction: set
    ) -> Dict[set, float]:
        """Computes the SI values for a single interaction over all subsets of size k for a given
        game.

        Args:
            game (function): function that evaluates the game for a given subset
            k (int): size of the subsets
            interaction (set): interaction to compute the SI values for

        Returns:
            results (dict): dictionary containing the SI values for the given interaction
        """
        results = {interaction: 0}
        for T in powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            s_t = len(set(interaction).intersection(T))
            results[interaction] += game_eval * self.weights[len(interaction)][t, s_t]
        return results

    def compute_interactions_from_budget_one(
            self,
            game,
            budget,
            interaction,
            pairing=False,
            sampling_kernel="ksh",
            sampling_only=False
    ) -> Dict[set, np.ndarray]:
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
        result_complete = {interaction: 0}

        if budget > 0:
            if sampling_only:
                # sampling for all subset sizes
                incomplete_subsets = list(range(self.n + 1))
            else:
                # sampling dependent on weights p
                complete_subsets, incomplete_subsets, budget = determine_complete_subsets(0, self.n,
                                                                                          budget, p)

            for k in complete_subsets:
                # compute all deterministic subset sizes
                result_complete = self.update_results(result_complete,
                                                      self._compute_interactions_complete_k_one(
                                                          game, k, interaction))

            # Adjust budget, if pairwise sampling is used
            if pairing:
                budget = 2 * int(budget / 2)
            else:
                budget = budget

            # Sample the remaining budget and update the approximations
            if len(incomplete_subsets) > 0:
                subset_weight_vector = np.zeros(self.n + 1)
                n_samples = 0
                for k in incomplete_subsets:
                    subset_weight_vector[k] = q[k] * binom(self.n, k)

                subset_weight_vector /= np.sum(subset_weight_vector[incomplete_subsets])
                subset_sizes_samples = random.choices(incomplete_subsets, k=budget,
                                                      weights=subset_weight_vector[
                                                          incomplete_subsets])
                r = np.zeros(self.n + 1)
                for k in incomplete_subsets:
                    r[k] = subset_weight_vector[k] / binom(self.n, k)

                result_sample_mean = {interaction: 0}
                result_sample_s2 = {interaction: 0}
                for k in subset_sizes_samples:
                    T = set(np.random.choice(self.n, k, replace=False))
                    result_sample_update = self._evaluate_subset_one(game, T, r[k], interaction)
                    result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(
                        result_sample_mean, result_sample_s2, n_samples, result_sample_update)
                    if pairing:
                        T_c = self.N - T
                        k_c = len(T_c)
                        result_sample_update = self._evaluate_subset_one(game, T_c, r[k_c],
                                                                         interaction)
                        result_sample_mean, result_sample_s2, n_samples = self.update_mean_variance(
                            result_sample_mean, result_sample_s2, n_samples, result_sample_update)
        self.interaction_values = result_complete
        return copy.deepcopy(result_complete)

    def compute_from_samples(
            self,
            S_list: List[set],
            game_values: List,
            val_empty,
            val_full,
            epsilon_correction=True
    ) -> Dict[int, np.ndarray]:
        """ Estimates all Shapley values using a list of sampled subsets for a given game

        Args:
            S_list (List[set]): List of sampled subsets
            game_values (List): List of game evaluations for elements in S_list
            val_empty ([type]): model evaluated for the empty set
            val_full ([type]): model evaluated for the full set (self.N)
            epsilon_correction (bool, optional): if True, then yields the exact values of Unbiased
                KernelSHAP (only differs for sets with game({}) <> 0. Defaults to True.

        Returns:
            Dict[int, np.ndarray]: Dictionary of interaction scores for each subset size
        """
        subsets = copy.deepcopy(S_list)
        S_game_mapping = {tuple(S): game_value for S, game_value in zip(subsets, game_values)}
        S_game_mapping[tuple(set())] = val_empty
        S_game_mapping[tuple(self.N)] = val_full
        # Constant c_1
        baseline = (val_full - val_empty) / self.n
        # Harmonic number
        h = 0
        for k in range(1, self.n):
            h += 1 / k

        epsilons = self.init_results()
        results_sample = self.init_results()
        results = self.init_results()
        for T in subsets:
            game_eval = S_game_mapping[tuple(T)]
            t = len(T)
            for S in powerset(self.N, self.min_order, self.s_0):
                s_t = len(set(S).intersection(T))
                results_sample[len(S)][S] += 2 * h * game_eval * (s_t - t / self.n)
                epsilons[len(S)][S] += 2 * h * val_empty * (s_t - t / self.n)

        results_sample[len(S)] = results_sample[len(S)] / len(subsets)
        epsilons[len(S)] = epsilons[len(S)] / len(subsets)
        results[len(S)] = baseline + results_sample[len(S)]

        if epsilon_correction:
            results[len(S)] = results[len(S)] - epsilons[len(S)]

        result_out = copy.deepcopy(self._smooth_with_epsilon(results))
        return result_out

    def transform_interactions_in_n_shapley(
            self,
            interaction_values: Dict[int, np.ndarray] = None,
            n: int = None,
            reduce_one_dimension: bool = False
    ) -> Union[Dict[int, np.ndarray], Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]]:
        """Computes the n-Shapley values from the interaction values

        Args:
            interaction_values (Dict[int, np.ndarray], optional): The interaction values. Defaults to None.
            n (int, optional): The order of the Shapley values. Defaults to None.
            reduce_one_dimension (bool, optional): If True, the n-Shapley values are reduced to one dimension. Defaults to False.

        Returns:
            dict: A dictionary containing the n-Shapley values
        """
        if n is None:
            n = self.s_0
        if interaction_values is None:
            interaction_values = self.interaction_values
        bernoulli_numbers = bernoulli(n)
        result = self.init_results()
        # all subsets S with 1 <= |S| <= n
        for S in powerset(self.N, min_size=1, max_size=n):
            # get un-normalized interaction value (delta_S(x))
            S_effect = interaction_values[len(S)][S]
            subset_size = len(S)
            # go over all subsets T of length |S| + 1, ..., n that contain S
            for T in powerset(self.N, min_size=subset_size + 1, max_size=n):
                if not set(S).issubset(T):
                    continue
                # get the effect of T
                T_effect = interaction_values[len(T)][T]
                # normalization with bernoulli numbers
                S_effect += bernoulli_numbers[len(T) - subset_size] * T_effect
            result[len(S)][tuple(S)] = S_effect
        if not reduce_one_dimension:
            return result
        return self._convert_n_shapley_values_to_one_dimension(result, n=n)

    def _convert_n_shapley_values_to_one_dimension(
            self,
            n_shapley_values, n: int = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """Converts the n-Shapley values to one dimension

        Args:
            n_shapley_values (Dict[int, np.ndarray]): The n-Shapley values
            n (int, optional): The order of the Shapley values. Defaults to None.

        Returns:
            tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]: A tuple containing the positive and
            negative parts of the n-Shapley values
        """
        if n is None:
            n = max(n_shapley_values.keys())
        result_pos = {order: {player: 0. for player in range(self.n)} for order in range(1, n + 1)}
        result_neg = {order: {player: 0. for player in range(self.n)} for order in range(1, n + 1)}
        result_values = {order: {player: [] for player in range(self.n)} for order in range(1, n + 1)}

        for S in powerset(self.N, min_size=1, max_size=n):
            n_shap_value = n_shapley_values[len(S)][tuple(S)]
            for player in S:
                result_values[len(S)][player].append(n_shap_value)

        for S in powerset(self.N, min_size=1, max_size=n):
            n_shap_value = n_shapley_values[len(S)][tuple(S)]
            for player in S:
                if n_shap_value > 0:
                    result_pos[len(S)][player] += n_shap_value / len(S)
                if n_shap_value < 0:
                    result_neg[len(S)][player] += n_shap_value / len(S)
        return result_pos, result_neg

    @staticmethod
    def _compute_bernoulli_numbers(n: int) -> np.ndarray:
        """ Computes the Bernoulli numbers up to order n and returns a list of length n+1

        Args:
            n (int): The order of the Bernoulli numbers

        Returns:
            np.ndarray: The Bernoulli numbers up to order n
        """
        bernoulli_numbers = np.zeros(n + 1)
        bernoulli_numbers[0] = 1
        for k in range(1, n):
            bernoulli_number = -1 / (k + 1) * np.sum(
                [binom(k + 1, j) * bernoulli_numbers[j] for j in range(k)])
            bernoulli_numbers[k] = bernoulli_number
        return bernoulli_numbers
