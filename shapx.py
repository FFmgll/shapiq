import typing

import numpy as np
import itertools
import copy
from scipy.special import binom


class BaseShapleyInteractions:

    def __init__(self, N, max_order, min_order=1):
        self.min_order = min_order
        self.s = max_order
        self.N = N
        self.n = len(N)
        self.counter = 0
        self.weights = np.zeros((self.n + 1, self.s + 1))

    def get_paired_subsets(self):
        subset_sizes = list(range(self.s, self.n - self.s + 1))
        n_paired_subsets = int(len(subset_sizes) / 2)
        paired_subsets = [(subset_sizes[subset_size - 1], subset_sizes[-subset_size])
                          for subset_size in range(1, n_paired_subsets + 1)]
        unpaired_subset = None
        if n_paired_subsets < len(subset_sizes) / 2:
            unpaired_subset = int(np.median(subset_sizes))
        return paired_subsets, unpaired_subset

    def determine_complete_subsets(self, budget, q):
        complete_subsets = []
        paired_subsets, unpaired_subset = self.get_paired_subsets()

        incomplete_subsets = list(range(self.s, self.n - self.s + 1))
        weight_vector = copy.copy(q)
        sum_weight_vector = np.sum(weight_vector)
        weight_vector = np.divide(weight_vector, sum_weight_vector, out=weight_vector, where=sum_weight_vector != 0)
        allowed_budget = weight_vector * budget
        for subset_size_1, subset_size_2 in paired_subsets:
            subset_budget = int(binom(self.n, subset_size_1))
            if allowed_budget[subset_size_1] >= subset_budget and allowed_budget[subset_size_1] > 0:
                complete_subsets.extend((subset_size_1, subset_size_2))
                incomplete_subsets.remove(subset_size_1)
                incomplete_subsets.remove(subset_size_2)
                weight_vector[subset_size_1] = 0
                weight_vector[subset_size_2] = 0
                if not np.sum(weight_vector) == 0:
                    weight_vector /= np.sum(weight_vector)
                budget -= subset_budget * 2
            else:
                return complete_subsets, incomplete_subsets, budget
            allowed_budget = weight_vector * budget
        if unpaired_subset is not None:
            subset_budget = int(binom(self.n, unpaired_subset))
            if budget - subset_budget >= 0:
                complete_subsets.append(unpaired_subset)
                incomplete_subsets.remove(unpaired_subset)
                budget -= subset_budget
        return complete_subsets, incomplete_subsets, budget

    def constant_R(self, incomplete_subsets, q):
        R = 0
        for t in incomplete_subsets:
            R += q[t] / binom(self.n, t)
        return R / len(incomplete_subsets)

    def powerset(self, iterable, min_size=-1, max_size=None):
        if max_size is None and min_size > -1:
            max_size = min_size
        s = list(iterable)
        if max_size is None:
            max_size = len(s)
        else:
            max_size = min(max_size, len(s))
        return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))

    def init_results(self):
        results = {}
        for k in range(self.min_order, self.s + 1):
            results[k] = np.zeros(np.repeat(self.n, k))
        return results

    @staticmethod
    def update_results(current, update):
        rslt = {}
        for l in current:
            rslt[l] = current[l] + update[l]
        return rslt

    @staticmethod
    def scale_results(current, factor):
        rslt = {}
        for l in current:
            rslt[l] = current[l] * factor
        return rslt

    @staticmethod
    def _smooth_with_epsilon(interaction_results: typing.Union[dict, np.ndarray], eps=0.00001) -> typing.Union[dict, np.ndarray]:
        if not isinstance(interaction_results, dict):
            interaction_results[np.abs(interaction_results) < eps] = 0
            return copy.deepcopy(interaction_results)
        interactions = {}
        for interaction_order, interaction_values in interaction_results.items():
            interaction_values[np.abs(interaction_values) < eps] = 0
            interactions[interaction_order] = interaction_values
        return copy.deepcopy(interactions)


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

        complete_subsets, incomplete_subsets, budget = self.determine_complete_subsets(budget, q)
        result_complete = self.init_results()
        result_constant = self._constant_c(game)

        # Calculate the subsets at the edge completely and store it away
        for k in complete_subsets:
            result_complete = self.update_results(result_complete, self._compute_interactions_complete_k(game, k))
        constant_complete = self.update_results(result_constant, result_complete)
        self.last_const_complete = copy.deepcopy(self._smooth_with_epsilon(constant_complete))
        result_complete = copy.deepcopy(constant_complete)

        # Update weights for samplings for the remaining subsets
        subset_weight_vector = np.asarray(
            [q[subset_size] for subset_size in incomplete_subsets])

        # Split the budget in case of pairing
        if pairing:
            budget = 2 * int(budget / 2)

        # Sample the remaining budget and update the approximations
        if len(subset_weight_vector) > 0:
            subset_weight_vector /= np.sum(subset_weight_vector)
            p = np.zeros(self.n + 1)
            for i, k in enumerate(incomplete_subsets):
                result_sample = self.init_results()
                n_samples = int(budget * subset_weight_vector[i])
                p[k] = n_samples / (binom(self.n, k))
                for j in range(n_samples):  # TODO add counter here with proper weighting
                    T = set(np.random.choice(self.n, k, replace=False))
                    result_sample = self.update_results(result_sample, self._evaluate_subset(game, T, p[k]))
                    if pairing:
                        T_c = self.N - T
                        result_sample = self.update_results(result_sample, self._evaluate_subset(game, T_c, p[k]))
                result_complete = self.update_results(result_complete, result_sample)
        results_out = self._smooth_with_epsilon(result_complete)
        return copy.deepcopy(results_out)

    def compute_interactions_complete(self, game):
        """Computes the Exact Shapley interactions given a game (becomes computationally challenging around n = 15)."""
        results = self.init_results()
        for T in self.powerset(self.N):
            game_eval = game(T)
            t = len(T)
            for S in self.powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        result_out = copy.deepcopy(self._smooth_with_epsilon(results))
        return result_out

    def compute_efficiency(self, game):
        result = 0
        for t in range(self.s):
            factor = binom(self.n - t, self.s - t - 1)
            sign = (-1) ** t
            for S in self.powerset(self.N, t, t):
                result += factor * sign * ((-1) ** self.s * game(S) + game(set(self.N) - set(S)))
        return result / self.s

    def _evaluate_subset(self, game, T, p):
        tmp = self.init_results()
        game_eval = game(T)
        t = len(T)
        for S in self.powerset(self.N, self.min_order, self.s):
            size_intersection = len(set(S).intersection(T))
            tmp[len(S)][S] += game_eval * self.weights[t, size_intersection] / p
        return tmp

    def _constant_c(self, game):
        rslt = self.init_results()
        for T in self.powerset(self.N, 0, self.s - 1):
            game_val = game(T)
            t = len(T)
            for S in self.powerset(self.N, self.min_order, self.s):
                rslt[len(S)][S] += game_val * self.weights[t, len(set(S).intersection(T))]

        for T in self.powerset(self.N, self.n - self.s + 1, self.n):
            game_val = game(T)
            t = len(T)
            for S in self.powerset(self.N, self.min_order, self.s):
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
        for T in self.powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            for S in self.powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        return results


class PermutationSampling(BaseShapleyInteractions):

    def __init__(self, N, max_order, min_order=1, interaction_type="SII"):
        super().__init__(N, max_order, min_order)
        self.interaction_type = interaction_type

    def permutation_approximation(self, game, budget):
        results = np.zeros(np.repeat(self.n, self.s))
        counts = np.zeros(np.repeat(self.n, self.s))
        val_empty = game({})
        val_full = game(self.N)
        iteration_cost = 0
        n_permutations = 0
        self.counter = 0
        while budget >= iteration_cost:
            start_counter = self.counter
            vals = np.zeros(self.n + 1)
            vals[0] = val_empty
            vals[-1] = val_full
            pi = np.arange(self.n)
            np.random.shuffle(pi)
            result_it, counts_it = self._estimate_from_permutation(game, pi)
            results += result_it
            counts_it = np.clip(counts_it, a_min=0, a_max=1, out=counts_it)
            counts += counts_it
            n_permutations += 1
            iteration_cost = self.counter - start_counter
            budget -= iteration_cost
        if self.interaction_type == "SII":
            results_out = np.divide(results, counts, out=results, where=counts != 0)
        else:  # STI
            results_out = results / n_permutations
        results_out = self._smooth_with_epsilon(results_out)
        return results_out

    def _estimate_from_permutation(self, game, pi):
        results = np.zeros(np.repeat(self.n, self.s))
        counts = np.zeros(np.repeat(self.n, self.s))
        if self.interaction_type == "SII":
            results, counts = self._estimate_from_permutation_sii(game, pi, results, counts)
        if self.interaction_type == "STI":
            results, counts = self._estimate_from_permutation_sti(game, pi, results, counts)
        return results, counts

    def _estimate_from_permutation_sti(self, game, pi, results, counts):
        for S in self.powerset(self.N, self.s, self.s):
            idx = 0
            for i in pi:
                if i in S:
                    break
                else:
                    idx += 1
            subset = tuple(pi[:idx])
            for L in self.powerset(S):
                l = len(L)
                results[S] += game(subset + L) * (-1) ** (self.s - l)
                counts[S] += 1
                self.counter += 1
        return results, counts

    def _estimate_from_permutation_sii(self, game, pi, results, counts):
        for k in range(self.n - self.s + 1):
            S = tuple(sorted(pi[k:k + self.s]))
            subset = tuple(pi[:k])
            for L in self.powerset(S):
                l = len(L)
                results[S] += game(subset + L) * (-1) ** (self.s - l)
                counts[S] += 1
                self.counter += 1
        return results, counts


def get_approximation_error(approx: np.ndarray, exact: np.ndarray, eps: float = 0.00001) -> float:
    error = np.sum((approx - exact) ** 2)
    error = 0. if error < eps else error  # For pretty printing ...
    return error
