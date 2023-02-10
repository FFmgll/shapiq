import copy
import itertools
import typing

import numpy as np
from scipy.special import binom


class BaseShapleyInteractions:

    def __init__(self, N, max_order, min_order=1):
        self.min_order = min_order
        self.s = max_order
        self.N = N
        self.n = len(N)
        self.weights = np.zeros((self.n + 1, self.s + 1))

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
    def update_mean_variance(current_mean, current_s2, n_samples, update):
        rslt = {}
        for l in current_s2:
            n_samples += 1
            delta = update[l] - current_mean[l]
            current_mean[l] += delta / n_samples
            delta2 = update[l] - current_mean[l]
            current_s2[l] += delta*delta2
        return current_mean,current_s2,n_samples

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


def powerset(iterable, min_size=-1, max_size=None):
    if max_size is None and min_size > -1:
        max_size = min_size
    s = list(iterable)
    if max_size is None:
        max_size = len(s)
    else:
        max_size = min(max_size, len(s))
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))


def determine_complete_subsets(s, n, budget, q):
    complete_subsets = []
    paired_subsets, unpaired_subset = get_paired_subsets(s, n)

    incomplete_subsets = list(range(s, n - s + 1))
    weight_vector = copy.copy(q)
    sum_weight_vector = np.sum(weight_vector)
    weight_vector = np.divide(weight_vector, sum_weight_vector, out=weight_vector, where=sum_weight_vector != 0)
    allowed_budget = weight_vector * budget
    for subset_size_1, subset_size_2 in paired_subsets:
        subset_budget = int(binom(n, subset_size_1))
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
        subset_budget = int(binom(n, unpaired_subset))
        if budget - subset_budget >= 0:
            complete_subsets.append(unpaired_subset)
            incomplete_subsets.remove(unpaired_subset)
            budget -= subset_budget
    return complete_subsets, incomplete_subsets, budget


def get_paired_subsets(s, n):
    subset_sizes = list(range(s, n - s + 1))
    n_paired_subsets = int(len(subset_sizes) / 2)
    paired_subsets = [(subset_sizes[subset_size - 1], subset_sizes[-subset_size])
                      for subset_size in range(1, n_paired_subsets + 1)]
    unpaired_subset = None
    if n_paired_subsets < len(subset_sizes) / 2:
        unpaired_subset = int(np.median(subset_sizes))
    return paired_subsets, unpaired_subset