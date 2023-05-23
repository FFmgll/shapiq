import copy
import itertools
import typing

import numpy as np
from scipy.special import binom


class BaseShapleyInteractions:

    def __init__(self, N, max_order, min_order=1):
        self.min_order = min_order
        self.s_0 = max_order
        self.N = N
        self.n = len(N)
        self.weights = {}
        for s in range(min_order, max_order+1):
            self.weights[s] = np.zeros((self.n + 1, s + 1))

    def init_results(self):
        """Initialize the results dictionary with zero arrays.

        Returns:
            Dict[np.ndarray]: Dictionary with zero arrays for each interaction order.
        """
        results = {}
        for k in range(self.min_order, self.s_0 + 1):
            results[k] = np.zeros(np.repeat(self.n, k))
        return results

    @staticmethod
    def update_results(current, update):
        """Update the current results with the new update."""
        rslt = {}
        for l in current:
            rslt[l] = current[l] + update[l]
        return rslt

    @staticmethod
    def update_mean_variance(current_mean, current_s2, n_samples, update):
        """Update the mean and variance of the current results with the new update."""
        for l in current_s2:
            n_samples += 1
            delta = update[l] - current_mean[l]
            current_mean[l] += delta / n_samples
            delta2 = update[l] - current_mean[l]
            current_s2[l] += delta*delta2
        return current_mean, current_s2, n_samples

    @staticmethod
    def scale_results(current, factor):
        """Scale the current results with the given factor."""
        rslt = {}
        for l in current:
            rslt[l] = current[l] * factor
        return rslt

    @staticmethod
    def _smooth_with_epsilon(
            interaction_results: typing.Union[dict, np.ndarray],
            eps=0.00001
    ) -> typing.Union[dict, np.ndarray]:
        """Smooth the interaction results with a small epsilon to avoid numerical issues.

        Args:
            interaction_results (Union[dict, np.ndarray]): Interaction results.
            eps (float, optional): Small epsilon. Defaults to 0.00001.

        Returns:
            Union[dict, np.ndarray]: Smoothed interaction results.
        """
        if not isinstance(interaction_results, dict):
            interaction_results[np.abs(interaction_results) < eps] = 0
            return copy.deepcopy(interaction_results)
        interactions = {}
        for interaction_order, interaction_values in interaction_results.items():
            interaction_values[np.abs(interaction_values) < eps] = 0
            interactions[interaction_order] = interaction_values
        return copy.deepcopy(interactions)


def powerset(iterable, min_size=-1, max_size=None):
    """Return a powerset of the iterable with optional size limits.

    Args:
        iterable (iterable): Iterable.
        min_size (int, optional): Minimum size of the subsets. Defaults to -1.
        max_size (int, optional): Maximum size of the subsets. Defaults to None.

    Returns:
        iterable: Powerset of the iterable.
    """
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
    """Given a computational budget, determines the complete subsets that can be computed explicitly
    and the corresponding incomplete subsets that need to be estimated via sampling.

    Args:
        s (int): interaction order.
        n (int): number of features/players.
        budget (int): total allowed budget for the computation.
        q (np.ndarray): weight vector.

    Returns:
        list, list, int: complete subsets, incomplete subsets, remaining budget.

    """
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
    """Given an interaction order and the number of features/players, determines the paired subsets

    Args:
        s (int): interaction order.
        n (int): number of features/players.

    Returns:
        list, Union[int, None]: paired subsets, unpaired subset. If there is no unpaired subset,
            None is returned.
    """
    subset_sizes = list(range(s, n - s + 1))
    n_paired_subsets = int(len(subset_sizes) / 2)
    paired_subsets = [(subset_sizes[subset_size - 1], subset_sizes[-subset_size])
                      for subset_size in range(1, n_paired_subsets + 1)]
    unpaired_subset = None
    if n_paired_subsets < len(subset_sizes) / 2:
        unpaired_subset = int(np.median(subset_sizes))
    return paired_subsets, unpaired_subset