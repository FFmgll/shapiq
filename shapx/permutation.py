import numpy as np

from .base import BaseShapleyInteractions, powerset
from scipy.special import binom


class PermutationSampling(BaseShapleyInteractions):

    def __init__(self, N, max_order, min_order=1, interaction_type="SII"):
        super().__init__(N, max_order, min_order)
        self.interaction_type = interaction_type

    def approximate_with_budget(self, game, budget, pairing: bool = False, interaction_subsets={}):
        results = np.zeros(np.repeat(self.n, self.s))
        counts = np.zeros(np.repeat(self.n, self.s))
        val_empty = game({})
        val_full = game(self.N)
        if self.interaction_type == "SII":
            iteration_cost = (self.n-1)*2**self.s
        elif self.interaction_type == "STI":
            iteration_cost = binom(self.n, self.s)*2**self.s
        else:
            raise ValueError("Wrong Interaction Type")
        n_permutations = 0
        self.counter = 0
        while budget >= iteration_cost:
            start_counter = self.counter
            vals = np.zeros(self.n + 1)
            vals[0] = val_empty
            vals[-1] = val_full
            pi = np.arange(self.n)
            np.random.shuffle(pi)
            result_it, counts_it = self._estimate_from_permutation(game, pi,interaction_subsets)
            results += result_it
            counts_it = np.clip(counts_it, a_min=0, a_max=1, out=counts_it)
            counts += counts_it
            n_permutations += 1
            self.iteration_cost = iteration_cost
            self.iteration_cost2 = self.counter - start_counter
            budget -= iteration_cost
        if self.interaction_type == "SII":
            results_out = np.divide(results, counts, out=results, where=counts != 0)
        else:  # STI
            results_out = results / n_permutations
        results_out = self._smooth_with_epsilon(results_out)
        return results_out

    def _estimate_from_permutation(self, game, pi, interaction_subsets):
        results = np.zeros(np.repeat(self.n, self.s))
        counts = np.zeros(np.repeat(self.n, self.s))
        if self.interaction_type == "SII":
            results, counts = self._estimate_from_permutation_sii(game, pi, results, counts, interaction_subsets)
        if self.interaction_type == "STI":
            results, counts = self._estimate_from_permutation_sti(game, pi, results, counts, interaction_subsets)
        return results, counts

    def _estimate_from_permutation_sti(self, game, pi, results, counts, interaction_subsets={}):
        if len(interaction_subsets) == 0:
            interaction_subsets_iterator = powerset(self.N,self.s,self.s)
        else:
            interaction_subsets_iterator = interaction_subsets

        for S in interaction_subsets_iterator:
            idx = 0
            for i in pi:
                if i in S:
                    break
                else:
                    idx += 1
            subset = tuple(pi[:idx])
            for L in powerset(S):
                l = len(L)
                results[S] += game(subset + L) * (-1) ** (self.s - l)
                counts[S] += 1
                self.counter += 1
        return results, counts

    def _estimate_from_permutation_sii(self, game, pi, results, counts, interaction_subsets={}):
        for k in range(self.n - self.s + 1):
            S = tuple(sorted(pi[k:k + self.s]))
            if len(interaction_subsets) == 0 or S in interaction_subsets.keys():
                subset = tuple(pi[:k])
                for L in powerset(S):
                    l = len(L)
                    results[S] += game(subset + L) * (-1) ** (self.s - l)
                    counts[S] += 1
                    self.counter += 1
        return results, counts
