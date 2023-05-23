import copy

import numpy as np

from .base import BaseShapleyInteractions, powerset
from scipy.special import binom


class PermutationSampling(BaseShapleyInteractions):
    """ Estimates the SI (for SII, STI) using the permutation sampling apporach """
    def __init__(self, N, order, interaction_type="SII", top_order: bool = True):
        min_order = order if top_order else 1
        super().__init__(N, order, min_order)
        self.interaction_type = interaction_type

    def approximate_with_budget(self, game, budget):
        results = self.init_results()
        results_out = self.init_results()
        counts = self.init_results()
        val_empty = game({})
        val_full = game(self.N)
        if self.interaction_type == "SII":
            iteration_cost = 0
            for s in range(self.min_order, self.s_0+1):
                iteration_cost += (self.n-s+1)*2**s
        elif self.interaction_type == "STI":
            iteration_cost = binom(self.n, self.s_0)*2**self.s_0
            # Compute lower order interactions:
            lower_order_cost = 0
            if self.s_0 > 1:
                for s in range(self.min_order,self.s_0):
                    lower_order_cost += binom(self.n,s)
                results = self._compute_lower_order_sti(game, results)
                budget -= lower_order_cost
        else:
            raise ValueError("Interaction Type not supported")
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
            for s in range(self.min_order,self.s_0+1):
                results[s] += result_it[s]
                counts_it[s] = np.clip(counts_it[s], a_min=0, a_max=1, out=counts_it[s])
                counts[s] += counts_it[s]
            n_permutations += 1
            self.iteration_cost = iteration_cost
            self.iteration_cost2 = self.counter - start_counter
            budget -= iteration_cost
        if self.interaction_type == "SII":
            for s in range(self.min_order,self.s_0+1):
                results_out[s] = np.divide(results[s], counts[s], out=results[s], where=counts[s] != 0)
        else:  # STI
                results_out = copy.deepcopy(results)
                results_out[self.s_0] = results[self.s_0] / n_permutations
        results_out = self._smooth_with_epsilon(results_out)
        return results_out

    def approximate_with_budget_one(self, game, budget, interaction, pairing: bool = False):
        results = {interaction:0}
        results_out={interaction:0}
        counts = {interaction:0}
        val_empty = game({})
        val_full = game(self.N)
        if self.interaction_type == "SII":
            iteration_cost = 1
        elif self.interaction_type == "STI":
            iteration_cost = 2 ** self.s
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
            result_it, counts_it = self._estimate_from_permutation_one(game, pi, interaction)
            results[interaction] += result_it[interaction]
            #counts_it = np.clip(counts_it, a_min=0, a_max=1, out=counts_it)
            counts[interaction] += counts_it[interaction]
            n_permutations += 1
            self.iteration_cost = iteration_cost
            self.iteration_cost2 = self.counter - start_counter
            budget -= iteration_cost
        if self.interaction_type == "SII" and counts[interaction]>0:
            results_out[interaction] = np.divide(results[interaction], counts[interaction])
        else:  # STI
            if n_permutations>0:
                results_out[interaction] = results[interaction] / n_permutations
        return results_out

    def _estimate_from_permutation(self, game, pi):
        results = self.init_results()
        counts = self.init_results()
        if self.interaction_type == "SII":
            for s in range(self.min_order,self.s_0+1):
                #Compute interactions of order s
                results, counts = self._estimate_from_permutation_sii(game, pi, results, counts, s)
        if self.interaction_type == "STI":
            results, counts = self._estimate_from_permutation_sti(game, pi, results, counts)
        return results, counts

    def _estimate_from_permutation_one(self, game, pi, interaction):
        results = {interaction:0}
        counts = {interaction:0}
        if self.interaction_type == "SII":
            results, counts = self._estimate_from_permutation_sii(game, pi, results, counts,interaction)
        if self.interaction_type == "STI":
            results, counts = self._estimate_from_permutation_sti(game, pi, results, counts,interaction)
        return results, counts

    def _estimate_from_permutation_sti(self, game, pi, results, counts, interaction={}):
        if len(interaction) == 0:
            interaction_subsets_iterator = powerset(self.N,self.s_0,self.s_0)
        else:
            interaction_subsets_iterator = [interaction]

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
                results[self.s_0][S] += game(subset + L) * (-1) ** (self.s_0 - l)
                counts[self.s_0][S] += 1
                self.counter += 1
        return results, counts

    def _compute_lower_order_sti(self, game, results):
        game_evals = {}
        for S in powerset(self.N,max_size=self.s_0-1):
            game_evals[tuple(S)] = game(S)
        for S in powerset(self.N,min_size=self.min_order,max_size=self.s_0-1):
            s = len(S)
            for L in powerset(S):
                l = len(L)
                results[s][S] += game_evals[L]* (-1) ** (s - l)
        return results

    def _estimate_from_permutation_sii(self, game, pi, results, counts, s, interaction={}):
        for k in range(self.n - s + 1):
            S = tuple(sorted(pi[k:k + s]))
            if len(interaction) == 0 or S == interaction:
                subset = tuple(pi[:k])
                for L in powerset(S):
                    l = len(L)
                    game_eval = game(subset + L)
                    results[s][S] +=  game_eval*(-1) ** (s - l)
                    counts[s][S] += 1
                    self.counter += 1
        return results, counts