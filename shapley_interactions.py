import scipy
import numpy as np
import itertools
import random
from collections import Counter
import copy
from scipy.special import binom

from scipy import special

from utils.permutations import get_n_feature_masks, get_all_feature_masks

import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NN:
    def __init__(self, n):
        self.weights_1 = np.random.normal(loc=0, scale=10, size=((100, n)))
        self.bias_1 = np.random.normal(loc=0, scale=1, size=(100))
        self.weights_2 = np.random.normal(loc=0, scale=0.5, size=((10, 100)))
        self.bias_2 = np.random.normal(loc=0, scale=1)
        self.weights_3 = np.random.normal(loc=0, scale=0.05, size=((1, 10)))
        self.bias_3 = np.random.normal(loc=0, scale=0.05)
        self.n = n
        self.intx2 = 0
        self.intx3 = 0

    def call(self, x):
        return sigmoid(np.dot(self.weights_3, np.maximum(0, np.dot(self.weights_2, np.maximum(0, np.dot(self.weights_1,
                                                                                                        x) + self.bias_1)) + self.bias_2)))

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x) - self.call(np.zeros(self.n))


class Game:
    import numpy as np
    def __init__(self, n):
        self.weights = np.random.rand(n)
        self.n = n
        self.intx2 = 0
        self.intx3 = 0

    def call(self, x):
        return np.dot(x, self.weights) + x[1] * x[2] * self.intx2 + self.intx3 * x[1] * x[2] * x[3]

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class Shapley_Interactions:
    def __init__(self, N, max_order, min_order=1, type="SII"):
        self.type = type
        self.min_order = min_order
        self.s = max_order
        self.N = N
        self.n = len(N)
        self.counter = 0

        self.weights = np.zeros((self.n + 1, self.s + 1))
        for t in range(0, self.n + 1):
            for k in range(max(0, self.s + t - self.n), min(self.s, t) + 1):
                self.weights[t, k] = (-1) ** (self.s - k) * self.kernel_m(t - k)

    def constant_R(self, incomplete_subsets, q):
        R = 0
        for t in incomplete_subsets:
            R += q[t] / binom(self.n, t)
        return R / len(incomplete_subsets)

    def powerset(self, iterable, min_size=-1, max_size=None):
        if max_size == None and min_size > -1:
            max_size = min_size
        s = list(iterable)
        if max_size == None:
            max_size = len(s)
        else:
            max_size = min(max_size, len(s))
        return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))

    def constant_c(self, game):
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

    def compute_interactions_complete(self, game):
        results = self.init_results()

        for T in self.powerset(self.N):
            game_eval = game(T)
            t = len(T)
            for S in self.powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        return results

    def evaluate_subset(self, game, T, p):
        tmp = self.init_results()
        game_eval = game(T)
        t = len(T)
        for S in self.powerset(self.N, self.min_order, self.s):
            size_intersection = len(set(S).intersection(T))
            tmp[len(S)][S] += game_eval * self.weights[t, size_intersection] / p
        return tmp

    def init_sampling_weights(self, sampling_kernel):
        q = np.zeros(n + 1)
        for t in range(self.s, self.n - self.s + 1):
            q[t] = self.kernel_q(t, sampling_kernel)
        return q

    def kernel_q(self, t, sampling_kernel):
        if sampling_kernel == "ksh":
            return np.math.factorial(self.n - t - self.s) * np.math.factorial(t - self.s) / np.math.factorial(
                self.n - self.s + 1)
        if sampling_kernel == "faith":
            return np.math.factorial(self.n - t - 1) * np.math.factorial(t - 1) / np.math.factorial(self.n - 1)
        if sampling_kernel == "unif-size":
            return 1
        if sampling_kernel == "unif-set":
            return binom(self.n, t)

    def kernel_m(self, t):
        if self.type == "SII":
            return np.math.factorial(self.n - t - self.s) * np.math.factorial(t) / np.math.factorial(
                self.n - self.s + 1)
        if self.type == "STI":
            return self.s * np.math.factorial(self.n - t - 1) * np.math.factorial(t) / np.math.factorial(self.n)
        if self.type == "SFI":
            return np.math.factorial(2 * self.s - 1) / np.math.factorial(self.s - 1) ** 2 * np.math.factorial(
                self.n - t - 1) * np.math.factorial(t + self.s - 1) / np.math.factorial(self.n + self.s - 1)

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
        n_subsets = self.n - 1
        complete_subsets = []
        paired_subsets, unpaired_subset = self.get_paired_subsets()

        incomplete_subsets = list(range(self.s, self.n - self.s + 1))
        weight_vector = copy.copy(q)
        weight_vector /= np.sum(weight_vector)
        allowed_budget = weight_vector * budget
        for subset_size_1, subset_size_2 in paired_subsets:
            subset_budget = int(binom(self.n, subset_size_1))
            if allowed_budget[subset_size_1] >= subset_budget and allowed_budget[subset_size_1] > 0:
                complete_subsets.extend((subset_size_1, subset_size_2))
                incomplete_subsets.remove(subset_size_1)
                incomplete_subsets.remove(subset_size_2)
                weight_vector[subset_size_1] = 0
                weight_vector[subset_size_2] = 0
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

    def compute_interactions_complete_k(self, game, k):
        results = self.init_results()
        for T in self.powerset(self.N, k, k):
            game_eval = game(T)
            t = len(T)
            for S in self.powerset(self.N, self.min_order, self.s):
                s_t = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * self.weights[t, s_t]
        return results

    def init_results(self):
        results = {}
        for k in range(self.min_order, self.s + 1):
            results[k] = np.zeros(np.repeat(self.n, k))
        return results

    def update_results(self, current, update):
        rslt = {}
        for l in current:
            rslt[l] = current[l] + update[l]
        return rslt

    def scale_results(self, current, factor):
        rslt = {}
        for l in current:
            rslt[l] = current[l] * factor
        return rslt

    def compute_interactions_from_budget(self, game, budget, pairing=True, sampling_kernel="ksh"):
        self.sampling_kernel = sampling_kernel
        q = self.init_sampling_weights(sampling_kernel)
        self.last_sampling_weights = q
        results = self.init_results()

        self.complete_subsets, self.incomplete_subsets, budget = self.determine_complete_subsets(budget, q)
        rslt_complete = self.init_results()
        rslt_sample = self.init_results()
        rslt_constant = self.constant_c(game)
        const_complete = self.update_results(rslt_constant, rslt_complete)
        self.last_const_complete = const_complete

        for k in self.complete_subsets:
            rslt_complete = self.update_results(rslt_complete, self.compute_interactions_complete_k(game, k))

        # print(self.incomplete_subsets,p)
        subset_weight_vector = np.asarray(
            [q[subset_size] for subset_size in self.incomplete_subsets])

        if pairing:
            n_samples = int(budget / 2)
            new_budget = 2 * int(budget / 2)
        else:
            n_samples = budget
            new_budget = budget
        if len(subset_weight_vector) > 0:
            subset_weight_vector /= np.sum(subset_weight_vector)
            subset_sizes_samples = random.choices(self.incomplete_subsets, k=n_samples, weights=subset_weight_vector)

            p = np.zeros(n + 1)
            for i, k in enumerate(self.incomplete_subsets):
                rslt_sample = self.init_results()
                n_samples = int(budget*subset_weight_vector[i])
                #p[k] = new_budget*subset_weight_vector[i] / binom(self.n, k)
                p[k] = n_samples/ (binom(self.n, k))
                for j in range(n_samples):
            #for k in subset_sizes_samples:
                # add counter here with proper weighting
                # ---> so that no subset is evaluated twice
                    T = set(np.random.choice(self.n, k, replace=False))
                    rslt_sample = self.update_results(rslt_sample, self.evaluate_subset(game, T, p[k]))
                    if pairing:
                        T_c = self.N - T
                        rslt_sample = self.update_results(rslt_sample, self.evaluate_subset(game, T_c, p[k]))
                final = self.update_results(const_complete, rslt_sample)
        return final

    def estimate_from_permutation(self, game, pi):
        results = np.zeros(np.repeat(self.n, self.s))
        if self.type == "SII":
            for k in range(self.n - self.s + 1):
                S = tuple(pi[k:k + self.s])
                subset = tuple(pi[:k])
                for L in self.powerset(S):
                    l = len(L)
                    results[S] += game(subset + L) * (-1) ** (self.s - l)
                    self.counter += 1
        if self.type == "STI":
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
                    self.counter += 1
        return results

    def permutation_approximation(self, game, budget):
        results = np.zeros(np.repeat(self.n, self.s))
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
            results += self.estimate_from_permutation(game, pi)
            n_permutations += 1
            iteration_cost = self.counter-start_counter
            budget -= iteration_cost
        return results / n_permutations

    def compute_efficiency(self, game):
        rslt = 0
        for t in range(self.s):
            factor = binom(self.n - t, self.s - t - 1)
            sign = (-1) ** (t)
            for S in self.powerset(self.N, t, t):
                rslt += factor * sign * ((-1) ** self.s * game(S) + game(set(N) - set(S)))
        return rslt / self.s


if __name__ == "__main__":
    n = 15
    N = set(range(n))

    # game = Game(n)
    game = NN(n)

    min_order = 2
    shapley_interaction_order = 2
    shapley_extractor_sii = Shapley_Interactions(N, shapley_interaction_order, min_order=min_order, type="SII")
    shapley_extractor_sti = Shapley_Interactions(N, shapley_interaction_order, min_order=min_order, type="STI")
    shapley_extractor_sfi = Shapley_Interactions(N, shapley_interaction_order, min_order=min_order, type="SFI")

    game_fun = game.set_call

    shapx_exact = {}
    shapx_list = [shapley_extractor_sii, shapley_extractor_sti, shapley_extractor_sfi]

    # Compute exact interactions
    for shapx in shapx_list:
        shapx_exact[shapx.type] = shapx.compute_interactions_complete(game_fun)
    print("Exact computations finished")

    shapx_perm = {}
    shapx_sampling = {}
    approximation_errors = {}
    sampling_kernel_list = ["unif-size"]
    pairwise_list = [True, False]

    total_subsets = 2 ** n
    budget_list = [int(total_subsets * 0.01), int(total_subsets * 0.1), int(total_subsets * 0.25), int(total_subsets * 0.5),
                   int(total_subsets * 0.75), int(total_subsets * 0.90)]
    # Compute Approximations
    for shapx in shapx_list:
        print("Starting Sampling for ", shapx.type)
        # Permutation Approximations
        for budget in budget_list:
            if shapx.type in ("STI", "SII"):
                id = shapx.type + "_" + str(budget)
                shapx_perm[id] = shapx.permutation_approximation(game_fun, budget)
                approximation_errors[id + "_perm"] = np.sum(
                    (shapx_perm[id] - shapx_exact[shapx.type][shapley_interaction_order]) ** 2)
            # Sampling Approximations
            for sampling_kernel in sampling_kernel_list:
                for pairwise in pairwise_list:
                    id = shapx.type + "_sampling_" + sampling_kernel + "_" + str(pairwise) + "_" + str(budget)
                    shapx_sampling[id] = shapx.compute_interactions_from_budget(game_fun, budget, pairing=pairwise,
                                                                                sampling_kernel=sampling_kernel)
                    approximation_errors[id] = np.sum((shapx_sampling[id][shapley_interaction_order] -
                                                       shapx_exact[shapx.type][shapley_interaction_order]) ** 2)
                    approximation_errors[id + "_full"] = np.sum((shapx.last_const_complete[shapley_interaction_order] -
                                                                 shapx_exact[shapx.type][shapley_interaction_order]) ** 2)
        print("End ", shapx.type)
