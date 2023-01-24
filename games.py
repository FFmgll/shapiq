import abc
import os

import numpy as np
import copy

import pandas as pd
from transformers import pipeline
import random
import math
import typing
from scipy.special import binom

from shapx.base import powerset
from collections import Counter
from tqdm import tqdm



def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


class NLPGame:
    """Basic NLP Game, where a sentiment analysis model given a sentence is explained.

    Players: the tokenized words in the sentence.
    Output: in range -1 to 1 the sentiment of the subset.
    """

    def __init__(self, input_text: str, label_key: str = 'label'):
        self.classifier = pipeline(model="lvwerra/distilbert-imdb", task="sentiment-analysis")
        self.tokenizer = self.classifier.tokenizer
        self.tokenized_input = np.asarray(self.tokenizer(input_text)['input_ids'][1:-1])
        self.input_sentence = self.tokenizer.decode(self.tokenized_input)
        self.n = len(self.tokenized_input)
        self._label_key = label_key
        self.game_name = "language_model"
        self.original_output = self.call(self.input_sentence)

    def call(self, x):
        outputs = self.classifier(x)
        outputs = [
            output['score'] * 1 if output[self._label_key] == 'POSITIVE' else output['score'] * -1
            for output in outputs]
        return outputs

    def set_call_iterable(self, S):
        x_inputs = []
        for i, s in enumerate(S):
            token_subset = self.tokenized_input[s]
            x_text = self.tokenizer.decode(token_subset)
            x_inputs.append(x_text)
        return self.call(x_inputs)

    def set_call(self, S):
        token_subset = self.tokenized_input[list(S)]
        x_text = self.tokenizer.decode(token_subset)
        return self.call(x_text)[0]


class NLPLookupGame:

    def __init__(self, n: int, sentence_id: int = None, used_ids: set = None, set_zero: bool = False):
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids
        if sentence_id is None:
            files = os.listdir(os.path.join("data", "nlp_values", str(n)))
            files = list(set(files) - used_ids)
            if len(files) == 0:
                files = os.listdir(os.path.join("data", "nlp_values", str(n)))
                self.used_ids = set()
            sentence_id = random.choice(files)
            sentence_id = int(sentence_id.split(".")[0])
        self.used_ids.add(str(sentence_id) + ".csv")
        data_path = os.path.join("data", "nlp_values", str(n), str(sentence_id) + ".csv")
        self.df = pd.read_csv(data_path)
        self.game_name = "language_model"
        self.n = n
        whole_data = pd.read_csv(os.path.join("data", "simplified_imdb.csv"))
        self.input_sentence = str(whole_data[whole_data["id"] == sentence_id]["text"].values[0])

        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = 's'
        for player in sorted(S):
            S_id += str(player)
        return self.storage[S_id] - self.empty_value


class SparseLinearModel:
    """Synthetic Linear Function where you know the Shapley values and interaction terms beforehand.
    To be used to create high dimensional data with ground truths.

    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """

    def __init__(
            self,
            n: int = 10,
            n_non_important_features: int = 0,
            n_interactions_per_order: typing.Dict[int, int] = None):
        assert n_non_important_features <= n, f"Total number of non-important features {n_non_important_features} " \
                                              f"must be smaller than number features {n}."
        self.game_name = "sparse_linear_model"
        self.n = n
        n_important = n - n_non_important_features
        N_important = np.asarray(list(range(0, n_important)))
        self.interaction_weights = {}
        if n_interactions_per_order is not None:
            for interaction_order, n_interactions in n_interactions_per_order.items():
                interacting_features = set()
                if math.comb(n_important, interaction_order) < n_interactions:
                    raise ValueError(
                        f"The number of interaction per order {interaction_order} is {n_interactions} "
                        f"exceeding the total number of combinations "
                        f"({math.comb(n_important, interaction_order)}). Use a value smaller or equal to "
                        f"{math.comb(n_important, interaction_order)}.")
                while len(interacting_features) < n_interactions:  # might stall at certain parameters
                    interaction_sample = tuple(sorted(np.random.choice(N_important, size=interaction_order, replace=False)))
                    interacting_features.add(interaction_sample)
                interaction_weights = list(np.random.rand(len(interacting_features)))
                for interaction_feature_pair, interaction_weight in zip(interacting_features, interaction_weights):
                    self.interaction_weights[interaction_feature_pair] = interaction_weight
        self.N = np.asarray(list(range(0, self.n)))
        try:
            self._highest_interaction_order = max(n_interactions_per_order.keys())
        except AttributeError:
            self._highest_interaction_order = 0

    #OLD EXACT CALCULATION
    def exact_values_old(self, gamma_matrix, min_order, max_order):
        results = {}
        for s in range(min_order, max_order+1):
            results[s] = np.zeros(np.repeat(self.n, s))
            for subset, weight in self.interaction_weights.items():
                q = len(subset)
                for S in powerset(self.N, s, s):
                    r = len(set(subset).intersection(S))
                    results[s][S] += weight * self.coefficient_weighting(gamma_matrix, s, q, r)
        return results

    def exact_values(self, gamma_matrix, min_order, max_order, interaction_subsets={}):
        results = {}
        #pre-compute weights in matrix: order x interaction set sizes x intersection set sizes
        exact_value_weights = np.zeros((max_order+1,self.n+1,max_order+1))
        pbar_budget_precomputation = (max_order-min_order+1)*len(self.allowed_interaction_sizes)*(max_order+1)
        pbar = tqdm(total=pbar_budget_precomputation, desc="Exact values: pre-computed weights")
        for s in range(min_order,max_order+1):
            for q in self.allowed_interaction_sizes:
                for r in range(0,max_order+1):
                    exact_value_weights[s,q,r] = self.coefficient_weighting(gamma_matrix, s, q, r)
                    pbar.update(1)
        pbar.close()

        number_of_interactions = 0
        for s in range(min_order,max_order+1):
            number_of_interactions += binom(self.n,s)

        pbar_budget_final = (max_order-min_order+1)*len(self.interaction_weights)*number_of_interactions
        pbar = tqdm(total=pbar_budget_final, desc="Exact values: Final computation")
        for s in range(min_order, max_order+1):
            results[s] = np.zeros(np.repeat(self.n, s))
            for subset, weight in self.interaction_weights.items():
                q = len(subset)
                if len(interaction_subsets) == 0:
                    interaction_subsets_iterator = powerset(self.N, s, s)
                else:
                    interaction_subsets_iterator = copy.deepcopy(interaction_subsets)
                for S in interaction_subsets_iterator:
                    r = len(set(subset).intersection(S))
                    results[s][S] += weight * exact_value_weights[s,q,r]
                    pbar.update(1)
        pbar.close()
        return results

    def coefficient_weighting(self, gamma_matrix, s, q, s_cap_q):
        rslt = 0
        for t in range(q, self.n + 1):
            add = min(t - q, s - s_cap_q)
            for l in range(add + 1):
                rslt += binom(self.n - q - (s - s_cap_q), t - q - l) * binom(s - s_cap_q, l) * \
                        gamma_matrix[t, l + s_cap_q]
        return rslt

    def call(self, x):
        output = sum([
            np.prod(x[[*features]]) * interaction_weight
            for features, interaction_weight in self.interaction_weights.items()])
        return output

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class ParameterizedSparseLinearModel(SparseLinearModel):

    def __init__(self, n, weighting_scheme, n_interactions,
                 max_interaction_size=-1, min_interaction_size=1,n_non_important_features=0):
        self.n_non_important_features = n_non_important_features
        if max_interaction_size == -1:
            max_interaction_size = n
        max_interaction_size = min(n-n_non_important_features,max_interaction_size)
        weighting_ratios = np.zeros(n + 1)
        allowed_interaction_sizes = np.arange(min_interaction_size, max_interaction_size + 1)
        self.allowed_interaction_sizes = allowed_interaction_sizes
        for k in allowed_interaction_sizes:
            if weighting_scheme == "uniform":
                weighting_ratios[k] += 1
            if weighting_scheme == "center":
                weighting_ratios[k] += binom(n, k)
            if weighting_scheme == "tail":
                weighting_ratios[k] += 1 / binom(n, k)
        weighting_ratios /= np.sum(weighting_ratios)
        interaction_sizes = random.choices(allowed_interaction_sizes, k=n_interactions,
                                           weights=weighting_ratios[allowed_interaction_sizes])
        n_interactions_per_order = Counter(interaction_sizes)
        for k in allowed_interaction_sizes:
            n_interactions_per_order[k] = min(n_interactions_per_order[k],binom(n-n_non_important_features,k))
        super().__init__(n=n, n_interactions_per_order=n_interactions_per_order,n_non_important_features=n_non_important_features)


class SimpleGame:
    def __init__(self, n):
        self.weights = np.random.rand(n)
        self.n = n
        self.intx2 = random.random()
        self.intx3 = 0

    def call(self, x):
        return np.dot(x, self.weights) + x[1] * x[2] * self.intx2 + self.intx3 * x[1] * x[2] * x[3]

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class SyntheticNeuralNetwork:
    """Synthetic Neural Network
    To be used to have a function behaving like a neural network with custom controls

    Players: the input features (zero or one)
    Output: classification score between 0 and 1
    """

    def __init__(self, n,set_zero=True):
        self.n = n
        self.game_name = "synth_neural_network"
        self.weights_1 = np.random.normal(loc=0, scale=10, size=(100, self.n))
        self.bias_1 = np.random.normal(loc=0, scale=1, size=100)
        self.weights_2 = np.random.normal(loc=0, scale=0.5, size=(10, 100))
        self.bias_2 = np.random.normal(loc=0, scale=1)
        self.weights_3 = np.random.normal(loc=0, scale=0.05, size=(10, 10))
        self.bias_3 = np.random.normal(loc=0, scale=0.05)
        self.weights_4 = np.random.normal(loc=0, scale=0.05, size=(1, 10))
        self.bias_4 = np.random.normal(loc=0, scale=0.05)
        self.empty_val = 0
        if set_zero:
            self.empty_val = self.set_call({})

    def call(self, x):
        x = np.maximum(0, np.dot(self.weights_1, x) + self.bias_1)
        x = np.maximum(0, np.dot(self.weights_2, x) + self.bias_2)
        x = np.maximum(0, np.dot(self.weights_3, x) + self.bias_3)
        x = np.dot(self.weights_4, x) + self.bias_4
        y = _sigmoid(x)
        return y

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x) - self.empty_val

if __name__ == "__main__":
    pass
