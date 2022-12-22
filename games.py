import numpy as np
import copy
from transformers import pipeline
import random
import math


# TODO add exact_values computation on the games side if they allow it


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

        self.original_output = self.call(self.input_sentence)

    def call(self, x):
        outputs = self.classifier(x)
        outputs = [output['score'] * 1 if output[self._label_key] == 'POSITIVE' else output['score'] * -1 for output in outputs]
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


class SynthLinearFunction:
    """Synthetic Linear Function where you know the Shapley values and interaction terms beforehand.
    To be used to create high dimensional data with ground truths.

    Players: the input features (zero or one)
    Output: regression score -Inf, Inf
    """
    # TODO implement it correctly

    def __init__(self, n: int = 50, n_important_features: int = 10, interaction_order: int = 3):
        assert n_important_features <= n, f"Total number of features must be greater than number of important features."
        self.n = n
        self.weights = np.zeros(self.n)
        important_weights = np.random.choice(list(range(0, self.n)), size=n_important_features, replace=False)
        print(important_weights)
        self.weights[important_weights] = np.random.rand(n_important_features)
        self.n = n
        self.intx2 = 0
        self.intx3 = 0  # TODO add interaction weights

    def call(self, x):
        return np.dot(x, self.weights) + x[1] * x[2] * self.intx2 + self.intx3 * x[1] * x[2] * x[3]

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)


class SimpleGame:
    def __init__(self, n):
        self.weights = np.random.rand(n)
        self.n = n
        self.intx2 = 0
        self.intx3 = 0  # TODO delete later

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
    def __init__(self, n):
        self.n = n
        self.weights_1 = np.random.normal(loc=0, scale=10, size=(100, self.n))
        self.bias_1 = np.random.normal(loc=0, scale=1, size=100)
        self.weights_2 = np.random.normal(loc=0, scale=0.5, size=(10, 100))
        self.bias_2 = np.random.normal(loc=0, scale=1)
        self.weights_3 = np.random.normal(loc=0, scale=0.05, size=(1, 10))
        self.bias_3 = np.random.normal(loc=0, scale=0.05)

    def call(self, x):
        x = np.maximum(0, np.dot(self.weights_1, x) + self.bias_1)
        x = np.maximum(0, np.dot(self.weights_2, x) + self.bias_2)
        x = np.maximum(0, np.dot(self.weights_3, x) + self.bias_3)
        y = _sigmoid(x)
        return y

    def set_call(self, S):
        x = np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x) - self.call(np.zeros(self.n))


if __name__ == "__main__":
    game = SynthLinearFunction(n=10, n_important_features=3)
