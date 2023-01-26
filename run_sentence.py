import copy

from games import NLPGame, NLPLookupGame
from shapx import ShapleyInteractionsEstimator
import numpy as np
from pprint import pprint

from transformers import pipeline


def get_top_k(interaction_values: np.ndarray, input_words: list, k: int = 3, direction="top"):
    interaction_order = len(interaction_values.shape)
    interactions_flat = interaction_values.flatten()
    interactions_sorted = list(np.sort(interactions_flat))

    if direction == "top":
        top_k = list(reversed(interactions_sorted[-k:]))
    else:
        top_k = list(interactions_sorted[:k])

    top_k_features = []

    replaced_sentences = []

    for i in range(0, k):
        k_value = top_k[i]
        indices = np.where(interaction_values == k_value)
        words_og = [input_words[int(indices[i])] for i in range(interaction_order)]
        features = copy.deepcopy(words_og)
        features.append(round(k_value, 3))
        for i in range(1, 15 + 1):
            if i / 15 >= abs(k_value):
                break
        color_value = 15 - i + 1
        if direction == "top":
            colorbox = "\colorbox{blue" + str(color_value) + "}{"
        else:
            colorbox = "\colorbox{red" + str(color_value) + "}{"
        words = [colorbox + str(word) + "}" for word in words_og]
        features.extend(words)

        top_k_features.append(tuple(features))

        sentence_rep = sentence
        for i, word in enumerate(words_og):
            sentence_rep = sentence_rep.replace(word, words[i])
        replaced_sentences.append(sentence_rep)

    return top_k_features, replaced_sentences


if __name__ == "__main__":
    SHAPLEY_INTERACTION_ORDER = 3
    n = 14

    classifier = pipeline(model="lvwerra/distilbert-imdb", task="sentiment-analysis")
    tokenizer = classifier.tokenizer

    # sentence = "I cannot remember why I liked the movie in the first place"
    # input_tokens = np.asarray(tokenizer(sentence)['input_ids'][1:-1])
    # input_words = [tokenizer.decode(input_tokens[i]) for i in range(len(input_tokens))]

    game = NLPLookupGame(n=n)
    sentence = game.input_sentence
    input_tokens = np.asarray(tokenizer(sentence)['input_ids'][1:-1])
    input_words = [tokenizer.decode(input_tokens[i]) for i in range(len(input_tokens))]

    print(game.input_sentence)
    n = game.n
    N = set(range(0, n))
    budget = min(2 ** n, 2 ** 14) + 3

    full_value = game.set_call(N)
    print(full_value)

    shapley_extractor_sii = ShapleyInteractionsEstimator(
        N, SHAPLEY_INTERACTION_ORDER, min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="SII")
    shapley_extractor_sti = ShapleyInteractionsEstimator(
        N, SHAPLEY_INTERACTION_ORDER, min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="STI")
    shapley_extractor_sfi = ShapleyInteractionsEstimator(
        N, SHAPLEY_INTERACTION_ORDER, min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="SFI")

    approximators = {
        "SII": shapley_extractor_sii,
        "STI": shapley_extractor_sti,
        "SFI": shapley_extractor_sfi
    }

    results = {}
    top_k_features = {}
    bot_k_features = {}
    for approx_type, approximator in approximators.items():
        result = approximator.compute_interactions_from_budget(
            game=game.set_call, budget=budget, sampling_kernel="unif-set", pairing=False)
        results[approx_type] = result[SHAPLEY_INTERACTION_ORDER]
        top_k_features[approx_type], sentence_rep = get_top_k(
            interaction_values=results[approx_type], input_words=input_words, k=3, direction="top")
        #print(approx_type, "top", sentence_rep[0])
        bot_k_features[approx_type], sentence_rep = get_top_k(
            interaction_values=results[approx_type], input_words=input_words, k=3, direction="bot")
        #print(approx_type, "bot", sentence_rep[0])

    print("Top Interactions")
    pprint(top_k_features)
    print("Bot Interactions")
    pprint(bot_k_features)
