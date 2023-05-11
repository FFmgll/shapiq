import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from games import NLPLookupGame, NLPGame, ParameterizedSparseLinearModel
from approximators import SHAPIQEstimator
from approximators.shap_iq_fast import SHAPIQEstimator as FastSHAPIQEstimator

if __name__ == "__main__":
    n = 10
    N = set(range(n))
    interaction_order = 3
    budget = 2 ** 10

    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII")

    # setup the game function (here we use a
    #game = NLPLookupGame(n=14, sentence_id=105, set_zero=True)
    game = ParameterizedSparseLinearModel(n, "uniform", 20)
    exact_values = game.exact_values(gamma_matrix=shap_iq.weights[interaction_order], min_order=1, max_order=3)
    print()
    print("exact")
    print(exact_values)
    print(game.interaction_weights)
    print()

    game_name = game.game_name
    game_fun = game.set_call


    start_time = time.time()
    approx_values_shapiq = shap_iq.compute_interactions_from_budget(
        game=game.set_call,
        budget=budget,
        pairing=False,
        sampling_kernel="ksh",
        sampling_only=False,
        stratification=False
    )
    print("approx shapiq")
    print(approx_values_shapiq)
    print(f"Time: {time.time() - start_time}")


    n_shapley_values = shap_iq.transform_interactions_in_n_shapley(
        interaction_values=approx_values_shapiq,
        n=interaction_order, reduce_one_dimension=True)
    print("n_shapley_values shapiq")
    print(n_shapley_values)
    n_shapley_values_pos, n_shapley_values_neg = n_shapley_values


    #input_sentence = game.input_sentence
    # print(input_sentence)
    #nlp_model_game = NLPGame(input_text=input_sentence)
    #input_sentence_tokens = nlp_model_game.input_sentence

    #x_words = []
    #for token in nlp_model_game.tokenized_input:
    #    word = nlp_model_game.tokenizer.decode(token)
    #    x_words.append(word)

    #x = np.arange(len(x_words))
    x = [i for i in range(n)]
    x_words = x
    values_pos = []
    for order, values in n_shapley_values_pos.items():
        values_pos.append(values)
    values_pos = pd.DataFrame(values_pos)
    values_neg = []
    for order, values in n_shapley_values_neg.items():
        values_neg.append(values)
    values_neg = pd.DataFrame(values_neg)

    reference_pos = np.zeros(n)
    reference_neg = copy.deepcopy(np.asarray(values_neg.loc[0]))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axis = plt.subplots(1, 1, figsize=(10, 5))
    for order in range(len(values_pos)):
        axis.bar(x, height=values_pos.loc[order], bottom=reference_pos, label=f"Order {order + 1}", color=colors[order])
        axis.bar(x, height=abs(values_neg.loc[order]), bottom=reference_neg, color=colors[order])
        axis.axhline(y=0, color="black", linestyle="solid")
        reference_pos += values_pos.loc[order]
        try:
            reference_neg += values_neg.loc[order + 1]
        except KeyError:
            pass

    axis.set_xticks(x, x_words)
    axis.legend()
    plt.show()

    # new --------------------------

    shap_iq = FastSHAPIQEstimator(
        player_names=N,
        max_order=interaction_order,
        min_order=1,
        interaction_type="SII"
    )

    start_time = time.time()
    approx_values_fast = shap_iq.compute_interactions_from_budget(
        game=game.set_call,
        budget=budget
    )
    print("approx shapiq 'fast'")
    print(approx_values_fast)
    print(f"Time: {time.time() - start_time}")

    n_shapley_values = shap_iq.transform_interactions_in_n_shapley(
        n=interaction_order, reduce_one_dimension=True)
    print("n_shapley_values shapiq 'fast'")
    print(n_shapley_values)
    n_shapley_values_pos, n_shapley_values_neg = n_shapley_values

    #input_sentence = game.input_sentence
    #print(input_sentence)
    #nlp_model_game = NLPGame(input_text=input_sentence)
    #input_sentence_tokens = nlp_model_game.input_sentence

    #x_words = []
    #for token in nlp_model_game.tokenized_input:
    #    word = nlp_model_game.tokenizer.decode(token)
    #    x_words.append(word)

    x = np.arange(len(x_words))
    x_words = x
    values_pos = []
    for order, values in n_shapley_values_pos.items():
        values_pos.append(values)
    values_pos = pd.DataFrame(values_pos)
    values_neg = []
    for order, values in n_shapley_values_neg.items():
        values_neg.append(values)
    values_neg = pd.DataFrame(values_neg)

    reference_pos = np.zeros(n)
    reference_neg = copy.deepcopy(np.asarray(values_neg.loc[0]))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    fig, axis = plt.subplots(1, 1, figsize=(10, 5))
    for order in range(len(values_pos)):
        axis.bar(x, height=values_pos.loc[order], bottom=reference_pos, label=f"Order {order + 1}",
                 color=colors[order])
        axis.bar(x, height=abs(values_neg.loc[order]), bottom=reference_neg, color=colors[order])
        axis.axhline(y=0, color="black", linestyle="solid")
        reference_pos += values_pos.loc[order]
        try:
            reference_neg += values_neg.loc[order + 1]
        except KeyError:
            pass

    axis.set_xticks(x, x_words)
    axis.legend()
    plt.show()

    error_shapiq = 0.
    error_shapiq_fast = 0.
    for order in range(1, interaction_order + 1):
        error_shapiq += np.sum(abs(exact_values[order] - approx_values_shapiq[order]))
        error_shapiq_fast += np.sum(abs(exact_values[order] - approx_values_fast[order]))

    print("Error shapiq", error_shapiq)
    print("Error shapiq fast", error_shapiq_fast)
