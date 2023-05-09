import copy
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from games import NLPLookupGame, NLPGame
from approximators import SHAPIQEstimator
from approximators.shap_iq_fast import SHAPIQEstimator as FastSHAPIQEstimator

if __name__ == "__main__":

    # setup the game function (here we use a
    game = NLPLookupGame(n=14, sentence_id=105, set_zero=True)

    game_name = game.game_name
    game_fun = game.set_call
    n = game.n
    N = set(range(n))

    interaction_order = 3
    budget = 2 ** 14 + 4



    shapley_extractor_sii_old = SHAPIQEstimator(
        N=N,
        max_order=interaction_order,
        min_order=1,
        interaction_type="SII"
    )

    """
    start_time = time.time()
    approx_value_old = shapley_extractor_sii_old.compute_interactions_from_budget(
        game=game.set_call,
        budget=budget,
        pairing=False,
        sampling_kernel="ksh",
        sampling_only=False,
        stratification=False
    )
    print("Time elapsed (old):", time.time() - start_time)
    """

    shapley_extractor_sii_fast = FastSHAPIQEstimator(
        player_names=N,
        max_order=interaction_order,
        min_order=1,
        interaction_type="SII"
    )

    start_time = time.time()
    approx_value_fast = shapley_extractor_sii_fast.compute_interactions_from_budget(
        game=game.set_call,
        budget=budget
    )
    print("Time elapsed (new):", time.time() - start_time)


    # ground truth values
    ground_truth_results = shapley_extractor_sii_old.compute_interactions_complete(game_fun)


    #print(ground_truth_results)

    error_old = 0
    error_new = 0
    for order in range(1, interaction_order + 1):
        #error_old += np.sum(np.abs(approx_value_old[order] - ground_truth_results[order]))
        error_new += np.sum(np.abs(approx_value_fast[order] - ground_truth_results[order]))

    print("MAE old implementation", error_old)
    print("MAE new implementation", error_new)


    n_shapley_values = shapley_extractor_sii_fast.transform_interactions_in_n_shapley(
        n=interaction_order, reduce_one_dimension=True)
    print(n_shapley_values)
    n_shapley_values_pos, n_shapley_values_neg = n_shapley_values

    input_sentence = game.input_sentence
    print(input_sentence)
    nlp_model_game = NLPGame(input_text=input_sentence)
    input_sentence_tokens = nlp_model_game.input_sentence

    x_words = []
    for token in nlp_model_game.tokenized_input:
        word = nlp_model_game.tokenizer.decode(token)
        x_words.append(word)

    x = np.arange(len(x_words))
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
        print(values_pos.loc[order].values)
        print(reference_pos)
        print(values_neg.loc[order].values)
        print(reference_neg)
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


