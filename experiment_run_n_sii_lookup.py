import copy
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from approximators import SHAPIQEstimator, PermutationSampling
from games import NLPGame, LookUpGame, NLPLookupGame

if __name__ == "__main__":

    interaction_order = 4
    budget = int(2 ** 14 * 0.5)

    # CONSTANTS ------------------------------------------------------------------------------------
    data_id = 281
    data_n = 14

    data_set_name = "nlp_values"
    game_list = []
    game = LookUpGame(data_folder=data_set_name, n=data_n, data_id=data_id, set_zero=True)

    # get information about the input senctence and the model --------------------------------------
    input_sentence = NLPLookupGame(n=data_n, sentence_id=data_id, set_zero=True).input_sentence
    nlp_model_game = NLPGame(input_text=input_sentence)
    original_ouput = nlp_model_game.original_output
    input_sentence_tokens = nlp_model_game.input_sentence
    input_words = []
    for token in nlp_model_game.tokenized_input:
        word = nlp_model_game.tokenizer.decode(token)
        input_words.append(word)
    print(input_words)

    print("Input sentence: ", input_sentence, "original output: ", original_ouput)

    # get number of players ------------------------------------------------------------------------
    n = game.n
    N = set(range(n))

    # get estiamtes  -------------------------------------------------------------------------------
    estimates = {}
    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    estimates['GT'] = shap_iq.compute_interactions_complete(game=game.set_call)
    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    estimates['shapiq'] = shap_iq.compute_interactions_from_budget(game=game.set_call, budget=budget, show_pbar=True)
    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    estimates['shapiq_explicit'] = shap_iq.compute_interactions_from_budget(game=game.set_call, budget=budget, only_expicit=True, show_pbar=True)
    permutation_sampling = PermutationSampling(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    estimates['permutation'] = permutation_sampling.approximate_with_budget(game=game.set_call, budget=budget)

    # Draw plot ------------------------------------------------------------------------------------
    params = {
        'legend.fontsize': 'x-large', 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'
    }

    fig, axis = plt.subplots(figsize=(6, 4.15))

    x = np.arange(n)
    x_pos_dict = {'GT': x - 0.3, 'shapiq': x - 0.1, 'shapiq_explicit': x + 0.1, 'permutation': x + 0.3}
    width = 0.15
    #x_pos_dict = {'GT': x - 0.3, 'shapiq': x, 'shapiq_explicit': x + 0.3}
    #
    #width = 0.25

    min_max_values = [0, 0]

    # a color list that works for color blind people
    colors = ["#D81B60", "#FFB000", "#1E88E5", "#FE6100", "#FFB000"]
    for i, estimator_id in enumerate(estimates.keys()):

        # get estimates
        sii_estimates = estimates[estimator_id]

        # plot ground truth values
        n_shapley_values = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_estimates, n=interaction_order, reduce_one_dimension=True)
        n_shapley_values_pos, n_shapley_values_neg = n_shapley_values

        # transform data to make plotting easier
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

        for order in range(len(values_pos)):
            axis.bar(x_pos_dict[estimator_id], height=values_pos.loc[order], bottom=reference_pos, width=width, color=colors[order])
            axis.bar(x_pos_dict[estimator_id], height=abs(values_neg.loc[order]), bottom=reference_neg, width=width, color=colors[order])
            axis.axhline(y=0, color="black", linestyle="solid")
            reference_pos += values_pos.loc[order]
            try:
                reference_neg += values_neg.loc[order + 1]
            except KeyError:
                pass
            min_max_values[0] = min(min_max_values[0], min(reference_neg))
            min_max_values[1] = max(min_max_values[1], max(reference_pos))

    # add legend
    legend_elements = []
    for order in range(interaction_order):
        legend_elements.append(Patch(facecolor=colors[order], edgecolor='black', label=f"Order {order + 1}"))
    axis.legend(handles=legend_elements, loc='upper center', ncol=interaction_order)

    axis.set_title(r"n-SII values for a sentence provided to the LM")

    x_ticks_labels = [word for word in input_words]
    axis.set_xticks(x)
    axis.set_xticklabels(x_ticks_labels, rotation=45, ha='right')

    axis.set_xlim(-0.5, n - 0.5)
    axis.set_ylim(min_max_values[0] * 1.05, min_max_values[1] * 1.3)

    axis.set_ylabel("n-SII values")

    plt.tight_layout()

    # save plot ------------------------------------------------------------------------------------
    save_path = os.path.join("plots", "n_SII_sampling.pdf")
    fig.savefig(save_path)

    plt.show()
