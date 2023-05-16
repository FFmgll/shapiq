import copy
import os
from typing import Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from approximators import SHAPIQEstimator
from games import NLPGame, ParameterizedSparseLinearModel


def k_largest_index_argpartition_v1(a, k):
    idx = np.argpartition(-a.ravel(), k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))


def find_top_k_n_shapley(n_shapley_dict: Dict[int, np.ndarray], k: int):
    """Find the top k n-Shapley values and indices for each order and returns the index of the top k values."""
    top_k_indices = {}
    top_k_values = {}
    for order in n_shapley_dict.keys():
        # find the indices where n_shapley_dict[order] is absolute value is higher than the k-th highest value
        top_k_indices[order] = k_largest_index_argpartition_v1(a=np.abs(n_shapley_dict[order]), k=k)
        top_k_values_order = []
        for i in range(k):
            index = tuple(top_k_indices[order][i])
            top_k_values_order.append(n_shapley_dict[order][index])
        top_k_values[order] = np.array(top_k_values_order)
    return top_k_indices, top_k_values


if __name__ == "__main__":

    interaction_order = 3
    budget = 2**15

    #og = "However, I really liked the ending so much, I actually smiled and cried tears of joy. I felt good."
    #sentence = "however I really liked the ending so much I actually smiled and cried tears of joy I felt good"

    #og = "I actually liked the ending even though it did not make a lot of sense."
    #sentence = "i actually liked the ending even though it did not make a lot of sense"

    #og = "I liked it more than most people, and actually rated it a 4 out of 10."
    #sentence = "i liked it more than most people and actually rated it a 4 out of 10"

    og = "My first thought after this movies conclusion was this. Not good, but not bad, for early-to-mid eighties."
    sentence = "my first thought after this movies conclusion was this not good but not bad for early to mid eighties"

    #og = "It is a gruesome cannibal movie. But it's not bad. If you like Hannibal, you'll love this."
    #sentence = "it is a gruesome cannibal movie but it is not bad if you like hannibal you will love this"

    #og = "Not bad for a Mixed tag match, it had entertaining antics, and passed the time well."
    #sentence = "not bad for a mixed tag match it had entertaining antics and passed the time well"

    #og = "At the time I did not know Richard Attenborough had directed it. But I am not surprised."
    #sentence = "at the time I did not know richard attenborough had directed it but I am not surprised"

    #og = "However, I would have liked to have seen more in the movie about WHY he took on this persona"
    #sentence = "however I would have liked to have seen more in the movie about why he took on this persona"

    game = NLPGame(input_text=sentence)
    n = game.n
    N = set(range(n))

    input_sentence_tokens = game.input_sentence
    input_words = []
    for token in game.tokenized_input:
        word = game.tokenizer.decode(token)
        input_words.append(word)

    print("Original Output:", game.original_output, "n:", game.n)
    print("'words':", input_words)

    x = np.arange(n)

    # estimate the SII values and n-SHAPL values

    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)
    sii_estimates = shap_iq.compute_interactions_from_budget(budget=budget, game=game.set_call, pairing=True, show_pbar=True)


    n_shapley_values = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_estimates, n=interaction_order, reduce_one_dimension=True)
    n_shapley_values_pos, n_shapley_values_neg = n_shapley_values



    # plot the n-Shapley values --------------------------------------------------------------------

    params = {
        'legend.fontsize': 'x-large', 'axes.labelsize': 'x-large', 'axes.titlesize': 'x-large',
        'xtick.labelsize': 'x-large', 'ytick.labelsize': 'x-large'
    }
    fig, axis = plt.subplots(figsize=(6, 4.35))

    x = np.arange(n)
    min_max_values = [0, 0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
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
        axis.bar(x, height=values_pos.loc[order], bottom=reference_pos, color=colors[order])
        axis.bar(x, height=abs(values_neg.loc[order]), bottom=reference_neg, color=colors[order])
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
        legend_elements.append(
            Patch(facecolor=colors[order], edgecolor='black', label=f"Order {order + 1}"))
    axis.legend(handles=legend_elements, loc='upper center', ncol=interaction_order)

    axis.set_title(r"n-Shapley values for a sentence provided to the LM")

    x_ticks_labels = [word for word in input_words]
    axis.set_xticks(x)
    axis.set_xticklabels(x_ticks_labels, rotation=45, ha='right')

    axis.set_xlim(-0.5, n - 0.5)
    axis.set_ylim(min_max_values[0] * 1.05, min_max_values[1] * 1.3)

    axis.set_ylabel("n-Shapley value")

    plt.tight_layout()

    # save plot ------------------------------------------------------------------------------------
    save_path = os.path.join("plots", f"n_shap_sentence_{sentence[:10]}_{budget}.pdf")
    fig.savefig(save_path)

    plt.show()

    color_steps = np.linspace(-1, 1, 20)
    # return the top-k n-Shapley values ------------------------------------------------------------

    n_shapley_values_not_single = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_estimates, n=interaction_order, reduce_one_dimension=False)

    top_k_indices, top_k_values = find_top_k_n_shapley(n_shapley_values_not_single, k=14)

    x_words_arr = np.asarray(x)
    x_input_words = np.asarray(input_words)
    for order in top_k_indices.keys():
        print(f"Order {order}")
        for i in range(len(top_k_indices[order])):
            index = top_k_indices[order][i]
            color_index = np.argmin(abs(color_steps - top_k_values[order][i])) + 1
            print(f"{top_k_values[order][i]} {x_words_arr[index]}  {x_input_words[index]} {color_index}")
    print(sentence)
    print()

    # return the top-k SII values ------------------------------------------------------------
    top_k_indices, top_k_values = find_top_k_n_shapley(sii_estimates, k=5)

    x_words_arr = np.asarray(x)
    x_input_words = np.asarray(input_words)
    for order in top_k_indices.keys():
        print(f"Order {order}")
        for i in range(len(top_k_indices[order])):
            index = top_k_indices[order][i]
            # get clostest color point
            color_index = np.argmin(abs(color_steps - top_k_values[order][i])) + 1
            print(f"{top_k_values[order][i]} {x_words_arr[index]}  {x_input_words[index]} {color_index}")
    print(sentence)
