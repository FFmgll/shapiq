import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from approximators import SHAPIQEstimator
from games import NLPGame, LookUpGame

if __name__ == "__main__":

    #sentence = "this movie was actually better than expected but I see why the critics did not like the acting"
    #game = NLPGame(input_text=sentence)
    #print("Original Output:", game.original_output, "n:", game.n)

    data_set_name = "feature_selection"
    game_list = []
    game = LookUpGame(data_folder=data_set_name, n=12, set_zero=True)

    # get number of players
    n = game.n
    N = set(range(n))

    interaction_order = 4

    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)

    budget = min(2 ** n, 2 ** 14)

    sii_estimates = shap_iq.compute_interactions_from_budget(
        budget=budget, game=game.set_call, pairing=True)

    n_shapley_values = shap_iq.transform_interactions_in_n_shapley(
        interaction_values=sii_estimates, n=interaction_order, reduce_one_dimension=True)
    n_shapley_values_pos, n_shapley_values_neg = n_shapley_values

    x = np.arange(n)

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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
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

    axis.set_title(f"N-Shapley Values for {data_set_name} with budget {budget}")
    axis.legend()
    plt.show()
