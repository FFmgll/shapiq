import copy

import numpy as np

from approximators.regression import RegressionEstimator
from approximators.shapiq import SHAPIQEstimator, SHAPIQEstimator
from games import NLPLookupGame, ParameterizedSparseLinearModel
from pprint import pprint


if __name__ == "__main__":

    budget = 10_000
    interaction_order = 2

    game = ParameterizedSparseLinearModel(n=30, n_interactions=75, min_interaction_size=1, max_interaction_size=30, weighting_scheme="uniform")
    #game = NLPLookupGame(n=14, sentence_id=588)
    n = game.n
    game_fun = game.set_call
    N = set(range(0, n))

    # get exact values
    exact_estimator = SHAPIQEstimator(N, interaction_order, interaction_order, "SFI")
    try:
        exact_values = game.exact_values(gamma_matrix=exact_estimator.weights, max_order=interaction_order, min_order=interaction_order)[interaction_order]
    except AttributeError:
        exact_values = copy.deepcopy(exact_estimator.compute_interactions_complete(game_fun))[interaction_order]
    pprint(exact_values)

    # get approximated values with the regression approximator
    est = RegressionEstimator(N, interaction_order)
    phi_regression = est.approximate_with_budget(game_fun=game_fun, budget=budget)
    pprint(phi_regression)

    # get approximated values with the SHAPIQ approximator
    shapiq = SHAPIQEstimator(N, interaction_order, interaction_order, "SFI")
    phi_shapiq = shapiq.compute_interactions_from_budget(game=game_fun, budget=budget)[interaction_order]
    pprint(phi_shapiq)

    # compute MSE
    mse_regression = np.sum((exact_values - phi_regression) ** 2)
    mse_shapiq = np.sum((exact_values - phi_shapiq) ** 2)

    print("MSE - Regression:", mse_regression)
    print("MSE - SHAPIQ:", mse_shapiq)
