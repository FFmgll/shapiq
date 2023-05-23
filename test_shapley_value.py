import copy

import numpy as np

from approximators.exact_sv import ExhaustiveSearch
from approximators.regression import RegressionEstimator
from approximators.shapiq import SHAPIQEstimator
from games import ParameterizedSparseLinearModel
from pprint import pprint


if __name__ == "__main__":

    budget = 2**14
    interaction_order = 1

    game = ParameterizedSparseLinearModel(n=14, n_interactions=50, min_interaction_size=1,
                                          max_interaction_size=14, weighting_scheme="uniform")
    n = game.n
    game_fun = game.set_call
    N = set(range(0, n))

    # get exact values
    shap_iq_estimator = SHAPIQEstimator(N, interaction_order, "FSI")
    try:
        exact_values = game.exact_values(gamma_matrix=shap_iq_estimator.weights[interaction_order], max_order=interaction_order, min_order=interaction_order)[interaction_order]
    except AttributeError:
        exact_values = copy.deepcopy(shap_iq_estimator.compute_interactions_complete(game_fun))[interaction_order]
    pprint(exact_values)

    # get approximated values with the regression approximator
    est = RegressionEstimator(N, interaction_order)
    phi_regression = est.approximate_with_budget(game_fun=game_fun, budget=budget)[interaction_order]
    pprint(phi_regression)

    # get approximated values with the SHAPIQ approximator
    phi_shapiq = shap_iq_estimator.compute_interactions_from_budget(game=game_fun, budget=budget)[interaction_order]
    pprint(phi_shapiq)

    # compute sv via exhaustive search
    exhaustive_search = ExhaustiveSearch(N)
    phi_exhaustive = exhaustive_search.compute_exact_shapley_values(game=game_fun)
    pprint(phi_exhaustive)

    # compute MSE
    mse_regression = np.sum((exact_values - phi_regression) ** 2)
    mse_shapiq = np.sum((exact_values - phi_shapiq) ** 2)
    mse_exhaustive = np.sum((exact_values - phi_exhaustive) ** 2)

    print("MSE - Regression:", mse_regression)
    print("MSE - SHAPIQ:", mse_shapiq)
    print("MSE - Exhaustive Search:", mse_exhaustive)
