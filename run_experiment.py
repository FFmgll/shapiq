import copy
import os

import numpy as np
import pandas as pd
from scipy.special import binom
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from games import ParameterizedSparseLinearModel, SparseLinearModel
from shapx import ShapleyInteractionsEstimator, PermutationSampling
from shapx.regression import RegressionEstimator


def get_approximation_error(approx: np.ndarray, exact: np.ndarray, eps: float = 0.00001) -> float:
    error = np.sum((approx - exact) ** 2) / binom(N_FEATURES, SHAPLEY_INTERACTION_ORDER)
    error = 0. if error < eps else error  # For pretty printing ...
    return error


if __name__ == "__main__":

    MAX_BUDGET: int = 10_000
    BUDGET_STEPS = np.arange(0, 1.05, 0.05)
    SHAPLEY_INTERACTION_ORDERS: list = [2]
    ITERATIONS = 5
    SAMPLING_KERNELS = ["faith"]
    PAIRWISE_LIST = [False]

    approx_errors_list = []
    for iteration in range(1, ITERATIONS + 1):
        print(f"Starting Iteration {iteration}")

        # Game Function ----------------------------------------------------------------------------
        N_FEATURES: int = 10
        game = ParameterizedSparseLinearModel(
            n=N_FEATURES, weighting_scheme="uniform", n_interactions=20, max_interaction_size=5)
        #game = SparseLinearModel(
        #    n=N_FEATURES, n_interactions_per_order={1: 10, 2: 20, 3: 20}, n_non_important_features=0)
        game_name = game.game_name
        game_fun = game.set_call

        for SHAPLEY_INTERACTION_ORDER in SHAPLEY_INTERACTION_ORDERS:
            print(f"Interaction Order: {SHAPLEY_INTERACTION_ORDER}")

            # Game Parameters ----------------------------------------------------------------------
            n = game.n
            N = set(range(n))

            # Budgets ------------------------------------------------------------------------------
            total_subsets = 2 ** N_FEATURES
            budgets = [int(budget * min(MAX_BUDGET, total_subsets)) for budget in BUDGET_STEPS]
            all_budgets = sum(budgets)

            # Approximation Estimators -------------------------------------------------------------
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

            # Baseline Estimator -------------------------------------------------------------------
            shapley_extractor_sii_permutation = PermutationSampling(
                N, SHAPLEY_INTERACTION_ORDER, min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="SII")
            shapley_extractor_sti_permutation = PermutationSampling(
                N, SHAPLEY_INTERACTION_ORDER, min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="STI")
            shapley_extractor_sfi_regression = RegressionEstimator(
                N, SHAPLEY_INTERACTION_ORDER)

            baselines = {
                "SII": shapley_extractor_sii_permutation,
                "STI": shapley_extractor_sti_permutation,
                "SFI": shapley_extractor_sfi_regression
            }

            # Compute exact interactions -----------------------------------------------------------
            print("Starting exact computations")
            shapx_exact_values = {}
            for interaction_type, approximator in approximators.items():
                if hasattr(game, "exact_values"):
                    print("Exact values from game are used.")
                    shapx_exact_values[interaction_type] = copy.deepcopy(
                        game.exact_values(gamma_matrix=approximator.weights,
                                          min_order=SHAPLEY_INTERACTION_ORDER,
                                          max_order=SHAPLEY_INTERACTION_ORDER)
                    )
                else:
                    print("Exact values are calculated via brute force.")
                    shapx_exact_values[interaction_type] = copy.deepcopy(
                        approximator.compute_interactions_complete(game_fun)
                    )
            print("Exact computations finished")

            # Approximate --------------------------------------------------------------------------
            print("Starting approximation computations")
            shapx_sampling = {}
            approximation_errors = {}
            results = {}
            # Each Interaction Index ---------------------------------------------------------------
            for interaction_type, approximator in approximators.items():
                pbar_budget = all_budgets * len(PAIRWISE_LIST) * len(SAMPLING_KERNELS) + all_budgets
                pbar = tqdm(total=pbar_budget, desc=interaction_type)
                exact_values = shapx_exact_values[interaction_type][SHAPLEY_INTERACTION_ORDER]
                for budget in budgets:
                    relative_budget = round(budget / total_subsets, 2)
                    run_id1 = '_'.join((interaction_type, str(budget), str(relative_budget)))

                    # Baseline Approximations ------------------------------------------------------
                    baseline_run_id = '_'.join((run_id1, 'baseline'))
                    baseline_approximator = baselines[interaction_type]
                    approximated_interactions = copy.deepcopy(
                        baseline_approximator.approximate_with_budget(game_fun, budget))
                    approximation_errors[baseline_run_id] = get_approximation_error(
                        approx=approximated_interactions,
                        exact=exact_values
                    )
                    pbar.update(budget)

                    # Sampling Approximations ------------------------------------------------------
                    for sampling_kernel in SAMPLING_KERNELS:
                        run_id2 = '_'.join((run_id1, 'approximation', sampling_kernel))

                        # Pairwise Approximations --------------------------------------------------
                        for pairwise in PAIRWISE_LIST:
                            pairwise_id = 'pairwise' if pairwise else 'not-paired'
                            run_id3 = '_'.join((run_id2, pairwise_id))

                            # Const. and Sampling --------------------------------------------------
                            approximated_interactions = copy.deepcopy(
                                approximator.compute_interactions_from_budget(
                                    game_fun, budget,  pairing=pairwise,
                                    sampling_kernel=sampling_kernel, sampling_only=False)
                            )
                            results['_'.join((run_id3, 'not full'))] = approximated_interactions
                            approximation_errors['_'.join((run_id3, 'not full'))] = get_approximation_error(
                                approx=approximated_interactions[SHAPLEY_INTERACTION_ORDER],
                                exact=exact_values
                            )

                            # Const. Approximations ------------------------------------------------
                            approximated_interactions = copy.deepcopy(
                                approximator.last_const_complete[SHAPLEY_INTERACTION_ORDER]
                            )
                            results['_'.join((run_id3, 'full'))] = approximated_interactions
                            approximation_errors['_'.join((run_id3, 'full'))] = get_approximation_error(
                                approx=approximator.last_const_complete[SHAPLEY_INTERACTION_ORDER],
                                exact=exact_values
                            )
                            pbar.update(budget)
                pbar.close()

            # Store Iteration  ---------------------------------------------------------------------
            for approximator_id, approximation_error in approximation_errors.items():
                run_dict = {}
                id_parts = approximator_id.split('_')
                try:
                    run_dict['shapley_interaction_order'] = SHAPLEY_INTERACTION_ORDER
                    run_dict['iteration'] = iteration
                    run_dict['approx_value'] = approximation_error
                    run_dict['interaction_index'] = id_parts[0]
                    run_dict['n_absolute'] = id_parts[1]
                    run_dict['n_relative'] = id_parts[2]
                    run_dict['approx_type'] = id_parts[3]
                    run_dict['sampling_kernel'] = id_parts[4]
                    run_dict['pairing'] = id_parts[5]
                    run_dict['full'] = id_parts[6]
                except IndexError:
                    pass
                approx_errors_list.append(run_dict)

    # Store All ------------------------------------------------------------------------------------
    save_name = "_".join((game_name, str(N_FEATURES))) + ".csv"
    approx_errors_df = pd.DataFrame(approx_errors_list)
    approx_errors_df.to_csv(os.path.join("results", save_name), index=False)

    # Plot -----------------------------------------------------------------------------------------

    # Full -----------------------------------------------------------------------------------------
    data = pd.read_csv(os.path.join("results", save_name))
    shapley_interaction_order = str(data["shapley_interaction_order"][0])
    data = data[data['full'].isin(['full', np.NAN])].drop(columns=['full'])
    data = data[data['n_absolute'] > 0]

    data = data.rename(columns={"interaction_index": "Interaction Index", "approx_type": "Method"})

    plot_title = ' '.join(("Order:", shapley_interaction_order, "(only constant)"))
    ax = sns.lineplot(
        x='n_absolute',
        y="approx_value",
        hue="Interaction Index",
        style='Method',
        data=data
    )
    ax.set(
        ylabel='Approximation Error',
        xlabel='N Samples (absolute)',
        title=plot_title
    )
    plt.show()

    # Not Full -------------------------------------------------------------------------------------
    data = pd.read_csv(os.path.join("results", save_name))
    shapley_interaction_order = str(data["shapley_interaction_order"][0])
    data = data[data['full'].isin(['not full', np.NAN])].drop(columns=['full'])
    data = data[data['n_absolute'] > 0]

    data = data.rename(columns={"interaction_index": "Interaction Index", "approx_type": "Method"})

    plot_title = ' '.join(("Order:", shapley_interaction_order, "(with sampling)"))
    ax = sns.lineplot(
        x='n_absolute',
        y="approx_value",
        hue="Interaction Index",
        style='Method',
        data=data
    )
    ax.set(
        ylabel='Approximation Error',
        xlabel='N Samples (absolute)',
        title=plot_title
    )
    plt.show()
