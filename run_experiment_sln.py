"""This module is used to run experiments given the SparseLinearModels with exact ground-truth values."""
import copy
import os
import time

import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import kendalltau
from tqdm import tqdm

from games import ParameterizedSparseLinearModel
from approximators import SHAPIQEstimator, PermutationSampling
from approximators.regression import RegressionEstimator


def get_approximation_error(approx: np.ndarray, exact: np.ndarray) -> float:
    error = np.sum((approx - exact) ** 2) / binom(N_FEATURES, SHAPLEY_INTERACTION_ORDER)
    return error


def get_precision_at_k(approx: np.ndarray, exact: np.ndarray, k: int = 10):
    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    top_k_id_exact = set(exact_abs.argsort()[-k:])

    approx_abs = np.abs(approx)
    approx_abs = approx_abs.flatten()
    top_k_id_approx = set(approx_abs.argsort()[-k:])

    wrong_ids = len(top_k_id_approx - top_k_id_exact)
    correct_ids = k - wrong_ids
    return float(correct_ids / k)


def get_approximation_error_at_k(approx: np.ndarray, exact: np.ndarray, k: int = 10):
    exact_flat = exact.flatten()
    approx_flat = approx.flatten()

    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    top_k_id_exact = set(exact_abs.argsort()[-k:])

    error = np.sum((approx_flat[[*top_k_id_exact]] - exact_flat[[*top_k_id_exact]]) ** 2)
    return float(error)


def get_kendals_tau(approx: np.ndarray, exact: np.ndarray):
    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    ranking_exact = exact_abs.argsort()
    approx_abs = np.abs(approx)
    approx_abs = approx_abs.flatten()
    ranking_approx = approx_abs.argsort()
    tau, _ = kendalltau(ranking_exact, ranking_approx)
    return tau


def save_values(save_path: str, values: list):
    save_dir = os.path.join(*os.path.split(save_path)[0:-1])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    df = pd.DataFrame(values)
    if not os.path.isfile(save_path):
        df.to_csv(save_path, header=True, index=False)
    else:
        df.to_csv(save_path, mode='a',  header=False, index=False)


if __name__ == "__main__":

    RESULT_DIR = os.path.join("results", "sln")
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    # game settings
    N_FEATURES: int = 30  # number of players
    N_INTERACTIONS: int = 100  # number of interactions in the model
    N_NON_IMPORTANT_FEATURE_RATIO = 0.  # percentage of dummy players (zero value players)

    # SHAP-IQ settings
    SHAPLEY_INTERACTION_ORDER = 2  # interaction order to compute values for
    SAMPLING_KERNELS = ["faith"]  # sampling weights (for drawing subsets)
    PAIRWISE_LIST = [False]  # weather or not to use pairwise sampling (also select the inverse of the subset choosen) or not possible values [True, False]

    # sampling budgets settings
    MAX_BUDGET: int = 2 ** 14  # max computation budget
    BUDGET_STEPS = np.arange(0, 1.05, 0.05)  # step size of computation budgets

    # experiment iterations settings
    ITERATIONS = 10
    INNER_ITERATIONS = 1
    MIN_MAX_INTERACTIONS = [(0, 10), (0, 15), (0, 20), (0, 25), (0, 30)]

    # evaluation settings
    K = 10  # approx at k value

    START_TIME = str(time.time())
    time_id = str(int(time.time()))
    approx_errors_list = []
    # Game Function --------------------------------------------------------------------------------
    for iteration in range(1, ITERATIONS + 1):
        print(f"Starting Iteration {iteration}")

        for MIN_INTERACTIONS, MAX_INTERACTIONS in MIN_MAX_INTERACTIONS:
            print(f"SLN with {N_NON_IMPORTANT_FEATURE_RATIO * 100}% unimportant features and "
                  f"min {MIN_INTERACTIONS}, max {MAX_INTERACTIONS} interactions")
            game = ParameterizedSparseLinearModel(
                n=N_FEATURES,
                weighting_scheme="uniform",
                min_interaction_size=MIN_INTERACTIONS,
                max_interaction_size=MAX_INTERACTIONS,
                n_interactions=N_INTERACTIONS,
                n_non_important_features=int(N_FEATURES * N_NON_IMPORTANT_FEATURE_RATIO)
            )

            game_name = game.game_name
            game_fun = game.set_call
            n = game.n
            N = set(range(n))

            # SAVE paths ---------------------------------------------------------------------------
            SAVE_NAME = '_'.join((
                START_TIME, game_name, str(N_FEATURES), str(SHAPLEY_INTERACTION_ORDER),
                str(N_INTERACTIONS), str(N_NON_IMPORTANT_FEATURE_RATIO),
                str(MIN_INTERACTIONS), str(MAX_INTERACTIONS)
            )) + ".csv"
            SAVE_PATH = os.path.join(RESULT_DIR, SAVE_NAME)

            # Start Iterations ---------------------------------------------------------------------
            for inner_iteration in range(1, INNER_ITERATIONS + 1):
                print(f"Interaction Order: {SHAPLEY_INTERACTION_ORDER}")

                # Budgets --------------------------------------------------------------------------
                total_subsets = 2 ** N_FEATURES
                biggest_budget = min(MAX_BUDGET, total_subsets)
                budgets = [int(budget * biggest_budget) for budget in BUDGET_STEPS]
                all_budgets = sum(budgets)

                # Approximation Estimators ---------------------------------------------------------
                shapley_extractor_sii = SHAPIQEstimator(
                    N, SHAPLEY_INTERACTION_ORDER,
                    min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="SII")
                shapley_extractor_sti = SHAPIQEstimator(
                    N, SHAPLEY_INTERACTION_ORDER,
                    min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="STI")
                shapley_extractor_sfi = SHAPIQEstimator(
                    N, SHAPLEY_INTERACTION_ORDER,
                    min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="SFI")

                approximators = {
                    "SII": shapley_extractor_sii,
                    "STI": shapley_extractor_sti,
                    "SFI": shapley_extractor_sfi
                }

                # Baseline Estimator ---------------------------------------------------------------
                shapley_extractor_sii_permutation = PermutationSampling(
                    N, SHAPLEY_INTERACTION_ORDER,
                    min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="SII")
                shapley_extractor_sti_permutation = PermutationSampling(
                    N, SHAPLEY_INTERACTION_ORDER,
                    min_order=SHAPLEY_INTERACTION_ORDER, interaction_type="STI")
                shapley_extractor_sfi_regression = RegressionEstimator(
                    N, SHAPLEY_INTERACTION_ORDER)

                baselines = {
                    "SII": shapley_extractor_sii_permutation,
                    "STI": shapley_extractor_sti_permutation,
                    "SFI": shapley_extractor_sfi_regression
                }

                # Compute exact interactions -------------------------------------------------------
                if inner_iteration <= 1:
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

                # Approximate ----------------------------------------------------------------------
                print("Starting approximation computations")
                shapx_sampling = {}
                approximation_errors = {}
                results = {}
                precisions = {}
                approximation_errors_at_k = {}
                kendal_taus = {}
                # Each Interaction Index -----------------------------------------------------------
                for interaction_type, approximator in approximators.items():
                    time.sleep(0.1)
                    pbar_budget = all_budgets * len(PAIRWISE_LIST) * len(SAMPLING_KERNELS) + all_budgets
                    pbar = tqdm(total=pbar_budget, desc=interaction_type)
                    exact_values = shapx_exact_values[interaction_type][SHAPLEY_INTERACTION_ORDER]
                    for budget in budgets:
                        relative_budget = round(budget / total_subsets, 2)
                        run_id1 = '_'.join((interaction_type, str(budget), str(relative_budget)))

                        # Baseline Approximations --------------------------------------------------
                        baseline_run_id = '_'.join((run_id1, 'baseline'))
                        baseline_approximator = baselines[interaction_type]
                        approximated_interactions = copy.deepcopy(
                            baseline_approximator.approximate_with_budget(game_fun, budget))
                        approximation_errors[baseline_run_id] = get_approximation_error(
                            approx=approximated_interactions,
                            exact=exact_values
                        )
                        precisions[baseline_run_id] = get_precision_at_k(
                            approx=approximated_interactions,
                            exact=exact_values,
                            k=K
                        )
                        approximation_errors_at_k[baseline_run_id] = get_approximation_error_at_k(
                            approx=approximated_interactions,
                            exact=exact_values,
                            k=K
                        )
                        kendal_taus[baseline_run_id] = get_kendals_tau(
                            approx=approximated_interactions,
                            exact=exact_values
                        )
                        pbar.update(budget)

                        # Sampling Approximations --------------------------------------------------
                        for sampling_kernel in SAMPLING_KERNELS:
                            run_id2 = '_'.join((run_id1, 'approximation', sampling_kernel))

                            # Pairwise Approximations ----------------------------------------------
                            for pairwise in PAIRWISE_LIST:
                                pairwise_id = 'pairwise' if pairwise else 'not-paired'
                                run_id3 = '_'.join((run_id2, pairwise_id))

                                # Const. and Sampling ----------------------------------------------
                                approximated_interactions = copy.deepcopy(
                                    approximator.compute_interactions_from_budget(
                                        game_fun, budget,  pairing=pairwise,
                                        sampling_kernel=sampling_kernel)
                                )
                                results['_'.join((run_id3, 'const and sampling'))] = copy.deepcopy(
                                    approximated_interactions)
                                approximation_errors[
                                    '_'.join((
                                        run_id3, 'const and sampling'))
                                ] = get_approximation_error(
                                    approx=approximated_interactions[SHAPLEY_INTERACTION_ORDER],
                                    exact=exact_values
                                )
                                precisions[
                                    '_'.join((run_id3, 'const and sampling'))
                                ] = get_precision_at_k(
                                    approx=approximated_interactions[SHAPLEY_INTERACTION_ORDER],
                                    exact=exact_values,
                                    k=K
                                )
                                approximation_errors_at_k[
                                    '_'.join((run_id3, 'const and sampling'))] = get_approximation_error_at_k(
                                    approx=approximated_interactions[SHAPLEY_INTERACTION_ORDER],
                                    exact=exact_values,
                                    k=K
                                )
                                kendal_taus[
                                    '_'.join((run_id3, 'const and sampling'))] = get_kendals_tau(
                                    approx=approximated_interactions[SHAPLEY_INTERACTION_ORDER],
                                    exact=exact_values
                                )
                                pbar.update(budget)
                    pbar.close()

                # Store Iteration  -----------------------------------------------------------------
                for approximator_id, approximation_error in approximation_errors.items():
                    run_dict = {}
                    id_parts = approximator_id.split('_')
                    precision_at_k = precisions[approximator_id]
                    approx_value_at_k = approximation_errors_at_k[approximator_id]
                    kendal_tau = kendal_taus[approximator_id]
                    try:
                        run_dict['shapley_interaction_order'] = SHAPLEY_INTERACTION_ORDER
                        run_dict["max_interactions"] = MAX_INTERACTIONS
                        run_dict["min_interactions"] = MIN_INTERACTIONS
                        run_dict["n_interactions"] = N_INTERACTIONS
                        run_dict["n_unimportant"] = N_NON_IMPORTANT_FEATURE_RATIO
                        run_dict['iteration'] = iteration
                        run_dict['inner_iteration'] = inner_iteration
                        run_dict['approx_value'] = approximation_error
                        run_dict['precision_at_k'] = precision_at_k
                        run_dict['approx_value_at_k'] = approx_value_at_k
                        run_dict['kendal_tau'] = kendal_tau
                        run_dict['interaction_index'] = id_parts[0]
                        run_dict['n_absolute'] = id_parts[1]
                        run_dict['n_relative'] = id_parts[2]
                        run_dict['approx_type'] = id_parts[3]
                        run_dict['sampling_kernel'] = id_parts[4]
                        run_dict['pairing'] = id_parts[5]
                        run_dict['sampling'] = id_parts[6]
                    except IndexError:
                        pass
                    approx_errors_list.append(run_dict)

                save_values(SAVE_PATH, approx_errors_list)
                approx_errors_list = []
