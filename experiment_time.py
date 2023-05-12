"""This module is used to run experiments given any game function, where ground-truth values are computed via brute force."""
import copy
import os

import time

import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import kendalltau
from tqdm import tqdm

from games import ParameterizedSparseLinearModel, BaseSparseLinearModel, SyntheticNeuralNetwork, \
    NLPLookupGame, NLPGame, MachineLearningGame, MachineLearningMetaGame
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
    if not os.path.iFSIle(save_path):
        df.to_csv(save_path, header=True, index=False)
    else:
        df.to_csv(save_path, mode='a',  header=False, index=False)


if __name__ == "__main__":
    PLOT = True  # weather or not to plot in the end of the experiment
    INDEPENDENT_ITERATIONS = False  # controls if each iteration is saved independetly (one at a time for running on cluster)

    RESULT_DIR = os.path.join("results_old", "time")
    if not os.path.exists(RESULT_DIR):
        os.mkdir(RESULT_DIR)

    # game settings
    #game = NLPGame(input_text='I like this movie a lot but it is cool wow')

    meta_game = MachineLearningMetaGame(dataset_name='adult')
    game = MachineLearningGame(meta_game)
    N_FEATURES = game.n
    print(N_FEATURES)

    #N_FEATURES: int = 14  # player size
    used_ids = set()  # only used for LookUpGames such that we draw a new game every iteration

    # SHAP-IQ settings
    SHAPLEY_INTERACTION_ORDER = 2  # interaction order to compute values for
    SAMPLING_KERNELS = ["faith"]  # sampling weights (for drawing subsets)
    PAIRWISE_LIST = [False]  # weather or not to use pairwise sampling (also select the inverse of the subset choosen) or not possible values [True, False]

    # sampling budgets settings
    MAX_BUDGET: int = 5000  # max computation budget
    BUDGET_STEPS = [0.5, 1.05]  # step size of computation budgets

    # experiment iterations settings
    ITERATIONS = 2  # number of experiments to run
    INNER_ITERATIONS = 1  # number of runs of the same game function (to sample different subsets)

    # evaluation settings
    K = 10  # approx at k value

    START_TIME = str(time.time())
    approx_errors_list = []
    run_times_list = []
    for iteration in range(1, ITERATIONS + 1):
        print(f"Starting Iteration {iteration}")
        #game = NLPLookupGame(n=N_FEATURES, set_zero=True, used_ids=used_ids)

        #game = SyntheticNeuralNetwork(n=N_FEATURES, set_zero=True)
        #game = ParameterizedSparseLinearModel(
        #    n=30, weighting_scheme="uniform", min_interaction_size=1,
       #     max_interaction_size=5, n_interactions=100, n_non_important_features=0
        #)

        try:
            used_ids = game.used_ids
        except AttributeError:
            pass
        game_name = game.game_name
        game_fun = game.set_call

        # SAVE paths -------------------------------------------------------------------------------
        SAVE_NAME = '_'.join((START_TIME, game_name, str(N_FEATURES), str(SHAPLEY_INTERACTION_ORDER))) + ".csv"
        if INDEPENDENT_ITERATIONS:
            SAVE_NAME = '_'.join((START_TIME, game_name, str(N_FEATURES), str(SHAPLEY_INTERACTION_ORDER), str(iteration))) + ".csv"
        SAVE_PATH = os.path.join(RESULT_DIR, SAVE_NAME)

        for inner_iteration in range(1, INNER_ITERATIONS + 1):
            print(f"Interaction Order: {SHAPLEY_INTERACTION_ORDER}")

            # Game Parameters ----------------------------------------------------------------------
            n = game.n
            N = set(range(n))

            # Budgets ------------------------------------------------------------------------------
            total_subsets = 2 ** N_FEATURES
            biggest_budget = min(MAX_BUDGET, total_subsets)
            budgets = [int(budget * biggest_budget) for budget in BUDGET_STEPS]
            all_budgets = sum(budgets)

            # Approximation Estimators -------------------------------------------------------------
            shapley_extractor_sii = SHAPIQEstimator(N, SHAPLEY_INTERACTION_ORDER,
                                                    interaction_type="SII")
            shapley_extractor_sti = SHAPIQEstimator(N, SHAPLEY_INTERACTION_ORDER,
                                                    interaction_type="STI")
            shapley_extractor_FSI = SHAPIQEstimator(N, SHAPLEY_INTERACTION_ORDER,
                                                    interaction_type="FSI")

            approximators = {
                "SII": shapley_extractor_sii,
                "STI": shapley_extractor_sti,
                "FSI": shapley_extractor_FSI
            }

            # Baseline Estimator -------------------------------------------------------------------
            shapley_extractor_sii_permutation = PermutationSampling(N, SHAPLEY_INTERACTION_ORDER,
                                                                    interaction_type="SII")
            shapley_extractor_sti_permutation = PermutationSampling(N, SHAPLEY_INTERACTION_ORDER,
                                                                    interaction_type="STI")
            shapley_extractor_FSI_regression = RegressionEstimator(
                N, SHAPLEY_INTERACTION_ORDER)

            baselines = {
                "SII": shapley_extractor_sii_permutation,
                "STI": shapley_extractor_sti_permutation,
                "FSI": shapley_extractor_FSI_regression
            }

            # Approximate --------------------------------------------------------------------------
            print("Starting approximation computations")
            shapx_sampling = {}
            approximation_errors = {}
            run_times = {}
            # Each Interaction Index ---------------------------------------------------------------
            for interaction_type, approximator in approximators.items():
                time.sleep(0.1)
                pbar_budget = all_budgets * len(PAIRWISE_LIST) * len(SAMPLING_KERNELS) + all_budgets
                pbar = tqdm(total=pbar_budget, desc=interaction_type)
                for budget in budgets:
                    relative_budget = round(budget / total_subsets, 2)
                    run_id1 = '_'.join((interaction_type, str(budget), str(relative_budget)))

                    # Baseline Approximations ------------------------------------------------------
                    baseline_run_id = '_'.join((run_id1, 'baseline'))
                    baseline_approximator = baselines[interaction_type]
                    start_run_time = time.time()
                    _ = baseline_approximator.approximate_with_budget(game_fun, budget)
                    end_run_time = time.time() - start_run_time
                    run_times[baseline_run_id] = end_run_time
                    approximation_errors[baseline_run_id] = 0
                    pbar.update(budget)

                    # Sampling Approximations ------------------------------------------------------
                    for sampling_kernel in SAMPLING_KERNELS:
                        run_id2 = '_'.join((run_id1, 'approximation', sampling_kernel))

                        # Pairwise Approximations --------------------------------------------------
                        for pairwise in PAIRWISE_LIST:
                            pairwise_id = 'pairwise' if pairwise else 'not-paired'
                            run_id3 = '_'.join((run_id2, pairwise_id))

                            # Const. and Sampling --------------------------------------------------
                            start_run_time = time.time()
                            _ = approximator.compute_interactions_from_budget(
                                game_fun, budget,  pairing=pairwise, sampling_kernel=sampling_kernel
                            )

                            end_run_time = time.time() - start_run_time
                            run_times['_'.join((run_id3, 'const and sampling'))] = end_run_time
                            approximation_errors['_'.join((run_id3, 'const and sampling'))] = 0
                            pbar.update(budget)
                pbar.close()

            # Store Iteration  ---------------------------------------------------------------------
            for approximator_id, approximation_error in approximation_errors.items():
                run_dict = {}
                id_parts = approximator_id.split('_')
                try:
                    run_time = run_times[approximator_id]
                except KeyError:
                    run_time = 0.
                try:
                    run_dict['shapley_interaction_order'] = SHAPLEY_INTERACTION_ORDER
                    run_dict['iteration'] = iteration
                    run_dict['inner_iteration'] = inner_iteration
                    run_dict['approx_value'] = approximation_error
                    run_dict['run_time'] = run_time
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

    df = pd.read_csv(SAVE_PATH)
    df_times = df[df['run_time'] > 0.]
    df_grouped = df_times.groupby(by=['approx_type', 'interaction_index', 'n_absolute']).aggregate(
        {'run_time': ['mean', 'std']}).reset_index()