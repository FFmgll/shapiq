import sys
platform_os = sys.platform
if platform_os == "linux" or platform_os == "linux2":
    sys.path.insert(0, '/dss/dsshome1/lxc04/ra93yex2/shapiq')
    sys.path.insert(0, '/dss/dsshome1/lxc04/ra93yex2/shapiq/games')
    sys.path.insert(0, '/dss/dsshome1/lxc04/ra93yex2/shapiq/approximators')

import random
import os
import time

import numpy as np
import pandas as pd

from approximators import SHAPIQEstimator, PermutationSampling, RegressionEstimator
from experiment import run_top_order_experiment
from games import LookUpGame, ParameterizedSparseLinearModel
from utils_experiment import get_gt_values_for_game

import warnings
warnings.filterwarnings("ignore")


def run_experiment(max_budget: int):

    # CONSTANTS ------------------------------------------------------------------------------------
    MAX_BUDGET = max_budget
    BUDGET_STEPS = list(np.arange(0.05, 1.05, 0.05))  # step size of computation budgets

    PAIRING = False
    STRATIFICATION = False
    SAMPLING_KERNEL = "ksh"

    # Setup games ----------------------------------------------------------------------------------

    # initialize games
    game_list = []
    used_ids = set()
    for i in range(NUMBER_OF_RUNS):
        game = LookUpGame(data_folder=data_folder, n=data_n, set_zero=True, used_ids=used_ids)
        game_list.append(game)
        used_ids = game.used_ids

    # get number of players
    n = game_list[0].n
    N = set(range(n))

    # define folder name and save path
    SAVE_FOLDER = os.path.join("results", '_'.join((data_folder, str(n))), interaction_index)
    file_name = '_'.join((
        f"n-{n}", f"runs-{NUMBER_OF_RUNS}", f"s0-{order}", f"top-order-{RUN_TOP_ORDER}",
        f"pairing-{PAIRING}", f"stratification-{STRATIFICATION}", f"weights-{SAMPLING_KERNEL}"
    ))
    if singleton_run:
        file_name += f"_{time.time()}"
    file_name += ".json"
    SAVE_PATH = os.path.join(SAVE_FOLDER, file_name)

    print("Loaded games.")
    print("Number of games: ", len(game_list))

    # Initialize estimators ------------------------------------------------------------------------
    # SHAP-IQ estimator for all three indices
    shapiq_estimator = SHAPIQEstimator(
        N=N, order=order, interaction_type=interaction_index, top_order=RUN_TOP_ORDER)

    # get baseline estimator
    if interaction_index in ['SII', 'STI']:
        # for SII and STI we use the permutation sampling estimator
        baseline_estimator = PermutationSampling(
            N=N, order=order, interaction_type=interaction_index, top_order=RUN_TOP_ORDER)
    else:
        # for SFI we use the regression estimator
        baseline_estimator = RegressionEstimator(
            N=N, max_order=order)
    print("Initialized estimators.")

    # Pre-compute the gt values --------------------------------------------------------------------
    print("Precomputing gt values.")
    all_gt_values = {}
    for n, game in enumerate(game_list, start=1):
        gt_values = get_gt_values_for_game(game=game, shapiq=shapiq_estimator, order=order)
        all_gt_values[n] = gt_values

    # Run experiments ------------------------------------------------------------------------------
    print("Starting experiments.")
    RESULTS = run_top_order_experiment(
        top_order=RUN_TOP_ORDER,
        game_list=game_list,
        shapiq_estimator=shapiq_estimator,
        baseline_estimator=baseline_estimator,
        all_gt_values=all_gt_values,
        order=order,
        max_budget=MAX_BUDGET,
        pairing=PAIRING,
        stratification=STRATIFICATION,
        sampling_kernel=SAMPLING_KERNEL,
        budget_steps=BUDGET_STEPS,
        save_path=SAVE_PATH,
        save_folder=SAVE_FOLDER
    )

    # Save results ---------------------------------------------------------------------------------
    print("Saving results.")
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    results_df = pd.DataFrame(RESULTS)
    results_df.to_json(SAVE_PATH)
    print("Done.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    run_parameters = sys.argv
    has_no_param = not len(run_parameters) > 1
    print("run parameters", run_parameters)

    # PARAMETERS -----------------------------------------------------------------------------------
    data_folder = "image_classifier" if has_no_param else run_parameters[1]
    data_n = 14 if has_no_param else int(run_parameters[2])
    interaction_index = "STI" if has_no_param else run_parameters[3]
    RUN_TOP_ORDER = False if has_no_param else run_parameters[4] == "True"
    order = 4 if has_no_param else int(run_parameters[5])
    NUMBER_OF_RUNS = 1 if has_no_param else int(run_parameters[6])
    try:
        singleton_run = bool(run_parameters[7])
    except IndexError:
        singleton_run = False
    try:
        MAX_BUDGET = int(run_parameters[8])
    except IndexError:
        MAX_BUDGET = 2**data_n

    print("selected parameters:", data_folder, data_n, interaction_index, RUN_TOP_ORDER, order, NUMBER_OF_RUNS, singleton_run, MAX_BUDGET)

    if singleton_run:
        for i in range(NUMBER_OF_RUNS):
            seconds_sleep = random.randint(1, 7)
            print(f"Sleeping for {seconds_sleep} seconds.")
            time.sleep(seconds_sleep)
            NUMBER_OF_RUNS = 1
            run_experiment(max_budget=MAX_BUDGET)
    else:
        run_experiment(max_budget=MAX_BUDGET)
