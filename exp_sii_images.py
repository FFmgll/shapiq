import os

import pandas as pd

from approximators import SHAPIQEstimator, PermutationSampling
from experiment import run_top_order_experiment
from games import LookUpGame
from utils_experiment import get_gt_values_for_game

if __name__ == "__main__":

    # PARAMETERS -----------------------------------------------------------------------------------
    NUMBER_OF_RUNS = 3
    RUN_TOP_ORDER = False
    ORDER = 4

    # CONSTANTS ------------------------------------------------------------------------------------
    MAX_BUDGET: int = 2 ** 14  # max computation budget

    PAIRING = True
    STRATIFICATION = False
    SAMPLING_KERNEL = "ksh"

    # Setup games ----------------------------------------------------------------------------------

    # initialize games
    game_list = []
    used_ids = set()
    data_folder = "nlp_values"

    for i in range(NUMBER_OF_RUNS):
        game = LookUpGame(data_folder=data_folder, n=14, set_zero=True,
                          used_ids=used_ids)
        game_list.append(game)
        used_ids = game.used_ids

    # get number of players
    n = game_list[0].n
    N = set(range(n))

    # define folder name and save path
    SAVE_FOLDER = os.path.join("results", '_'.join((data_folder, str(n))))
    file_name = '_'.join((str(n), str(NUMBER_OF_RUNS), str(ORDER))) + ".json"
    SAVE_PATH = os.path.join(SAVE_FOLDER, file_name)

    print("Loaded games.")
    print("Number of games: ", len(game_list))

    # Initialize estimators ------------------------------------------------------------------------
    shapiq_estimator = SHAPIQEstimator(N=N, order=ORDER, interaction_type="SII", top_order=RUN_TOP_ORDER)
    baseline_estimator = PermutationSampling(N=N, order=ORDER, interaction_type='SII', top_order=RUN_TOP_ORDER)
    print("Initialized estimators.")

    # Pre-compute the gt values --------------------------------------------------------------------
    print("Precomputing gt values.")
    all_gt_values = {}
    for n, game in enumerate(game_list, start=1):
        gt_values = get_gt_values_for_game(game=game, shapiq=shapiq_estimator, order=ORDER)
        all_gt_values[n] = gt_values

    # Run experiments ------------------------------------------------------------------------------
    print("Starting experiments.")
    RESULTS = run_top_order_experiment(
        top_order=RUN_TOP_ORDER,
        game_list=game_list,
        shapiq_estimator=shapiq_estimator,
        baseline_estimator=baseline_estimator,
        all_gt_values=all_gt_values,
        order=ORDER,
        max_budget=MAX_BUDGET,
        pairing=PAIRING,
        stratification=STRATIFICATION,
        sampling_kernel=SAMPLING_KERNEL,
    )

    # Save results ---------------------------------------------------------------------------------
    print("Saving results.")
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    results_df = pd.DataFrame(RESULTS)
    results_df.to_json(SAVE_PATH)
    print("Done.")
