import numpy as np

from approximators import SHAPIQEstimator, PermutationSampling
from games import LookUpGame
from utils_experiment import get_gt_values_for_game, get_all_errors

if __name__ == "__main__":

    # PARAMETERS -----------------------------------------------------------------------------------
    NUMBER_OF_RUNS = 50
    RUN_TOP_ORDER = True
    ORDER = 4

    # CONSTANTS ------------------------------------------------------------------------------------
    MAX_BUDGET: int = 2 ** 14  # max computation budget
    BUDGET_STEPS = np.arange(0, 1.05, 0.05)  # step size of computation budgets

    PAIRING = True
    STRATIFICATION = False
    SAMPLING_KERNEL = "ksh"

    # Setup games ----------------------------------------------------------------------------------

    # initialize games
    game_list = []
    used_ids = set()
    data_folder = "image_classifier"
    for i in range(NUMBER_OF_RUNS):
        game = LookUpGame(data_folder=data_folder, n=14, set_zero=True,
                          used_ids=used_ids)
        game_list.append(game)
        used_ids = game.used_ids

    # get number of players
    n = game_list[0].n
    N = set(range(n))

    print("Loaded games.")
    print("Number of games: ", len(game_list))

    RESULTS = {}
    print("Running top order Experiment.")

    # Initialize estimators ------------------------------------------------------------------------
    shapiq_estimator = SHAPIQEstimator(N=N, order=ORDER, interaction_type="SII", top_order=RUN_TOP_ORDER)
    baseline_estimator = PermutationSampling(N=N, order=ORDER, interaction_type='SII', top_order=RUN_TOP_ORDER)
    print("Initialized estimators.")

    # Run experiments ------------------------------------------------------------------------------
    gt_values = {}
    for n, game in enumerate(game_list):
        gt_values = get_gt_values_for_game(game=game, shapiq=shapiq_estimator, order=ORDER)

        for budget_step in BUDGET_STEPS:
            budget = int(budget_step * MAX_BUDGET)

            # approximate with shapiq
            shap_iq_approx = shapiq_estimator.compute_interactions_from_budget(
                game=game.set_call, budget=budget,
                sampling_kernel=SAMPLING_KERNEL, pairing=PAIRING, stratification=STRATIFICATION)

            # approximate with baseline
            baseline_approx = baseline_estimator.approximate_with_budget(
                game=game.set_call, budget=budget)

            errors_shapiq = get_all_errors(shap_iq_approx, gt_values, n=n, order=ORDER)
            errors_baseline = get_all_errors(baseline_approx, gt_values, n=n, order=ORDER)
