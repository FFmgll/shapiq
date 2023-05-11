import copy
from typing import Union

import numpy as np
import pandas as pd
import tqdm

from approximators import SHAPIQEstimator, PermutationSampling, RegressionEstimator
from utils_experiment import get_gt_values_for_game, get_all_errors


def run_top_order_experiment(
        top_order: bool,
        game_list: list,
        shapiq_estimator: SHAPIQEstimator,
        baseline_estimator: Union[PermutationSampling, RegressionEstimator],
        all_gt_values: dict,
        max_budget: int,
        order: int,
        sampling_kernel='ksh',
        stratification=False,
        pairing=True,
        budget_steps: list = None
) -> list:
    """Computes the experiment for a given list of games and shapiq estiamtors."""

    # get the budget list
    if budget_steps is None:
        budget_steps = np.arange(0, 1.05, 0.05)  # step size of computation budgets

    RESULTS = []
    pbar = tqdm.tqdm(total=np.sum(budget_steps * max_budget) * len(game_list) * 2)

    for budget_step in budget_steps:
        budget = int(budget_step * max_budget)

        budget_errors_shapiq = []
        budget_errors_baseline = []
        for i, game in enumerate(game_list, start=1):
            n = game.n
            # get the correct gt_values
            gt_values = all_gt_values[i]

            # approximate with shapiq
            shap_iq_approx = shapiq_estimator.compute_interactions_from_budget(
                game=game.set_call, budget=budget,
                sampling_kernel=sampling_kernel, pairing=pairing, stratification=stratification)
            pbar.update(budget)

            # approximate with baseline
            baseline_approx = baseline_estimator.approximate_with_budget(
                game=game.set_call, budget=budget)
            pbar.update(budget)

            # get errors and append to list
            errors_shapiq = get_all_errors(shap_iq_approx, gt_values, n=n, order=order, top_order=top_order)
            errors_baseline = get_all_errors(baseline_approx, gt_values, n=n, order=order, top_order=top_order)
            for order in errors_shapiq.keys():
                budget_errors_shapiq.append(errors_shapiq[order])
                budget_errors_baseline.append(errors_baseline[order])

        budget_errors_shapiq_df = pd.DataFrame(budget_errors_shapiq)
        budget_errors_baseline_df = pd.DataFrame(budget_errors_baseline)

        # compute mean, std, and var
        mean_shapiq = dict(budget_errors_shapiq_df.mean())
        std_shapiq = dict(budget_errors_shapiq_df.std())
        var_shapiq = dict(budget_errors_shapiq_df.var())

        mean_baseline = dict(budget_errors_baseline_df.mean())
        std_baseline = dict(budget_errors_baseline_df.std())
        var_baseline = dict(budget_errors_baseline_df.var())

        # append to results
        RESULTS.append({
            'budget': budget,
            'mean_shapiq': mean_shapiq, 'std_shapiq': std_shapiq, 'var_shapiq': var_shapiq,
            'mean_baseline': mean_baseline, 'std_baseline': std_baseline, 'var_baseline': var_baseline
        })
    return copy.deepcopy(RESULTS)
