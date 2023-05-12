import copy
import os
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
        budget_steps: list = None,
        save_folder: str = None,
        save_path: str = None,
) -> dict:
    """Computes the experiment for a given list of games and shapiq estiamtors."""

    # get the budget list
    if budget_steps is None:
        budget_steps = np.arange(0, 1.05, 0.05)  # step size of computation budgets

    RESULTS = {'shapiq': {}, 'baseline': {}}  # initialize results dict
    pbar = tqdm.tqdm(total=np.sum(budget_steps * max_budget) * len(game_list) * 2)

    for budget_step in budget_steps:
        budget = int(budget_step * max_budget)

        budget_errors_shapiq = {}
        budget_errors_baseline = {}
        for i, game in enumerate(game_list, start=1):
            n = game.n
            # get the correct gt_values
            gt_values = all_gt_values[i]

            # approximate with shapiq
            shap_iq_approx = shapiq_estimator.compute_interactions_from_budget(
                game.set_call, budget,
                sampling_kernel=sampling_kernel, pairing=pairing, stratification=stratification)
            pbar.update(budget)

            # approximate with baseline
            baseline_approx = baseline_estimator.approximate_with_budget(game.set_call, budget)
            pbar.update(budget)

            # get errors and append to list
            errors_shapiq = get_all_errors(shap_iq_approx, gt_values, n=n, order=order, top_order=top_order)
            errors_baseline = get_all_errors(baseline_approx, gt_values, n=n, order=order, top_order=top_order)

            # append to dict
            for order_ in errors_shapiq.keys():
                try:
                    budget_errors_shapiq[order_].append(errors_shapiq[order_])
                    budget_errors_baseline[order_].append(errors_baseline[order_])
                except KeyError:
                    budget_errors_shapiq[order_] = [errors_shapiq[order_]]
                    budget_errors_baseline[order_] = [errors_baseline[order_]]

        for order_ in budget_errors_shapiq.keys():
            errors_shapiq_df = pd.DataFrame(budget_errors_shapiq[order_])
            errors_baseline_df = pd.DataFrame(budget_errors_baseline[order_])

            # compute mean, std, and var
            mean_shapiq = dict(errors_shapiq_df.mean())
            std_shapiq = dict(errors_shapiq_df.std())
            var_shapiq = dict(errors_shapiq_df.var())

            mean_baseline = dict(errors_baseline_df.mean())
            std_baseline = dict(errors_baseline_df.std())
            var_baseline = dict(errors_baseline_df.var())

            # append to results
            dict_to_append_shapiq = {
                'budget': budget, 'mean': mean_shapiq, 'std': std_shapiq, 'var': var_shapiq
            }
            dict_to_append_baseline = {
                'budget': budget, 'mean': mean_baseline, 'std': std_baseline, 'var': var_baseline
            }
            try:
                RESULTS['shapiq'][order_].append(dict_to_append_shapiq)
                RESULTS['baseline'][order_].append(dict_to_append_baseline)
            except KeyError:
                RESULTS['shapiq'][order_] = [dict_to_append_shapiq]
                RESULTS['baseline'][order_] = [dict_to_append_baseline]

        # save results to json
        if save_folder is not None and save_path is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            pd.DataFrame(RESULTS).to_json(save_path)

    pbar.close()
    return copy.deepcopy(RESULTS)
