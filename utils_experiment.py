"""This file contains utility functions for the project."""
from typing import Dict

import numpy as np
from scipy.special import binom
from scipy.stats import kendalltau

from approximators import SHAPIQEstimator


def get_all_errors(approx, exact, n: int, order: int):
    """Computes all errors for the given approximation and exact values."""
    errors = {}
    if type(exact) == dict and type(approx) == dict:
        for order in exact.keys():
            errors[order] = _get_all_errors_arr(approx[order], exact[order], n, order)
    else:
        errors = _get_all_errors_arr(approx, exact, n, order)
    return errors


def _get_all_errors_arr(approx: np.ndarray, exact: np.ndarray, n: int, order: int):
    """Helper for computing all errors for the given approximation and exact values."""
    errors = {}
    errors["approximation_error"] = get_approximation_error(approx, exact, n, order)
    errors["precision_at_10"] = get_precision_at_k(approx, exact, k=10)
    errors["approximation_error_at_10"] = get_approximation_error_at_k(approx, exact, k=10)
    errors["kendals_tau"] = get_kendals_tau(approx, exact)
    return errors


def get_approximation_error(approx: np.ndarray, exact: np.ndarray, n: int, order: int) -> float:
    """Computes the approximation error of the given approximation and exact values."""
    error = np.sum((approx - exact) ** 2) / binom(n, order)
    return error


def get_precision_at_k(approx: np.ndarray, exact: np.ndarray, k: int = 10) -> float:
    """Computes the precision at the top k absolute values."""
    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    top_k_id_exact = set(exact_abs.argsort()[-k:])

    approx_abs = np.abs(approx)
    approx_abs = approx_abs.flatten()
    top_k_id_approx = set(approx_abs.argsort()[-k:])

    wrong_ids = len(top_k_id_approx - top_k_id_exact)
    correct_ids = k - wrong_ids
    return float(correct_ids / k)


def get_approximation_error_at_k(approx: np.ndarray, exact: np.ndarray, k: int = 10) -> float:
    """Computes the approximation error at the top k absolute values."""
    exact_flat = exact.flatten()
    approx_flat = approx.flatten()

    exact_abs = np.abs(exact)
    exact_abs = exact_abs.flatten()
    top_k_id_exact = set(exact_abs.argsort()[-k:])

    error = np.sum((approx_flat[[*top_k_id_exact]] - exact_flat[[*top_k_id_exact]]) ** 2)
    return float(error)


def get_kendals_tau(approx: np.ndarray, exact: np.ndarray) -> float:
    """Computes the kendal's tau between the approximated and exact values."""
    exact = exact.flatten()
    ranking_exact = exact.argsort()
    approx = approx.flatten()
    ranking_approx = approx.argsort()
    tau, _ = kendalltau(ranking_exact, ranking_approx)
    return tau


def get_gt_values_for_game(game, shapiq: SHAPIQEstimator, order: int) -> Dict[int, np.ndarray]:
    """Computes the ground truth values for a given game and order."""
    try:
        gt_values = game.exact_values(gamma_matrix=shapiq.weights, min_order=order, max_order=order)
    except AttributeError:
        gt_values = shapiq.compute_interactions_complete(game=game.set_call)
    return gt_values


