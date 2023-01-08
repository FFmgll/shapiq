import copy

from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from games import NLPGame, SyntheticNeuralNetwork, SimpleGame, SparseLinearModel, customSparseLinearModel

from shapx.interaction import ShapleyInteractionsEstimator
from shapx.permutation import PermutationSampling

from scipy.special import binom

def get_approximation_error(approx: np.ndarray, exact: np.ndarray, eps: float = 0.00001) -> float:
    error = np.sum((approx - exact) ** 2)
    #error = 0. if error < eps else error  # For pretty printing ...
    return error

if __name__ == "__main__":
        # Game Function ----------------------------------------------------------------------------------------------------
        #game = NLPGame(input_text="I like the movie no more")
        #n_features=10
        #game = customSparseLinearModel(n_features,weighting_scheme="uniform",n_interactions=0.1*2**n_features)
        #game = SparseLinearModel(n=10, n_interactions_per_order={1: 6, 2:0, 3:6, 4:23, 5:40,6:50,7:20}, n_non_important_features=0)
        game = SyntheticNeuralNetwork(n=12)
        #game = SimpleGame(n=10)

        # Game Parameters --------------------------------------------------------------------------------------------------
        n = game.n
        N = set(range(n))
        total_subsets = 2 ** n

        # Parameters -------------------------------------------------------------------------------------------------------
        min_order = 2
        shapley_interaction_order = 2

        min_budget = 0
        for k in range(shapley_interaction_order):
            min_budget += 2*binom(n,k)

        #subset tests
        subset_size = [4,5,6]
        subset_weights = np.zeros(n + 1)
        subset_weights[subset_size] = [0.3,0.4,0.3]

        subset_size = np.arange(0,12)
        subset_weights = np.ones(n+1)/n
        max_budget = 0
        for k in subset_size:
            max_budget += binom(n,k)

        max_budget = 20000
        budgets = np.arange(0,1.05,0.05)
        #budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        budgets = [int(budget * max_budget) for budget in budgets]
        budgets = np.array(budgets)
        budgets = budgets[budgets>=min_budget]
        all_budgets = sum(budgets)

        shapx_sampling = {}
        approximation_errors = {}
        approximation_stds = {}
        sampling_kernels = ["unif-size"]#,"unif-set","ksh","faith"]
        pairwise_list = [False]#[True, False]

        # All interactions
        shapley_extractor_sii = ShapleyInteractionsEstimator(
            N, shapley_interaction_order, min_order=min_order, interaction_type="SII")
        shapley_extractor_sti = ShapleyInteractionsEstimator(
            N, shapley_interaction_order, min_order=min_order, interaction_type="STI")
        shapley_extractor_sfi = ShapleyInteractionsEstimator(
            N, shapley_interaction_order, min_order=min_order, interaction_type="SFI")

        game_fun = game.set_call

        shapx_exact_k = {}
        shapx_list = [shapley_extractor_sii, shapley_extractor_sti, shapley_extractor_sfi]


        # Compute exact interactions ---------------------------------------------------------------------------------------
        print("Starting exact computations")
        for shapx in shapx_list:
            shapx_exact_k[shapx.interaction_type] = np.zeros(np.repeat(shapx.n,shapley_interaction_order))
            for k in subset_size:
                shapx_exact_k[shapx.interaction_type] += shapx._compute_interactions_complete_k(game_fun,k)[shapx.s]
        print("Exact computations finished")

        ITERATIONS = 1

        approx_errors_list = []

        for iteration in range(1, ITERATIONS + 1):
            print(f"Starting Iteration {iteration}")
            # Approximate ------------------------------------------------------------------------------------------------------
            for shapx in shapx_list:
                interaction_type = shapx.interaction_type
                pbar_budget = all_budgets * len(pairwise_list) * len(sampling_kernels)
                pbar = tqdm(total=pbar_budget, desc=interaction_type)
                exact_values = shapx_exact_k[interaction_type]
                # Sampling Approximations
                for sampling_kernel in sampling_kernels:
                    run_id = '_'.join((interaction_type, 'approximation', sampling_kernel))
                    result_sample_mean = shapx.init_results()
                    result_sample_s2 = shapx.init_results()
                    n_samples = 0
                    for pairwise in pairwise_list:
                        pairwise_id = 'pairwise' if pairwise else 'not-paired'
                        approx_run_id = '_'.join((run_id, pairwise_id))

                        for i in range(max(budgets)):
                            k = random.choices(subset_size, k=1, weights=subset_weights[subset_size])[0]
                            T = set(np.random.choice(shapx.n, k, replace=False))
                            # result_sample = shapx.update_results(result_sample, shapx._evaluate_subset(game_fun, T,budget*subset_weights[k]/binom(shapx.n,k)))
                            result_sample_update = shapx._evaluate_subset(game_fun, T,subset_weights[k]/binom(shapx.n,k))
                            result_sample_mean, result_sample_s2, n_samples = shapx.update_mean_variance(
                                result_sample_mean,
                                result_sample_s2, n_samples,
                                result_sample_update)
                            if n_samples in budgets:
                                relative_budget = round(n_samples / max_budget,2)
                                final_id = '_'.join((approx_run_id, str(n_samples), str(relative_budget)))
                                result_sample_std = shapx.apply_sqrt(shapx.scale_results(result_sample_s2, 1 / (n_samples)))
                                approximated_interactions = copy.deepcopy(result_sample_mean)
                                shapx_sampling['_'.join((final_id, 'not full'))] = approximated_interactions
                                approximation_errors['_'.join((final_id, 'not full'))] = get_approximation_error(
                                    approx=approximated_interactions[shapley_interaction_order], exact=exact_values)
                                approximation_stds['_'.join((final_id, 'std'))] = copy.deepcopy(result_sample_std)
                                pbar.update(n_samples)
                approximation_errors['_'.join((interaction_type, 'zero_estimate'))] = get_approximation_error(approx=np.zeros(np.repeat(n,shapx.s)), exact=exact_values)
                pbar.close()

            for approximator_id, approximation_error in approximation_errors.items():
                run_dict = {}
                id_parts = approximator_id.split('_')
                try:
                    run_dict['shapley_interaction_order'] = shapley_interaction_order
                    run_dict['iteration'] = iteration
                    run_dict['approx_value'] = approximation_error
                    run_dict['interaction_index'] = id_parts[0]
                    run_dict['n_absolute'] = id_parts[4]
                    run_dict['n_relative'] = id_parts[5]
                    run_dict['approx_type'] = id_parts[1]
                    run_dict['sampling_kernel'] = id_parts[2]
                    run_dict['pairing'] = id_parts[3]
                    run_dict['full'] = id_parts[6]
                except IndexError:
                    pass
                approx_errors_list.append(run_dict)

        approx_errors_df = pd.DataFrame(approx_errors_list)
        approx_errors_df.to_csv("interaction_sampling_k.csv", index=False)


