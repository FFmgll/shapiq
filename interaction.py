import copy

from tqdm import tqdm
import numpy as np
import pandas as pd

from games import NLPGame, SyntheticNeuralNetwork, SimpleGame, SparseLinearModel, ParameterizedSparseLinearModel

from shapx.interaction import ShapleyInteractionsEstimator
from shapx.permutation import PermutationSampling

from scipy.special import binom

def get_approximation_error(approx: np.ndarray, exact: np.ndarray, eps: float = 0.00001) -> float:
    error = np.sum((approx - exact) ** 2)
    error = 0. if error < eps else error  # For pretty printing ...
    return error


if __name__ == "__main__":
    # Game Function ----------------------------------------------------------------------------------------------------
    game = NLPGame(input_text="I like the movie no more")
    #game = SparseLinearModel(n=30, n_interactions_per_order={1: 6, 2: 6, 3:6, 4:20, 5:3,6:5,7:3}, n_non_important_features=0)
    #n_features = 10
    #game = ParameterizedSparseLinearModel(n_features, weighting_scheme="uniform", n_interactions=1000)
    #game = SyntheticNeuralNetwork(n=12)
    #game = SimpleGame(n=10)
    #N_FEATURES: int = 15
    #game = ParameterizedSparseLinearModel(
    #    n=N_FEATURES, weighting_scheme="uniform", n_interactions=30,max_interaction_size=5)

    game_name = game.game_name
    # Game Parameters --------------------------------------------------------------------------------------------------
    n = game.n
    N = set(range(n))
    total_subsets = 2 ** n

    # Parameters -------------------------------------------------------------------------------------------------------
    min_order = 2
    shapley_interaction_order = 2

    #If True, forces to sample all subset sizes
    sampling_only=False

    min_budget = 0
    for k in range(shapley_interaction_order):
        min_budget += 2*binom(n,k)

    budget_cap = 20000
    max_budget = min(total_subsets,budget_cap)
    #max_budget = 2000
    budgets = np.arange(0.05,1.01,0.05)
    #budgets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    budgets = [int(budget * max_budget) for budget in budgets]
    budgets = np.array(budgets)
    budgets = budgets[budgets>=min_budget]
    all_budgets = sum(budgets)

    shapx_sampling = {}
    approximation_errors = {}
    sampling_kernels = ["ksh"]
    pairwise_list = [False]#[True, False]

    # All interactions
    shapley_extractor_sii = ShapleyInteractionsEstimator(
        N, shapley_interaction_order, min_order=min_order, interaction_type="SII")
    shapley_extractor_sti = ShapleyInteractionsEstimator(
        N, shapley_interaction_order, min_order=min_order, interaction_type="STI")
    shapley_extractor_sfi = ShapleyInteractionsEstimator(
        N, shapley_interaction_order, min_order=min_order, interaction_type="SFI")

    # Permutation Estimator
    shapley_extractor_sii_permutation = PermutationSampling(
        N, shapley_interaction_order, min_order=min_order, interaction_type="SII")
    shapley_extractor_sti_permutation = PermutationSampling(
        N, shapley_interaction_order, min_order=min_order, interaction_type="STI")

    game_fun = game.set_call

    shapx_exact = {}
    shapx_list = [shapley_extractor_sii, shapley_extractor_sti,shapley_extractor_sfi]
    permutation_samplers = {"SII": shapley_extractor_sii_permutation, "STI": shapley_extractor_sti_permutation}

    # Compute exact interactions ---------------------------------------------------------------------------------------
    print("Starting exact computations")
    for shapx in shapx_list:
        if hasattr(game, "exact_values"):  # TODO when it's implemented correctly run this here
            #print("exact values used")
            shapx_exact[shapx.interaction_type] = copy.deepcopy(game.exact_values(shapx.weights,shapley_interaction_order,shapley_interaction_order))
        else:
            shapx_exact[shapx.interaction_type] = shapx.compute_interactions_complete(game_fun)
    print("Exact computations finished")

    ITERATIONS = 1

    approx_errors_list = []

    for iteration in range(1, ITERATIONS + 1):
        print(f"Starting Iteration {iteration}")

        # Approximate ------------------------------------------------------------------------------------------------------
        for shapx in shapx_list:
            interaction_type = shapx.interaction_type
            if interaction_type in ("STI", "SII"):
                pbar_budget = all_budgets * len(pairwise_list) * len(sampling_kernels) + all_budgets
            else:
                pbar_budget = all_budgets * len(pairwise_list) * len(sampling_kernels)
            pbar = tqdm(total=pbar_budget, desc=interaction_type)
            exact_values = shapx_exact[interaction_type][shapley_interaction_order]
            for budget in budgets:
                relative_budget = round(budget / total_subsets, 2)
                run_id1 = '_'.join((interaction_type, str(budget), str(relative_budget)))

                # Permutation Approximations
                if interaction_type in ("STI", "SII"):
                    perm_run_id = '_'.join((run_id1, 'permutation'))
                    perm_sampler = permutation_samplers[interaction_type]

                    approximated_interactions = copy.deepcopy(perm_sampler.approximate_with_budget(game_fun, budget))
                    shapx_sampling[perm_run_id] = approximated_interactions
                    approximation_errors[perm_run_id] = get_approximation_error(
                        approx=shapx_sampling[perm_run_id], exact=exact_values)/binom(shapx.n,shapx.s)
                    pbar.update(budget)

                # Sampling Approximations
                for sampling_kernel in sampling_kernels:
                    run_id2 = '_'.join((run_id1, 'approximation', sampling_kernel))
                    for pairwise in pairwise_list:
                        pairwise_id = 'pairwise' if pairwise else 'not-paired'
                        run_id3 = '_'.join((run_id2, pairwise_id))
                        approximated_interactions = copy.deepcopy(shapx.compute_interactions_from_budget(
                            game_fun, budget,  pairing=pairwise, sampling_kernel=sampling_kernel,sampling_only=sampling_only))
                        shapx_sampling['_'.join((run_id3, 'not full'))] = approximated_interactions
                        approximation_errors['_'.join((run_id3, 'not full'))] = get_approximation_error(
                            approx=approximated_interactions[shapley_interaction_order], exact=exact_values)/binom(shapx.n,shapx.s)

                        approximated_interactions = copy.deepcopy(shapx.last_const_complete[shapley_interaction_order])
                        shapx_sampling['_'.join((run_id3, 'full'))] = approximated_interactions
                        approximation_errors['_'.join((run_id3, 'full'))] = get_approximation_error(
                            approx=shapx.last_const_complete[shapley_interaction_order], exact=exact_values)/binom(shapx.n,shapx.s)
                        pbar.update(budget)
            pbar.close()

        for approximator_id, approximation_error in approximation_errors.items():
            run_dict = {}
            id_parts = approximator_id.split('_')
            try:
                run_dict['shapley_interaction_order'] = shapley_interaction_order
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

    approx_errors_df = pd.DataFrame(approx_errors_list)
    approx_errors_df.to_csv(game_name+"_"+str(n)+"_"+str(sampling_only)+".csv", index=False)


    from shapx.base import powerset

    def delta(game_fun,T,S):
        rslt = 0
        s = len(S)
        for L in powerset(S):
            rslt += (-1)**(s-len(L))*game_fun(tuple(T)+L)
        return rslt

    test = {}
    for k in range(3):
        test[k] = np.zeros(np.repeat(n, k))
    for S in powerset(N,1,1):
        test[1][S] = delta(game_fun,{},S)




    #test2 = np.sum(test[1])+ np.sum(shapx_sampling['STI_4000_0.12_approximation_ksh_not-paired_not full'][2])