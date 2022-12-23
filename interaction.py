import copy
from tqdm import tqdm

from games import NLPGame, SyntheticNeuralNetwork, SimpleGame, SparseLinearModel
from shapx import ShapleyInteractionsEstimator, get_approximation_error, PermutationSampling

if __name__ == "__main__":
    # Game Function ----------------------------------------------------------------------------------------------------
    game = NLPGame(input_text="I like the movie no more")
    #game = SparseLinearModel(n=9, n_interactions_per_order={4: 24}, n_non_important_features=0)
    #game = SyntheticNeuralNetwork(n=7)
    #game = SimpleGame(n=10)

    # Game Parameters --------------------------------------------------------------------------------------------------
    n = game.n
    N = set(range(n))
    total_subsets = 2 ** n

    # Parameters -------------------------------------------------------------------------------------------------------
    min_order = 5
    shapley_interaction_order = 5

    max_budget = min(total_subsets, 2 ** 13)
    budgets = [0.25, 0.5, 0.75, 1.0, 1.25]
    budgets = [int(budget * max_budget) for budget in budgets]
    all_budgets = sum(budgets)

    shapx_sampling = {}
    approximation_errors = {}
    sampling_kernels = ["unif-size"]
    pairwise_list = [True, False]

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
    shapx_list = [shapley_extractor_sii, shapley_extractor_sti, shapley_extractor_sfi]
    permutation_samplers = {"SII": shapley_extractor_sii_permutation, "STI": shapley_extractor_sti_permutation}

    # Compute exact interactions ---------------------------------------------------------------------------------------
    print("Starting exact computations")
    for shapx in shapx_list:
        if False:  # hasattr(game, "exact_values"):  # TODO when it's implemented correctly run this here
            shapx_exact[shapx.interaction_type] = copy.deepcopy(game.exact_values)
        else:
            shapx_exact[shapx.interaction_type] = shapx.compute_interactions_complete(game_fun)
    print("Exact computations finished")

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
            run_id = '_'.join((interaction_type, str(budget), str(relative_budget)))

            # Permutation Approximations
            if interaction_type in ("STI", "SII"):
                perm_run_id = '_'.join((run_id, 'permutation'))
                perm_sampler = permutation_samplers[interaction_type]

                approximated_interactions = copy.deepcopy(perm_sampler.permutation_approximation(game_fun, budget))
                shapx_sampling[perm_run_id] = approximated_interactions
                approximation_errors[perm_run_id] = get_approximation_error(
                    approx=shapx_sampling[perm_run_id], exact=exact_values)
                pbar.update(budget)

            # Sampling Approximations
            for sampling_kernel in sampling_kernels:
                run_id = '_'.join((run_id, 'approximation', sampling_kernel))
                for pairwise in pairwise_list:
                    pairwise_id = 'pairwise' if pairwise else 'not-paired'
                    approx_run_id = '_'.join((run_id, pairwise_id))

                    approximated_interactions = copy.deepcopy(shapx.compute_interactions_from_budget(
                        game_fun, budget,  pairing=pairwise, sampling_kernel=sampling_kernel))
                    shapx_sampling[approx_run_id] = approximated_interactions
                    approximation_errors[approx_run_id] = get_approximation_error(
                        approx=approximated_interactions[shapley_interaction_order], exact=exact_values)

                    approximated_interactions = copy.deepcopy(shapx.last_const_complete[shapley_interaction_order])
                    shapx_sampling['_'.join((approx_run_id, 'full'))] = approximated_interactions
                    approximation_errors['_'.join((approx_run_id, 'full'))] = get_approximation_error(
                        approx=shapx.last_const_complete[shapley_interaction_order], exact=exact_values)

                    pbar.update(budget)
        pbar.close()
