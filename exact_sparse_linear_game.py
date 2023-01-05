import copy

from tqdm import tqdm
import numpy as np
from scipy.special import binom

from games import NLPGame, SyntheticNeuralNetwork, SimpleGame, SparseLinearModel

from shapx.interaction import ShapleyInteractionsEstimator
from shapx.permutation import PermutationSampling


if __name__ == "__main__":

    game = SparseLinearModel(n=10, n_interactions_per_order={2: 5})

    n = game.n
    N = set(range(n))
    total_subsets = 2 ** n

    # Parameters -----------------------------------------------------------------------------------
    min_order = 2
    shapley_interaction_order = 2
    approximation_errors = {}

    # All interactions
    shapley_extractor_sii = ShapleyInteractionsEstimator(
        N, shapley_interaction_order, min_order=min_order, interaction_type="SII")
    shapley_extractor_sti = ShapleyInteractionsEstimator(
        N, shapley_interaction_order, min_order=min_order, interaction_type="STI")
    shapley_extractor_sfi = ShapleyInteractionsEstimator(
        N, shapley_interaction_order, min_order=min_order, interaction_type="SFI")

    game_fun = game.set_call

    shapx_exact = {}
    shapx_list = [shapley_extractor_sii, shapley_extractor_sti, shapley_extractor_sfi]

    # Compute exact interactions -------------------------------------------------------------------
    print("Starting exact computations")
    for shapx in shapx_list:
        shapx_exact[shapx.interaction_type+"_computed"] = copy.deepcopy(
            game.exact_values(gamma_matrix=shapx.weights, s=shapley_interaction_order))
        shapx_exact[shapx.interaction_type+"_bruteforce"] = shapx.compute_interactions_complete(game_fun)
    print("Exact computations finished")

    """
    s= shapley_interaction_order
    shapx = shapx_list[0]
    test = shapx.compute_interactions_complete(game_fun)
    test2 = game.exact_values(gamma_matrix=shapx.weights,s=shapley_interaction_order)
    game.coefficient_weighting(shapx.weights,shapley_interaction_order,2,2)
    
    
    
    import itertools
    def powerset(iterable, min_size=-1, max_size=None):
        if max_size is None and min_size > -1:
            max_size = min_size
        s = list(iterable)
        if max_size is None:
            max_size = len(s)
        else:
            max_size = min(max_size, len(s))
        return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))
    
    
    for key,val in game.interaction_weights.items():
        shapx_weights = set(key)
        
    intx_weights = {0,5}
    
    rslt_test = 0
    for T in powerset(N,s,n):
        if len(set(T).intersection(shapx_weights)) == 2:
            if len(set(T).intersection(intx_weights))==2:
                rslt_test += shapx.weights[len(T),2]
            if len(set(T).intersection(intx_weights))==1:
                rslt_test += shapx.weights[len(T),1]
            if len(set(T).intersection(intx_weights))==0:
                rslt_test += shapx.weights[len(T),0]
    
    
    q = 2
    r = 0
    rslt_test2 = 0
    for t in range(q,n+1):
            #print(top)
            add = min(t-q,s-r)
            for l in range(add+1):
                #tmp = shapx.weights[t,l]*binom(n-q-l+r,t-q-l+r)
                tmp = shapx.weights[t,l+r]*binom(n-q-(s-r),t-q-l)*binom(s-r,l)
                print(t,l+r,tmp,add, binom(n-q-(s-r),t-q-l))
                rslt_test2 += tmp
    print(rslt_test2)
    
    """
