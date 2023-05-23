import numpy as np

from approximators import SHAPIQEstimator
from games import NLPGame, LookUpGame, NLPLookupGame

if __name__ == "__main__":

    interaction_order = 4
    max_budget = int(2 ** 14)

    # CONSTANTS ------------------------------------------------------------------------------------
    data_id = 281
    data_n = 14

    data_set_name = "nlp_values"
    game_list = []
    game = LookUpGame(data_folder=data_set_name, n=data_n, data_id=data_id, set_zero=True)

    # get information about the input senctence and the model --------------------------------------
    input_sentence = NLPLookupGame(n=data_n, sentence_id=data_id, set_zero=True).input_sentence
    nlp_model_game = NLPGame(input_text=input_sentence)
    original_ouput = nlp_model_game.original_output
    input_sentence_tokens = nlp_model_game.input_sentence
    input_words = []
    for token in nlp_model_game.tokenized_input:
        word = nlp_model_game.tokenizer.decode(token)
        input_words.append(word)
    print(input_words)

    print("Input sentence: ", input_sentence, "original output: ", original_ouput)

    # get number of players ------------------------------------------------------------------------
    n = game.n
    N = set(range(n))

    # get estiamtes  -------------------------------------------------------------------------------
    shap_iq = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=False)

    budget_steps = np.linspace(100, max_budget, 10)

    full_minus_empty = game.set_call(N) - game.set_call(set())

    print("Difference sum: ", full_minus_empty)
    print("\nrunning sampling")

    for budget_step in budget_steps:
        sii_values = shap_iq.compute_interactions_from_budget(game=game.set_call, budget=int(budget_step), show_pbar=False)

        n_shapley_values = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_values, n=interaction_order)

        sum_n_shapley_values = 0
        for key in n_shapley_values.keys():
            sum_n_shapley_values += np.sum(n_shapley_values[key])

        print("Budget: ", round(budget_step / 2**n, 4), "sum n-shapley: ",
              round(sum_n_shapley_values, 4), "full_minus_empty: ", round(full_minus_empty, 4),
              "difference: ", round(full_minus_empty - sum_n_shapley_values, 4))

    print("\nrunning only explicit")

    for budget_step in budget_steps:
        sii_values = shap_iq.compute_interactions_from_budget(game=game.set_call, budget=int(budget_step), show_pbar=False, only_expicit=True)

        n_shapley_values = shap_iq.transform_interactions_in_n_shapley(interaction_values=sii_values, n=interaction_order)

        sum_n_shapley_values = 0
        for key in n_shapley_values.keys():
            sum_n_shapley_values += np.sum(n_shapley_values[key])

        print("Budget: ", round(budget_step / 2 ** n, 4), "sum n-shapley: ",
              round(sum_n_shapley_values, 4), "full_minus_empty: ", round(full_minus_empty, 4),
              "difference: ", round(full_minus_empty - sum_n_shapley_values, 4))



