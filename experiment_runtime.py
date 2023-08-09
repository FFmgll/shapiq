"""This module contains the code for the runtime experiments for the paper."""
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from approximators import SHAPIQEstimator, PermutationSampling, RegressionEstimator
from games import NLPGame


if __name__ == "__main__":

    k_values = [100, 500, 1000, 5000, 10_000]
    n_iterations = 5

    interaction_order = 2

    og = "I have never forgot this movie. All these years and it has remained in my life."
    sentence = "i have never forgot this movie all these years and it has remained in my life"

    game = NLPGame(input_text=sentence)
    n = game.n
    N = set(range(n))

    input_sentence_tokens = game.input_sentence
    input_words = []
    for token in game.tokenized_input:
        word = game.tokenizer.decode(token)
        input_words.append(word)

    print("Original Output:", game.original_output, "n:", game.n)
    print("'words':", input_words)

    x = np.arange(n)

    run_time_data = []

    for budget in k_values:
        for i in range(n_iterations):
            print("\nIteration: ", i, budget)

            run_times = {"k": budget}

            # compute time for SII -------------------------------------------------------------------------
            # SHAP-IQ
            shap_iq_sii = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="SII", top_order=True)
            start_time = time.time()
            sii_estimates = shap_iq_sii.compute_interactions_from_budget(budget=budget, game=game.set_call, pairing=False, show_pbar=False, only_expicit=False)
            elapsed_time = time.time() - start_time
            print("Elapsed time SII (SHAP-IQ): ", elapsed_time)
            run_times["SII_SHAP-IQ"] = elapsed_time
            # baseline (permutation sampling)
            baseline_estimator_sii = PermutationSampling(N=N, order=interaction_order, top_order=True, interaction_type="SII")
            start_time = time.time()
            _ = baseline_estimator_sii.approximate_with_budget(budget=budget, game=game.set_call)
            elapsed_time = time.time() - start_time
            print("Elapsed time SII (baseline): ", elapsed_time)
            run_times["SII_baseline"] = elapsed_time

            # compute time for STI -------------------------------------------------------------------------
            # SHAP-IQ
            shap_iq_sti = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="STI", top_order=True)
            start_time = time.time()
            _ = shap_iq_sti.compute_interactions_from_budget(budget=budget, game=game.set_call, pairing=False, show_pbar=False, only_expicit=False)
            elapsed_time = time.time() - start_time
            print("Elapsed time STI (SHAP-IQ): ", elapsed_time)
            run_times["STI_SHAP-IQ"] = elapsed_time
            # baseline (permutation sampling)
            baseline_estimator_sti = PermutationSampling(N=N, order=interaction_order, top_order=True, interaction_type="STI")
            start_time = time.time()
            _ = baseline_estimator_sti.approximate_with_budget(budget=budget, game=game.set_call)
            elapsed_time = time.time() - start_time
            print("Elapsed time STI (baseline): ", elapsed_time)
            run_times["STI_baseline"] = elapsed_time

            # compute time for FSI -------------------------------------------------------------------------
            # SHAP-IQ
            shap_iq_fsi = SHAPIQEstimator(N=N, order=interaction_order, interaction_type="FSI", top_order=True)
            start_time = time.time()
            _ = shap_iq_fsi.compute_interactions_from_budget(budget=budget, game=game.set_call, pairing=False, show_pbar=False, only_expicit=False)
            elapsed_time = time.time() - start_time
            print("Elapsed time FSI (SHAP-IQ): ", elapsed_time)
            run_times["FSI_SHAP-IQ"] = elapsed_time
            # baseline (regression)
            baseline_estimator_fsi = RegressionEstimator(N=N, max_order=interaction_order)
            start_time = time.time()
            _ = baseline_estimator_fsi.approximate_with_budget(budget=budget, game_fun=game.set_call, pairing=False)
            elapsed_time = time.time() - start_time
            print("Elapsed time FSI (baseline): ", elapsed_time)
            run_times["FSI_baseline"] = elapsed_time

            run_time_data.append(run_times)

    print("\n\n\n")
    print("Run time data:")
    run_time_data_df = pd.DataFrame(run_time_data)

    run_time_proportion_sii = run_time_data_df["SII_SHAP-IQ"] / run_time_data_df["SII_baseline"]
    print("SII proportion: ", run_time_proportion_sii.mean(), "std: ", run_time_proportion_sii.std())
    run_time_proportion_sti = run_time_data_df["STI_SHAP-IQ"] / run_time_data_df["STI_baseline"]
    print("STI proportion: ", run_time_proportion_sti.mean(), "std: ", run_time_proportion_sti.std())
    run_time_proportion_fsi = run_time_data_df["FSI_SHAP-IQ"] / run_time_data_df["FSI_baseline"]
    print("FSI proportion: ", run_time_proportion_fsi.mean(), "std: ", run_time_proportion_fsi.std())

    run_time_proportion_sii = run_time_data_df["SII_baseline"] / run_time_data_df["SII_SHAP-IQ"]
    print("SII proportion: ", run_time_proportion_sii.mean(), "std: ", run_time_proportion_sii.std())
    run_time_proportion_sti = run_time_data_df["STI_baseline"] / run_time_data_df["STI_SHAP-IQ"]
    print("STI proportion: ", run_time_proportion_sti.mean(), "std: ", run_time_proportion_sti.std())
    run_time_proportion_fsi = run_time_data_df["FSI_baseline"] / run_time_data_df["FSI_SHAP-IQ"]
    print("FSI proportion: ", run_time_proportion_fsi.mean(), "std: ", run_time_proportion_fsi.std())


    # agregate the data to get the mean and standard deviation over the iterations
    run_time_data_grouped_df = run_time_data_df.groupby("k").agg(["mean", "std"])
    run_time_data_grouped_df.columns = ["_".join(col) for col in run_time_data_grouped_df.columns.values]
    run_time_data_grouped_df = run_time_data_grouped_df.reset_index()

    LINE_MARKERS_DICT_INDEX = {'SII': "o", 'STI': "s", 'FSI': "X"}
    COLORS = {'SHAP-IQ': '#ef27a6', "Baseline": '#7d53de'}
    LINESTYLE_DICT_INDEX = {'SII': 'solid', 'STI': 'dashed', 'FSI': 'dashdot'}

    # plot the runtime curves for each baseline and SHAP-IQ over the budget (k_values) and standard deviation over the iterations
    fig, ax = plt.subplots()
    # SII
    plt.plot(k_values, run_time_data_grouped_df["SII_SHAP-IQ_mean"], label="SII SHAP-IQ", marker=LINE_MARKERS_DICT_INDEX["SII"], linestyle=LINESTYLE_DICT_INDEX["SII"], color=COLORS["SHAP-IQ"])
    plt.fill_between(k_values, run_time_data_grouped_df["SII_SHAP-IQ_mean"] - run_time_data_grouped_df["SII_SHAP-IQ_std"], run_time_data_grouped_df["SII_SHAP-IQ_mean"] + run_time_data_grouped_df["SII_SHAP-IQ_std"], alpha=0.2, color=COLORS["SHAP-IQ"])
    plt.plot(k_values, run_time_data_grouped_df["SII_baseline_mean"], label="SII Baseline", marker=LINE_MARKERS_DICT_INDEX["SII"], linestyle=LINESTYLE_DICT_INDEX["SII"], color=COLORS["Baseline"])
    plt.fill_between(k_values, run_time_data_grouped_df["SII_baseline_mean"] - run_time_data_grouped_df["SII_baseline_std"], run_time_data_grouped_df["SII_baseline_mean"] + run_time_data_grouped_df["SII_baseline_std"], alpha=0.2, color=COLORS["Baseline"])
    # STI
    plt.plot(k_values, run_time_data_grouped_df["STI_SHAP-IQ_mean"], label="STI SHAP-IQ", marker=LINE_MARKERS_DICT_INDEX["STI"], linestyle=LINESTYLE_DICT_INDEX["STI"], color=COLORS["SHAP-IQ"])
    plt.fill_between(k_values, run_time_data_grouped_df["STI_SHAP-IQ_mean"] - run_time_data_grouped_df["STI_SHAP-IQ_std"], run_time_data_grouped_df["STI_SHAP-IQ_mean"] + run_time_data_grouped_df["STI_SHAP-IQ_std"], alpha=0.2, color=COLORS["SHAP-IQ"])
    plt.plot(k_values, run_time_data_grouped_df["STI_baseline_mean"], label="STI Baseline", marker=LINE_MARKERS_DICT_INDEX["STI"], linestyle=LINESTYLE_DICT_INDEX["STI"], color=COLORS["Baseline"])
    plt.fill_between(k_values, run_time_data_grouped_df["STI_baseline_mean"] - run_time_data_grouped_df["STI_baseline_std"], run_time_data_grouped_df["STI_baseline_mean"] + run_time_data_grouped_df["STI_baseline_std"], alpha=0.2, color=COLORS["Baseline"])
    # FSI
    plt.plot(k_values, run_time_data_grouped_df["FSI_SHAP-IQ_mean"], label="FSI SHAP-IQ", marker=LINE_MARKERS_DICT_INDEX["FSI"], linestyle=LINESTYLE_DICT_INDEX["FSI"], color=COLORS["SHAP-IQ"])
    plt.fill_between(k_values, run_time_data_grouped_df["FSI_SHAP-IQ_mean"] - run_time_data_grouped_df["FSI_SHAP-IQ_std"], run_time_data_grouped_df["FSI_SHAP-IQ_mean"] + run_time_data_grouped_df["FSI_SHAP-IQ_std"], alpha=0.2, color=COLORS["SHAP-IQ"])
    plt.plot(k_values, run_time_data_grouped_df["FSI_baseline_mean"], label="FSI Baseline", marker=LINE_MARKERS_DICT_INDEX["FSI"], linestyle=LINESTYLE_DICT_INDEX["FSI"], color=COLORS["Baseline"])
    plt.fill_between(k_values, run_time_data_grouped_df["FSI_baseline_mean"] - run_time_data_grouped_df["FSI_baseline_std"], run_time_data_grouped_df["FSI_baseline_mean"] + run_time_data_grouped_df["FSI_baseline_std"], alpha=0.2, color=COLORS["Baseline"])

    plt.xlabel(r"budget ($K$)")
    plt.ylabel("run-time (in seconds)")
    plt.legend(title="$\\bf{Method}$")
    plt.title("Run-time")
    plt.savefig("runtime.pdf", bbox_inches='tight')
    plt.show()
