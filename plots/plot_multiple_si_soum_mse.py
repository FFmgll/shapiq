import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

COLORS = {'SHAP-IQ': '#ef27a6', "Baseline": '#7d53de'}
LINESTYLE_DICT_INDEX = {'SII': 'solid', 'STI': 'dashed', 'FSI': 'dashdot'}
LINESTYLE_DICT_ORDER = {0: "solid", 1: "dotted", 2: 'solid', 3: 'dashed', 4: 'dashdot'}
ERROR_NAME_DICT = {"approximation_error": "MSE", "kendals_tau": "Kendall's $\\tau$", "precision_at_10": "Pr@10"}
LINE_MARKERS_DICT_ORDER = {0: "o", 1: "o", 2: "s", 3: "X", 4: "d"}
LINE_MARKERS_DICT_INDEX = {'SII': "o", 'STI': "s", 'FSI': "X"}
GAME_NAME_DICT = {"nlp_values_14": "LM", "image_classifier_14": "ResNet", "bike_1_12": r"$bike$", "adult_1_14": r"$adult$", "SOUM_30": "SOUM"}


if __name__ == "__main__":

    SAVE_FIG = True

    # experiment parameters ------------------------------------------------------------------------
    game_name = "SOUM_30"
    N_PLAYER = int(game_name.split("_")[-1])
    interaction_indices = ['FSI', 'SII', 'STI']
    TOP_ORDER = True
    ORDER = 2
    NUMBER_OF_RUNS = 50

    # plot parameters ------------------------------------------------------------------------------
    error_to_plot_id = "approximation_error"  # "approximation_error" #'precision_at_10' #'precision_at_10' # 'approximation_error' # or 'kendals_tau'
    x_data = None
    plot_mean = True
    plot_iqr = False
    plot_std = True
    y_manual = 0.5 # 0.1  # 1 0.5
    x_min_to_plot = 1000
    x_max = None


    shapiq_dict_dfs = {}
    baseline_dict_dfs = {}
    for interaction_index in interaction_indices:

        file_name = f"n-{N_PLAYER}_runs-{NUMBER_OF_RUNS}_s0-{ORDER}_top-order-{TOP_ORDER}_pairing-False_stratification-False_weights-ksh.json"
        file_path = os.path.join("..", "results", game_name, interaction_index, file_name)

        # read json file with pandas
        df = pd.read_json(file_path)
        shapiq_dict = dict(df["shapiq"])[ORDER]
        baseline_dict = dict(df["baseline"])[ORDER]

        shapiq_dict_dfs[interaction_index] = {"SHAP-IQ": shapiq_dict, "Baseline": baseline_dict}


    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (7, 7),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    # get plot canvas
    fig, ax = plt.subplots()

    y_max_value = 0
    for interaction_index in interaction_indices:

        # get dataframes
        shapiq_dict_order_df = pd.DataFrame(shapiq_dict_dfs[interaction_index]['SHAP-IQ'])
        baseline_dict_order_df = pd.DataFrame(shapiq_dict_dfs[interaction_index]['Baseline'])

        # get x data
        x_data = shapiq_dict_order_df["budget"].values
        if x_min_to_plot is None:
            x_min_to_plot = 0.1 * max(x_data)
        # get first index of x_data that is greater than x_min_to_plot
        x_min_index = next(i for i, x in enumerate(x_data) if x > x_min_to_plot)
        x_data = x_data[x_min_index:]

        # get summary line
        if plot_mean:
            shapiq_error_values = shapiq_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            baseline_error_values = baseline_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
        else:
            shapiq_error_values = shapiq_dict_order_df['median'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            baseline_error_values = baseline_dict_order_df['median'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]

        # plot summary line (either mean or median)
        ax.plot(x_data, shapiq_error_values, color=COLORS["SHAP-IQ"], linestyle=LINESTYLE_DICT_INDEX[interaction_index], marker=LINE_MARKERS_DICT_INDEX[interaction_index], mec="white")
        ax.plot(x_data, baseline_error_values, color=COLORS["Baseline"], linestyle=LINESTYLE_DICT_INDEX[interaction_index], marker=LINE_MARKERS_DICT_INDEX[interaction_index], mec="white")
        y_max_value = max(y_max_value, max(shapiq_error_values), max(baseline_error_values))

        if plot_iqr:
            shapiq_q1_values = shapiq_dict_order_df['q_1'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            shapiq_q3_values = shapiq_dict_order_df['q_3'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, shapiq_q1_values, shapiq_q3_values, alpha=0.2, color=COLORS["SHAP-IQ"])

            baseline_q1_values = baseline_dict_order_df['q_1'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            baseline_q3_values = baseline_dict_order_df['q_3'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, baseline_q1_values, baseline_q3_values, alpha=0.2, color=COLORS["Baseline"])

        if plot_std:
            shapiq_std = shapiq_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, shapiq_error_values - shapiq_std, shapiq_error_values + shapiq_std, alpha=0.2, color=COLORS["SHAP-IQ"])
            baseline_std = baseline_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values[x_min_index:]
            ax.fill_between(x_data, baseline_error_values - baseline_std, baseline_error_values + baseline_std, alpha=0.2, color=COLORS["Baseline"])

    # plot legend
    ax.plot([], [], label="$\\bf{Method}$", color="none")
    ax.plot([], [], color=COLORS["SHAP-IQ"], linestyle="solid", label="SHAP-IQ")
    ax.plot([], [], color=COLORS["Baseline"], linestyle="solid", label="Baseline")

    n_col_legend = 2
    order_title = r"$s_0$={"
    if len(interaction_indices) > 2:
        for _ in range(len(interaction_indices) - 2):
            ax.plot([], [], color="none", label=" ")
    ax.plot([], [], label="$\\bf{Order}$", color="none")
    for interaction_index in interaction_indices:
        label_text = interaction_index
        ax.plot([], [], color="black", linestyle=LINESTYLE_DICT_INDEX[interaction_index], label=label_text, marker=LINE_MARKERS_DICT_INDEX[interaction_index], mec="white")

    ax.legend(ncols=n_col_legend)

    # set y axis limits
    ax.set_ylim((0, y_max_value * 1.1))
    if error_to_plot_id == "kendals_tau" or error_to_plot_id == "precision_at_10":
        ax.set_ylim((0, 1))
    if y_manual is not None:
        ax.set_ylim((0, y_manual))

    # set x axis limits to 10% of max value
    ax.set_xlim((x_min_to_plot, x_max))

    # set x_ticklabels and x_label
    x_ticklabels_abs = ax.get_xticks()
    if N_PLAYER <= 14:
        x_tick_relative = [x_ticklabel / 2**N_PLAYER for x_ticklabel in x_ticklabels_abs]
        x_ticklabels = [f"{abs_:.0f}\n{rel_:.2f}" for abs_, rel_ in zip(x_ticklabels_abs, x_tick_relative)]
        x_label = "model evaluations (absolute, relative)"
    else:
        x_ticklabels = [f"{abs_:.0f}" for abs_ in x_ticklabels_abs]
        x_label = "model evaluations"
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(x_label)

    # set y_label
    ax.set_ylabel(f"{ERROR_NAME_DICT[error_to_plot_id]}")

    # set title
    try:
        game_name = GAME_NAME_DICT[game_name]
    except KeyError:
        game_name = game_name
    title = f"SII, STI, and FSI for {game_name} (" \
            + fr"$s_0 = {ORDER}$" + ", " \
            + fr"$d = {N_PLAYER}$" \
            + ")"
    ax.set_title(title, fontsize="xx-large")

    plt.tight_layout()

    # save figure as pdf
    if SAVE_FIG:
        save_name = f"n-index-{len(interaction_indices)}_{game_name}_top-order-{TOP_ORDER}_{error_to_plot_id}" + ".pdf"
        plt.savefig(save_name)

    plt.show()
