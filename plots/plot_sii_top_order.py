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
GAME_NAME_DICT = {"nlp_values_14": "LM", "image_classifier_14": "ResNet", "bike_1_12": r"$bike$", "adult_1_14": r"$adult$", "SOUM_70": r"SOUM"}


if __name__ == "__main__":

    SAVE_FIG = False

    # experiment parameters ------------------------------------------------------------------------
    game_name = "SOUM_30"
    interaction_index = 'FSI'
    TOP_ORDER = True
    ORDER = 2
    NUMBER_OF_RUNS = 1
    N_PLAYER = int(game_name.split("_")[-1])

    file_name = f"n-{N_PLAYER}_runs-{NUMBER_OF_RUNS}_s0-{ORDER}_top-order-{TOP_ORDER}_pairing-True_stratification-False_weights-ksh.json"

    RESULTS_DIR = "results"
    folder_path = os.path.join("..", RESULTS_DIR, game_name, interaction_index)
    file_path = os.path.join(folder_path, file_name)

    # plot parameters ------------------------------------------------------------------------------
    error_to_plot_id = "approximation_error" #"approximation_error" #'precision_at_10' #'precision_at_10' # 'approximation_error' # or 'kendals_tau'

    orders_to_plot = [ORDER]  # or None for all orders
    if TOP_ORDER and orders_to_plot[0] == 0:
        orders_to_plot = None

    # get parameters from file name
    file_params = file_name.split("_")
    PAIRING = bool(file_params[4].split("-")[1])
    STRATIFICATION = bool(file_params[5].split("-")[1])
    SAMPLING_KERNEL = file_params[6].split("-")[1].split(".")[0]

    plot_title = f"Game: {game_name}, N: {N_PLAYER}, Order: {ORDER}, Any Orders"

    # read json file with pandas
    df = pd.read_json(file_path)
    shapiq_dict = dict(df["shapiq"])
    baseline_dict = dict(df["baseline"])

    orders_in_file = list(shapiq_dict.keys())
    orders_to_plot = orders_in_file if orders_to_plot is None else orders_to_plot

    x_data = None

    plot_mean = True

    plot_iqr = False
    plot_std = False
    y_manual = 5 # 10

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

    for order in orders_to_plot:

        # get dataframes
        shapiq_dict_order_df = pd.DataFrame(shapiq_dict[order])
        baseline_dict_order_df = pd.DataFrame(baseline_dict[order])

        # get x data
        x_data = shapiq_dict_order_df["budget"].values
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
        ax.plot(x_data, shapiq_error_values, color=COLORS["SHAP-IQ"], linestyle=LINESTYLE_DICT_ORDER[order], marker=LINE_MARKERS_DICT_ORDER[order], mec="white")
        ax.plot(x_data, baseline_error_values, color=COLORS["Baseline"], linestyle=LINESTYLE_DICT_ORDER[order], marker=LINE_MARKERS_DICT_ORDER[order], mec="white")
        y_max_value = max(y_max_value, max(shapiq_error_values))#, max(baseline_error_values))

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

    ax.set_title(f"Interaction index: {interaction_index}")

    # plot legend
    ax.plot([], [], label="$\\bf{Method}$", color="none")
    ax.plot([], [], color=COLORS["SHAP-IQ"], linestyle="solid", label="SHAP-IQ")
    ax.plot([], [], color=COLORS["Baseline"], linestyle="solid", label="Baseline")

    n_col_legend = 2
    order_title = r"$s_0$={"
    if len(orders_to_plot) > 2:
        for _ in range(len(orders_to_plot) - 2):
            ax.plot([], [], color="none", label=" ")
    ax.plot([], [], label="$\\bf{Order}$", color="none")
    for order in orders_to_plot:
        if order > 0:
            order_title += f"{order}, " if order < max(orders_to_plot) else f"{order}"
        label_text = r"$s_0$" + f" = {order}" if order > 0 else r"all to $s_0$" + f" = {max(orders_to_plot)}"
        ax.plot([], [], color="black", linestyle=LINESTYLE_DICT_ORDER[order], label=label_text, marker=LINE_MARKERS_DICT_ORDER[order], mec="white")
    order_title += "}"
    ax.legend(ncols=n_col_legend)

    if len(orders_to_plot) == 1:
        order_title = r"$s_0 =$" + f"{max(orders_to_plot)}"

    # set y axis limits
    ax.set_ylim((0, y_max_value * 1.1))
    if error_to_plot_id == "kendals_tau":
        ax.set_ylim((0, 1))
    if y_manual is not None:
        ax.set_ylim((0, y_manual))

    # set x axis limits to 10% of max value
    ax.set_xlim((max(x_data) * 0.1, None))

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
    #
    try:
        game_name = GAME_NAME_DICT[game_name]
    except KeyError:
        game_name = game_name
    title = f"{interaction_index}, {game_name} (" \
            + order_title + ", "\
            + fr"$d = {N_PLAYER}$" + ", " \
            + fr"$g = {NUMBER_OF_RUNS}$" \
            + ")"
    ax.set_title(title, fontsize="xx-large")

    plt.tight_layout()

    # save figure as pdf
    if SAVE_FIG:
        save_name = f"{interaction_index}_{game_name}_top-order-{TOP_ORDER}-{max(orders_to_plot)}_{error_to_plot_id}" + ".pdf"
        plt.savefig(save_name)

    plt.show()
