import os

import pandas as pd
from matplotlib import pyplot as plt

COLORS = {'SHAP-IQ': '#ef27a6', "Baseline": '#7d53de'}
LINESTYLE_DICT = {'SII': 'solid', 'STI': 'dashed', 'FSI': 'dashdot'}


def plot_errors_for_shapiq_and_baseline(
        x_data,
        shapiq_error,
        baseline_error,
        interaction_index: str,
        shapiq_std=None,
        baseline_std=None,
        linestyle="solid"
):
    fig, ax = plt.subplots()
    ax.plot(x_data, shapiq_error, color=COLORS["SHAP-IQ"], label="SHAP-IQ", linestyle=linestyle)
    ax.plot(x_data, baseline_error, color=COLORS["Baseline"], label="Baseline", linestyle=linestyle)
    if shapiq_std is not None:
        ax.fill_between(x_data, shapiq_error - shapiq_std, shapiq_error + shapiq_std, color=COLORS["SHAP-IQ"], alpha=0.2)
    if baseline_std is not None:
        ax.fill_between(x_data, baseline_error - baseline_std, baseline_error + baseline_std, color=COLORS["Baseline"], alpha=0.2)
    ax.set_title(f"Interaction index: {interaction_index}")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Error")
    ax.legend()
    plt.show()


if __name__ == "__main__":

    # experiment parameters ------------------------------------------------------------------------
    game_name = "nlp_values_14"
    interaction_index = 'SII'
    TOP_ORDER = True
    ORDER = 2
    NUMBER_OF_RUNS = 3

    file_name = f"n-14_runs-{NUMBER_OF_RUNS}_s0-{ORDER}_top-order-{TOP_ORDER}_pairing-True_stratification-False_weights-ksh.json"

    RESULTS_DIR = "results"
    folder_path = os.path.join("..", RESULTS_DIR, game_name, interaction_index)
    file_path = os.path.join(folder_path, file_name)

    # plot parameters ------------------------------------------------------------------------------
    error_to_plot_id = 'approximation_error'
    orders_to_plot = [2]  # or None for all orders

    # get parameters from file name
    file_params = file_name.split("_")
    N_PLAYER = int(file_params[0].split("-")[1])
    NUMBER_OF_RUNS = int(file_params[1].split("-")[1])
    ORDER = int(file_params[2].split("-")[1])
    RUN_TOP_ORDER = bool(file_params[3].split("-")[2])
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

    for order in orders_to_plot:
        shapiq_dict_order_df = pd.DataFrame(shapiq_dict[order])
        baseline_dict_order_df = pd.DataFrame(baseline_dict[order])
        x_data = shapiq_dict_order_df["budget"].values

        # get mean_values of error_to_plot_id
        shapiq_mean_values = shapiq_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values
        baseline_mean_values = baseline_dict_order_df['mean'].apply(lambda x: x[error_to_plot_id]).values

        shapiq_std_values = shapiq_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values
        baseline_std_values = baseline_dict_order_df['std'].apply(lambda x: x[error_to_plot_id]).values

        # plot the errors
        plot_errors_for_shapiq_and_baseline(
            x_data=x_data,
            shapiq_error=shapiq_mean_values,
            baseline_error=baseline_mean_values,
            interaction_index=interaction_index,
            shapiq_std=shapiq_std_values,
            baseline_std=baseline_std_values,
            linestyle=LINESTYLE_DICT[interaction_index]
        )
