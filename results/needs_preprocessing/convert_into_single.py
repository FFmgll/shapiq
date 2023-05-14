import copy
import os

import pandas as pd


def _convert_into_single(folder_name, run_folder_name):
    # get all filenames in a given folder
    filenames = os.listdir(run_folder_name)

    # read all json files and append them in a list
    dfs = []
    for filename in filenames:
        df = pd.read_json(os.path.join(run_folder_name, filename))
        dfs.append(df)

    result_dict = {}

    first_df = dfs[0]
    first_file_name = filenames[0]

    for col in first_df.columns:

        result_dict[col] = {}

        for index in first_df.index:

            result_dict[col][index] = None

            budget_results = []
            n_budgets = len(first_df[col][index])

            for budget_index in range(n_budgets):
                budget = first_df[col][index][budget_index]['budget']
                approx_errors = []
                for df in dfs:
                    approx_errors.append(df[col][index][budget_index]['mean'])
                approx_errors_df = pd.DataFrame(approx_errors)
                mean = dict(approx_errors_df.mean())
                median = dict(approx_errors_df.median())
                q_1 = dict(approx_errors_df.quantile(0.25))
                q_3 = dict(approx_errors_df.quantile(0.75))
                std = dict(approx_errors_df.std())
                var = dict(approx_errors_df.var())

                budget_results.append({
                    'budget': budget,
                    'mean': mean,
                    'median': median,
                    'q_1': q_1,
                    'q_3': q_3,
                    'std': std,
                    'var': var
                })

            result_dict[col][index] = copy.deepcopy(budget_results)

    save_file_name = first_file_name.split('_')[:-1]
    save_file_name[1] = f"runs-{len(dfs)}"
    save_file_name = '_'.join(save_file_name) + '.json'

    save_path = os.path.join(folder_name, save_file_name)

    # save_to_json
    pd.DataFrame(result_dict).to_json(save_path)


if __name__ == "__main__":

    interaction_indexes = ['SII']
    datasets_names = ["image_classifier_test"]
    runs = ["4_allOrder"] # ["3_allOrder", "4_allOrder", "3_topOrder", "4_topOrder"]

    # combinaiton of all three
    for dataset_name in datasets_names:
        for interaction_index in interaction_indexes:
            for run in runs:

                folder_name = os.path.join(dataset_name, interaction_index)
                run_folder_name = os.path.join(folder_name, run)

                try:
                    _convert_into_single(folder_name, run_folder_name)
                except FileNotFoundError:
                    pass







