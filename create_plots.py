import pandas as pd

from evaluation import draw_approx_curve

if __name__ == "__main__":
    fig_size = (5, 5)

    if True:  # LM --------------------------------------------------------------------

        file_name = "results_old/finished/language_model_14_4.csv"
        plot_title = r"LM ($s_0 = 4$, $d = 14$, $g = 10$)"
        x_min = 2_000

        y_max = 0.2
        y_min = -0.01

        df = pd.read_csv(file_name)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="average error at 10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=1, y_min=0,
                          y_label="P@10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          save_name=save_name)

        file_name = "results_old/finished/language_model_14_3.csv"
        plot_title = r"LM ($s_0 = 3$, $d = 14$, $g = 50$)"
        x_min = 2_000
        y_max = 0.1
        y_min = -0.001

        df = pd.read_csv(file_name)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="average error at 10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=1., y_min=0.,
                          y_label="P@10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          save_name=save_name)

        file_name = "results_old/finished/language_model_14_2.csv"
        plot_title = r"LM ($s_0 = 2$, $d = 14$, $g = 50$)"
        x_min = 2_000
        df = pd.read_csv(file_name)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/finished/language_model_14_2.csv"
        this_title = r"LM ($s_0 = 2$, $d = 14$, $g = 50$)"
        x_min = 2_000
        df = pd.read_csv(file_name)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_detail_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=this_title,
                          y_max=0.005, y_min=-0.0001,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)


        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        df = df[df["iteration"] <= 50]
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=0.02, y_min=-0.001,
                          y_label="average error at 10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=1.02, y_min=-0.02,
                          y_label="P@10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          save_name=save_name)

        file_name = "results_old/finished/language_model_14_1.csv"
        plot_title = r"LM, SV ($s_0 = 1$, $d = 14$, $g = 7$)"
        x_min = 2_000
        df = pd.read_csv(file_name)
        df = df.drop(df[df["interaction_index"] == "SII"].index)
        df = df.drop(
            df[(df["interaction_index"] == "STI") & (df["approx_type"] == "approximation")].index)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=-0.00001, y_max=0.0003,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name,
                          shap=True)

    if True:  # Gradient Boosted Trees

        file_name = "results_old/finished/tabular_game_14_2.csv"
        plot_title = r"Adult GBT ($s_0 = 2$, $d = 14$, $g = 50$)"
        x_min = 2_000

        df = pd.read_csv(file_name)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=0.00004, y_min=-0.000001,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name)

        df = pd.read_csv(file_name).drop(columns=["approx_value"])
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          #y_max=1.01, y_min=-0.01,
                          y_label="P@10",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          save_name=save_name)


        file_name = "results_old/finished/tabular_game_14_1.csv"
        plot_title = r"Adult-GBT, SV ($s_0 = 1$, $d = 14$, $g = 50$)"
        x_min = 2_000
        df = pd.read_csv(file_name)
        df = df.drop(df[df["interaction_index"] == "SII"].index)
        df = df.drop(df[(df["interaction_index"] == "STI") & (df["approx_type"] == "approximation")].index)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=0.0000008, y_min=-0.00000001,
                          y_label="MSE",
                          x_label="model evaluations (absolute, relative)",
                          max_computation_cost_n=14,
                          horizontal_line_y=0.,
                          save_name=save_name, shap=True)

    if True:  # Sparse Linear Models

        # average error ----------------------------------------------------------------------------
        x_min = 2000
        y_min = -0.01
        y_max = 0.9

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_10.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 10, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_15.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 15, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_20.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 20, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_25.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 25, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_30.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        x_min = 2_000
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])  # for uniquness
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        # Error at 10 ------------------------------------------------------------------------------

        x_min = 2000
        y_min = -0.1
        y_max = 7

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_10.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 10, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_15.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 15, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_20.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 20, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_25.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 25, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_30.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)


        # P@10 --------------------------------------------------------------------------

        x_min = 2000
        y_min = 0
        y_max = 1

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_10.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 10, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_15.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 15, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_20.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 20, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_25.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 25, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_100_0.0_0_30.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 100$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

    if True:  # Sparse Linear Models

        # average error ----------------------------------------------------------------------------
        x_min = 2000
        y_min = -0.001
        y_max = 0.1

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_10.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 10, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_15.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 15, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_20.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 20, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_25.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 25, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_min=y_min, y_max=y_max,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_30.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        x_min = 2_000
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])  # for uniquness
        df = pd.concat((df, df_2), axis=0)
        df = df[df["iteration"] <= 50]
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_avgmse" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        # Error at 10 ------------------------------------------------------------------------------

        x_min = 2000
        y_min = -0.1
        y_max = 7

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_10.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 10, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_15.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 15, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_20.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 20, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_25.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 25, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_30.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        df.rename(columns={"approx_value_at_k": "approx_value"}, inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] +  "_approx_at_k" + ".pdf"
        draw_approx_curve(df=df, error_type="approx_value",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="MSE at 10",
                          x_label="model evaluations",
                          horizontal_line_y=0.,
                          save_name=save_name)


        # P@10 --------------------------------------------------------------------------

        x_min = 2000
        y_min = 0
        y_max = 1

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_10.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 10, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_15.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 15, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_20.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_25.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 25, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)

        file_name = "results_old/sln_final/sparse_linear_model_30_2_30_0.0_0_30.csv"
        file_name_second = "results/sln_final/" + "1_" + file_name.split("/")[-1]
        plot_title = r"SOUM (0 - 30, $N = 30$, $s_0 = 2$, $d = 30$, $g = 50$)"
        df = pd.read_csv(file_name)
        df_2 = pd.read_csv(file_name_second)
        df_2['iteration'] = df_2['iteration'] + max(df['iteration'])
        df = pd.concat((df, df_2), axis=0)
        df.drop(columns=["approx_value"], inplace=True)
        save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + file_name.split("/")[-1].split(".")[1] + "_precision" + ".pdf"
        draw_approx_curve(df=df, error_type="precision_at_k",
                          figsize=fig_size, x_min=x_min, mean_aggregation=True, shading=True,
                          plot_title=plot_title,
                          y_max=y_max, y_min=y_min,
                          y_label="P@10",
                          x_label="model evaluations",
                          save_name=save_name)
