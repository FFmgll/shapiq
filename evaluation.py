import pandas as pd

from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

COLORS_SHAPLEY = {'SII': '#058ED9', 'STI': '#2D3142', 'SFI': '#CC2D35'}  # online inclusive colors
COLORS = {'SHAP-IQ': '#ef27a6', 'baseline': '#7d53de'}
BACKGROUND_COLOR = '#f8f8f8'
MARKERS = {'SII': 'X', 'STI': 'o', 'SFI': "s"}
LINESTYLES = {'SII': 'dashed', 'STI': 'dotted', 'SFI': "solid"}
LABELS = {'SII': 'Shapley Interaction', 'STI': 'Shapley Taylor', 'SFI': "Faith-SHAP", 'U-KSH': "Unbiased Kernel Shap", "U-KSH-R": "Unbiased Kernel Shap (replacement)"}

STD_ALPHA = 0.10


def draw_approx_curve(df: pd.DataFrame, figsize: tuple = (10, 10), error_type: str = "approx_value",
                      mean_aggregation: bool = False, shading: bool = True,
                      x_min: int = None, y_max: float = None, y_min: float = None, plot_title: str = None,
                      x_label: str = None, y_label: str = None, save_name: str = None, max_computation_cost_n: int = None,
                      horizontal_line_y: float = None, shap=False):

    grouping = ['n_absolute']

    data = df.copy()
    data = data[data['sampling'].isin(['const and sampling', np.NAN])].drop(columns=['sampling'])
    data = data[data['n_absolute'] > 0]

    data = data.rename(columns={"interaction_index": "Interaction Index", "approx_type": "Method"})

    fig, axis = plt.subplots(1, 1, figsize=figsize)

    for interaction_index in data["Interaction Index"].unique():
        subset = data[data["Interaction Index"] == interaction_index]
        if x_min is not None:
            subset = subset[subset["n_absolute"] >= x_min]

        if 'inner_iteration' in subset.columns:
            subset = subset.groupby(by=['iteration', 'n_absolute', "Method"]).aggregate({error_type: "mean"}).reset_index()

        baseline = subset[subset["Method"] == "baseline"]
        approximation = subset[subset["Method"] == "approximation"]

        if not mean_aggregation == "median":
            baseline_mean = baseline.groupby(by=grouping)[error_type].quantile(q=0.5).reset_index()
            baseline_quantile_l = baseline.groupby(by=grouping)[error_type].quantile(q=0.25).reset_index()[error_type]
            baseline_quantile_h = baseline.groupby(by=grouping)[error_type].quantile(q=0.75).reset_index()[error_type]
            approximation_mean = approximation.groupby(by=grouping)[error_type].quantile(q=0.5).reset_index()
            approximation_quantile_l = approximation.groupby(by=grouping)[error_type].quantile(q=0.25).reset_index()[error_type]
            approximation_quantile_h = approximation.groupby(by=grouping)[error_type].quantile(q=0.75).reset_index()[error_type]
        else:
            baseline_mean = baseline.groupby(by=grouping)[error_type].mean().reset_index()
            approximation_mean = approximation.groupby(by=grouping)[error_type].mean().reset_index()
            approximation_std = approximation.groupby(by=grouping)[error_type].std().reset_index()
            approximation_quantile_l = approximation_mean[error_type] - approximation_std[error_type]
            approximation_quantile_h = approximation_mean[error_type] + approximation_std[error_type]
            baseline_std = baseline.groupby(by=grouping)[error_type].std().reset_index()
            baseline_quantile_l = baseline_mean[error_type] - baseline_std[error_type]
            baseline_quantile_h = baseline_mean[error_type] + baseline_std[error_type]


        axis.plot(baseline_mean["n_absolute"], baseline_mean[error_type],
                  ls=LINESTYLES[interaction_index], color=COLORS["baseline"], linewidth=1,
                  marker=MARKERS[interaction_index], mec="white")
        axis.plot(approximation_mean["n_absolute"], approximation_mean[error_type],
                  ls=LINESTYLES[interaction_index], color=COLORS["SHAP-IQ"], linewidth=1,
                  marker=MARKERS[interaction_index], mec="white")

        if shading:
            try:
                axis.fill_between(baseline_mean["n_absolute"],
                                  baseline_quantile_l,
                                  baseline_quantile_h,
                                  color=COLORS["baseline"], alpha=STD_ALPHA, linewidth=0.)
            except Exception as e:
                print("Error occurred in baseline std:", e)
            try:
                axis.fill_between(approximation_mean["n_absolute"],
                                  approximation_quantile_l,
                                  approximation_quantile_h,
                                  color=COLORS["SHAP-IQ"], alpha=STD_ALPHA, linewidth=0.)
            except Exception as e:
                print("Error occurred in approximation std:", e)

    if not shap:
        title_1 = "$\\bf{Indices}$"
        axis.plot([], [], label=title_1, color="none")
        for interaction_index in data["Interaction Index"].unique():
            axis.plot([], [], color="black", ls=LINESTYLES[interaction_index],
                      marker=MARKERS[interaction_index], label=LABELS[interaction_index], mec="white")
        title_2 = "$\\bf{Methods}$"
        axis.plot([], [], label=title_2, color="none")
        axis.plot([], [], color=COLORS["SHAP-IQ"], ls="solid", label="SHAP-IQ")
        axis.plot([], [], color=COLORS["baseline"], ls="solid", label="baseline")

        leg = axis.legend()
        """
        for item, label in zip(leg.legendHandles, leg.texts):
            if label._text in [title_1, title_2]:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-2 * width, 0))
        """


    else:
        title_2 = "$\\bf{Methods}$"
        axis.plot([], [], label=title_2, color="none")
        axis.plot([], [], color=COLORS["SHAP-IQ"], ls="solid", label="SHAP-IQ")
        axis.plot([], [], color=COLORS["baseline"], ls="solid", label="KernelSHAP")
        axis.plot([], [], color=COLORS["baseline"], ls="dashed", label="Permutation Sampling")

        leg = axis.legend()
        """
        for item, label in zip(leg.legendHandles, leg.texts):
            if label._text in [title_2]:
                width = item.get_window_extent(fig.canvas.get_renderer()).width
                label.set_ha('left')
                label.set_position((-2 * width, 0))
        """



    plt.title(plot_title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_ylim((y_min, y_max))
    axis.set_facecolor(BACKGROUND_COLOR)

    if horizontal_line_y is not None:
        axis.axhline(y=horizontal_line_y, ls="dotted", c="gray")

    if max_computation_cost_n is not None:
        max_computation_cost = 2 ** max_computation_cost_n
        labels = [item.get_text() for item in axis.get_xticklabels()]
        for i in range(len(labels)):
            value = int(labels[i])
            labels[i] = labels[i] + "\n" + str(round(value / max_computation_cost, 2))
        axis.set_xticklabels(labels)

    if error_type == "approx_value":
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    else:
        axis.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    #plt.show()


def draw_shapley_values(uksh, uksh_rep, sii, sti, sfi, labels: list = None, figsize: tuple = (10, 10), save_name: str = None, plot_title: str = None):
    x = np.arange(len(uksh))

    alpha = 0.3

    colors = {index: tuple([*to_rgb(color)] + [alpha]) for index, color in COLORS_SHAPLEY.items()}
    colors_edge = {index: tuple([*to_rgb(color)] + [1.]) for index, color in COLORS_SHAPLEY.items()}

    n_approximators = 4
    width = 0.18

    fig, axis = plt.subplots(1, 1, figsize=figsize)

    title_1 = "$\\bf{Method}$"
    axis.plot([], [], label=title_1, color="none")

    axis.bar(x - width * 2, uksh, width * 0.7, label=LABELS["U-KSH"], fill=False, hatch='//////')
    axis.bar(x - width, uksh_rep, width * 0.7, label=LABELS["U-KSH-R"], fill=False, hatch='//////', edgecolor="gray")
    axis.bar(x, sii, width * 0.7, label=LABELS["SII"], color=colors["SII"], edgecolor=colors_edge["SII"])
    axis.bar(x + width, sti, width * 0.7, label=LABELS["STI"], color=colors["STI"], edgecolor=colors_edge["STI"])
    axis.bar(x + width * 2, sfi, width * 0.7, label=LABELS["SFI"], color=colors["SFI"], edgecolor=colors_edge["SFI"])

    axis.set_xlim(0 - width * 2 - 0.2, len(x) - 1 + width * 2 + 0.2)

    if labels is not None:
        axis.xaxis.set_major_locator(FixedLocator(x))
        #axis.set_xticks(ticks=x)
        #axis.set_xticklabels(labels)
        #axis.xaxis.set_ticks(x)
        print(len(labels))
        axis.xaxis.set_ticklabels(labels)

    leg = axis.legend(loc='best')
    for item, label in zip(leg.legendHandles, leg.texts):
        if label._text == title_1:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-2 * width, 0))

    plt.title(plot_title)
    axis.set_xlabel("Features")
    axis.set_ylabel("Shapley Values")
    axis.set_facecolor(BACKGROUND_COLOR)

    #axis.axhline(y=0, ls="solid", c="gray", linewidth=1., alpha=0.5)
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


if __name__ == "__main__":
    fig_size = (5, 5)

    if False:  # LM --------------------------------------------------------------------

        file_name = "results/finished/language_model_14_4.csv"
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

        file_name = "results/finished/language_model_14_3.csv"
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

        file_name = "results/finished/language_model_14_2.csv"
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

        file_name = "results/finished/language_model_14_2.csv"
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

        file_name = "results/finished/language_model_14_1.csv"
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

    if False:  # Gradient Boosted Trees

        file_name = "results/finished/tabular_game_14_2.csv"
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


        file_name = "results/finished/tabular_game_14_1.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_10.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_15.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_20.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_25.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_30.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_10.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_15.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_20.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_25.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_30.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_10.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_15.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_20.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_25.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_100_0.0_0_30.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_10.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_15.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_20.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_25.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_30.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_10.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_15.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_20.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_25.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_30.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_10.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_15.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_20.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_25.csv"
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

        file_name = "results/sln_final/sparse_linear_model_30_2_30_0.0_0_30.csv"
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
