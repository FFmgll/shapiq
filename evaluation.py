import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

#  COLORS = {'SII': '#058ED9', 'STI': '#2D3142', 'SFI': '#CC2D35'}  # online inclusive colors
COLORS = {'SII': '#44cfcb', 'STI': '#7d53de', 'SFI': '#ef27a6'}
MARKERS = {'SII': 'X', 'STI': 'o', 'SFI': "s"}
LABELS = {'SII': 'Shapley Interaction', 'STI': 'Shapley Taylor', 'SFI': "Shapley Faith"}

STD_ALPHA = 0.10


def draw_approx_curve(df: pd.DataFrame, figsize: tuple = (10, 10),
                      x_min: int = None, y_max: float = None, y_min: float = None, shading: str = None, plot_title: str = None,
                      x_label: str = None, y_label: str = None, save_name: str = None):

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
            subset = subset.groupby(by=['iteration', 'n_absolute', "Method"]).aggregate({"approx_value": "mean"}).reset_index()

        baseline = subset[subset["Method"] == "baseline"]
        approximation = subset[subset["Method"] == "approximation"]

        baseline_mean = baseline.groupby(by=grouping)["approx_value"].quantile(q=0.5).reset_index()
        baseline_quantile_l = baseline.groupby(by=grouping)["approx_value"].quantile(q=0.25).reset_index()["approx_value"]
        baseline_quantile_h = baseline.groupby(by=grouping)["approx_value"].quantile(q=0.75).reset_index()["approx_value"]
        approximation_mean = approximation.groupby(by=grouping)["approx_value"].quantile(q=0.5).reset_index()
        approximation_quantile_l = approximation.groupby(by=grouping)["approx_value"].quantile(q=0.25).reset_index()["approx_value"]
        approximation_quantile_h = approximation.groupby(by=grouping)["approx_value"].quantile(q=0.75).reset_index()["approx_value"]

        if shading == 'std':
            approximation_std = approximation.groupby(by=grouping)["approx_value"].std().reset_index()
            approximation_quantile_l = approximation_mean["approx_value"] - approximation_std["approx_value"]
            approximation_quantile_h = approximation_mean["approx_value"] + approximation_std["approx_value"]
            baseline_std = baseline.groupby(by=grouping)["approx_value"].std().reset_index()
            baseline_quantile_l = baseline_mean["approx_value"] - baseline_std["approx_value"]
            baseline_quantile_h = baseline_mean["approx_value"] + baseline_std["approx_value"]

        axis.plot(baseline_mean["n_absolute"], baseline_mean["approx_value"],
                  ls="dashed", color=COLORS[interaction_index], linewidth=1,
                  marker=MARKERS[interaction_index], mec="white")
        axis.plot(approximation_mean["n_absolute"], approximation_mean["approx_value"],
                  ls="solid", color=COLORS[interaction_index], linewidth=1,
                  marker=MARKERS[interaction_index], mec="white")

        if shading is not None:
            try:
                axis.fill_between(approximation_mean["n_absolute"],
                                  approximation_quantile_l,
                                  approximation_quantile_h,
                                  color=COLORS[interaction_index], alpha=STD_ALPHA, linewidth=0.)
            except Exception as e:
                print("Error occurred in baseline std:", e)
            try:
                axis.fill_between(approximation_mean["n_absolute"],
                                  baseline_quantile_l,
                                  baseline_quantile_h,
                                  color=COLORS[interaction_index], alpha=STD_ALPHA, linewidth=0.)
            except Exception as e:
                print("Error occurred in approximation std:", e)

    title_1 = "$\\bf{Indices}$"
    axis.plot([], [], label=title_1, color="none")
    for interaction_index in data["Interaction Index"].unique():
        axis.plot([], [], color=COLORS[interaction_index],
                  marker=MARKERS[interaction_index], label=LABELS[interaction_index], mec="white")

    title_2 = "$\\bf{Methods}$"
    axis.plot([], [], label=title_2, color="none")
    axis.plot([], [], color="black", ls="solid", label="SHAPX")
    axis.plot([], [], color="black", ls="dashed", label="baseline")

    leg = axis.legend()
    for item, label in zip(leg.legendHandles, leg.texts):
        if label._text in [title_1, title_2]:
            width = item.get_window_extent(fig.canvas.get_renderer()).width
            label.set_ha('left')
            label.set_position((-2 * width, 0))

    plt.title(plot_title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_ylim((y_min, y_max))
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name)
    plt.show()


if __name__ == "__main__":
    file_name = "results/synth_neural_network_14.csv"
    plot_title = r"Synthetic Neural Network ($l = 2$, $n = 14$, $g = 20$)"
    save_name = "plots/" + file_name.split("/")[-1].split(".")[0] + ".png"
    draw_approx_curve(df=pd.read_csv(file_name), figsize=(6, 5), x_min=2000, shading="quant",
                      plot_title=plot_title,
                      y_max=0.00044,
                      y_label="average squared distance", x_label="model evaluations",
                      save_name=save_name)
