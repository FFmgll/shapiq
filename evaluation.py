import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")

    data = pd.read_csv("third_run.csv")
    data = data[data['full'].isin(['full', np.NAN])].drop(columns=['full'])
    data = data[data['pairing'].isin(['not-paired', np.NAN])].drop(columns=['pairing'])
    data = data.drop(columns=['sampling_kernel'])

    """
    data_grouped = data.groupby(
        by=['iteration', 'approx_type', 'interaction_index', 'n_absolute', 'sampling_kernel', 'pairing', 'full'],
        dropna=False).agg({'approx_value': ['mean', 'std'], 'n_relative': 'first'}).reset_index()
    for interaction_index in data_grouped.interaction_index.unique():
        data_df = data_grouped[data_grouped.interaction_index == interaction_index]
    """

    ax = sns.lineplot(
        x='n_relative', y="approx_value",
        hue="approx_type", style='interaction_index',
        data=data
    )
    ax.set(ylabel='Approx Error', xlabel='N Samples (relative)', title='Interaction order: 4')
    plt.show()
