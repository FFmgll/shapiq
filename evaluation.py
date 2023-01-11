import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    sns.set_theme(style="darkgrid")

    file_name = "results/sparse_linear_model_10.csv"
    #file_name = "interaction_sampling_k.csv"

    data = pd.read_csv(file_name)

    shapley_interaction_order = str(data["shapley_interaction_order"][0])
    data = data[data['full'].isin(['not full', np.NAN])].drop(columns=['full'])
    #data = data.drop(columns=['pairing', 'sampling_kernel', 'shapley_interaction_order'])
    data = data[data['n_absolute'] > 0]

    data = data.rename(columns={"interaction_index": "Interaction Index", "approx_type": "Method"})

    plot_title = ' '.join(("Order:", shapley_interaction_order))

    ax = sns.lineplot(
        x='n_absolute', y="approx_value",
        hue="Interaction Index", style='Method',
        data=data
    )
    ax.set(ylabel='Approximation Error', xlabel='N Samples (absolute)',
           title=plot_title,
           #ylim=(-1, 5)
    )
    plt.show()


    """
    # approximation only
    #data = data[data["approx_type"].isin(['approximation',np.NAN])]
    #subset sizes
    #data = data[data["n_absolute"]>2000]
    #select single index
    #data = data[data["interaction_index"].isin(["SII",np.NAN])]
    #filter only approximation
    #data = data[data["approx_type"].isin(["approximation"])]
    #data = data.drop(columns=['sampling_kernel'])
    
    data_grouped = data.groupby(
        by=['iteration', 'approx_type', 'interaction_index', 'n_absolute', 'sampling_kernel', 'pairing', 'full'],
        dropna=False).agg({'approx_value': ['mean', 'std'], 'n_relative': 'first'}).reset_index()
    for interaction_index in data_grouped.interaction_index.unique():
        data_df = data_grouped[data_grouped.interaction_index == interaction_index]
    

    interaction_types = data["interaction_index"].unique()

    #Approximation Comparison
    for interaction_type in interaction_types:
        sampling_kernels = data["sampling_kernel"].dropna().unique()
        for kernel in sampling_kernels:
            #Select Kernel
            data_plot = data[data['sampling_kernel'].isin([kernel,np.NAN])]
            data_plot = data_plot[data_plot['interaction_index'].isin([interaction_type,np.NAN])]
            data_plot = data_plot[data_plot['full'].isin(['not full', np.NAN])].drop(columns=['full'])
            plt.figure()
            ax = sns.lineplot(
                x='n_absolute', y="approx_value",
                hue="approx_type", style='interaction_index',
                data=data_plot
            )
            ax.set(ylabel='Approx Error', xlabel='N Samples (relative)', title="Index: "+interaction_type+" Order: "+str(shapley_interaction_order) + " kernel:"+kernel)
            plt.show()

    #Visualization Kernel Comparison
    for interaction_type in interaction_types:
        data_plot = data[data["approx_type"].isin(['approximation',np.NAN])]
        #Select Kernel
        data_plot = data_plot[data_plot['interaction_index'].isin([interaction_type,np.NAN])]
        data_plot = data_plot[data_plot['full'].isin(['not full', np.NAN])].drop(columns=['full'])
        data_plot = data_plot[data_plot['sampling_kernel'].isin(['unif-size', np.NAN])]

        plt.figure()
        ax = sns.lineplot(
            x='n_absolute', y="approx_value",
            hue="sampling_kernel", style='interaction_index',
            data=data_plot
        )
        ax.set(ylabel='Approx Error', xlabel='N Samples (relative)', title="Index: "+interaction_type+" Order: "+str(shapley_interaction_order) + " kernel:"+kernel)
        plt.show()
        """

