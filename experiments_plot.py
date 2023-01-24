import copy
import os

import time

import numpy as np
import pandas as pd
from scipy.special import binom
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from evaluation import draw_approx_curve
from games import ParameterizedSparseLinearModel, SparseLinearModel, SyntheticNeuralNetwork, \
    NLPLookupGame
from shapx import ShapleyInteractionsEstimator, PermutationSampling
from shapx.regression import RegressionEstimator

if __name__ == "__main__":
    biggest_budget = 2**14
    file = "1674567209_sparse_linear_model_0.05_30_2.csv"
    name = file.split("_")
    plot_title = file

    data = pd.read_csv(os.path.join("results/sln",file))
    # data = pd.read_csv(os.path.join("results/sln", save_name))
    draw_approx_curve(df=data,
                      figsize=(6, 5), x_min=int(0.01 * biggest_budget), shading="quant",
                      y_min=0, y_max=2,
                      plot_title=plot_title,
                      y_label="average squared distance", x_label="model evaluations")