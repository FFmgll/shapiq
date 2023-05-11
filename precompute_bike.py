"""This Module can be used to precompute model outputs for the bike dataset TabularLookUpGame"""
import random
import os

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import tqdm

import datasets
from games import MachineLearningMetaGame, MachineLearningGame
from approximators.base import powerset


RANDOM_STATE = 1
n = 12
N_SAMPLES = 100


if __name__ == "__main__":

    sampled_n = 0

    dataset = datasets.BikeSharing()
    x_data = dataset.x_data.values
    y_data = dataset.y_data.values

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.3, random_state=RANDOM_STATE)

    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)

    print("Training Score", model.score(x_train, y_train))
    print("Testing Score", model.score(x_test, y_test))

    meta_game = MachineLearningMetaGame(model=model, n=n, regression=True, dataset_name="bike")

    df = meta_game.x_data

    save_dir = os.path.join("games/data", "bike_" + str(RANDOM_STATE), str(n))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while sampled_n < N_SAMPLES:
        data_index = random.randint(0, meta_game.n_samples)
        files = list(os.listdir(os.path.join("games/data", "bike_" + str(RANDOM_STATE), str(n))))
        sample_path = str(data_index) + ".csv"
        if sample_path in files:
            continue
        game = MachineLearningGame(meta_game=meta_game, data_index=data_index)
        N = set(range(0, n))
        calls = []

        print(f"Starting sample {sampled_n + 1} from  {N_SAMPLES}.\n"
              f"data_index: {data_index}\n"
              f"Sentence: {game.data_point}")
        pbar = tqdm.tqdm(total=2 ** n)

        for S in powerset(N, min_size=0, max_size=n):
            value = game.set_call(S)
            S_storage = 's'
            for player in S:
                S_storage += str(player)
            calls.append({"set": S_storage, "value": value})
            pbar.update(1)

        storage_df = pd.DataFrame(calls)
        storage_df.to_csv(os.path.join(save_dir, sample_path), index=False)
        sampled_n += 1
        pbar.close()
