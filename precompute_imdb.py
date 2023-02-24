"""This Module can be used to precompute model outpts for the language model for the NLPLookupGame"""
import random
import os

import pandas as pd
import tqdm

from games import NLPGame
from approximators.base import powerset

n = 14
N_SAMPLES = 1


if __name__ == "__main__":

    df = pd.read_csv("data/simplified_imdb.csv")
    df = df[df['length'] == n]

    sampled_n = 0

    save_dir = os.path.join("data", "nlp_values", str(n))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while sampled_n < N_SAMPLES:
        sentence_id = random.choice(list(df["id"].values))
        sentence = str(df[df["id"] == sentence_id]["text"].values[0])
        files = list(os.listdir(os.path.join("data", "nlp_values", str(n))))
        sentence_path = str(sentence_id) + ".csv"
        if sentence_path in files:
            continue
        game = NLPGame(input_text=sentence)
        N = set(range(0, n))
        calls = []

        print(f"Starting sample {sampled_n + 1} from  {N_SAMPLES}.\n"
              f"sentence_id: {sentence_id}\n"
              f"Sentence: {sentence}")
        pbar = tqdm.tqdm(total=2 ** n)

        for S in powerset(N, min_size=0, max_size=n):
            value = game.set_call(S)
            S_storage = 's'
            for player in S:
                S_storage += str(player)
            calls.append({"set": S_storage, "value": value})
            pbar.update(1)

        storage_df = pd.DataFrame(calls)
        storage_df.to_csv(os.path.join(save_dir, sentence_path), index=False)
        sampled_n += 1
        pbar.close()
