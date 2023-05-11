import os
import random
from pathlib import Path

import pandas as pd


class LookUpGame:
    """Wrapper for the Machine Learning Game to use precomputed model outputs for faster runtime in
    experimental settings."""

    def __init__(
            self,
            data_folder: str,
            n: int,
            data_id: int = None,
            used_ids: set = None,
            set_zero: bool = True
    ):
        self.n = n

        # to not use the same instance twice if we use the game multiple times
        if used_ids is None:
            used_ids = set()
        self.used_ids = used_ids

        # get paths to the files containing the value function calls
        game_path = Path(__file__).parent.absolute()
        data_dir = os.path.join(game_path, "data")
        instances_dir = os.path.join(data_dir, data_folder, str(n))

        # randomly select a file if none was explicitly provided
        if data_id is None:
            files = os.listdir(instances_dir)
            files = list(set(files) - used_ids)
            if len(files) == 0:
                files = os.listdir(instances_dir)
                self.used_ids = set()
            data_id = random.choice(files)
            data_id = data_id.split(".")[0]
        self.data_id = str(data_id)
        self.game_name = '_'.join(("LookUpGame", data_folder, str(n), self.data_id))

        self.used_ids.add(self.data_id + ".csv")

        # load file containing value functions into easily accessible dict
        file_path = os.path.join(instances_dir, self.data_id + ".csv")
        self.df = pd.read_csv(file_path)
        self.storage = {}
        for _, sample in self.df.iterrows():
            S_id = sample["set"]
            value = float(sample["value"])
            self.storage[S_id] = value

        # normalize empty coalition to zero (v_0({}) = 0)
        self.empty_value = 0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S):
        S_id = 's'
        for player in sorted(S):
            S_id += str(player)
        return self.storage[S_id] - self.empty_value

    def get_name(self):
        return self.game_name
