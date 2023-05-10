import numpy as np
from scipy.special import binom

from approximators.base import powerset


class ExhaustiveSearch:

    def __init__(self, N):
        self.N = N
        self.n = len(N)
        self.shapley_values = np.zeros(self.n)

    def compute_exact_shapley_values(self, game):
        all_player_set = set(self.get_all_players())
        for subset_size in range(1, self.n + 1):
            weight_s_plus = 1 / (self.n * binom(self.n - 1, subset_size - 1))
            weight_s_minus = 1 / (self.n * binom(self.n - 1, subset_size))
            for subset in powerset(all_player_set, subset_size, subset_size):
                subset_set = set(subset)
                game_value = game(subset)
                for player in subset_set:
                    self.shapley_values[player] = self.shapley_values[player] + weight_s_plus * game_value
                for player in all_player_set - subset_set:
                    self.shapley_values[player] = self.shapley_values[player] - weight_s_minus * game_value
        return self.shapley_values

    def approximate_shapley_values(self) -> dict:
        raise NotImplementedError

    def get_estimates(self):
        return self.shapley_values

    def get_name(self) -> str:
        return "ExhaustiveSearch"

    def get_all_players(self):
        return list(range(self.n))

