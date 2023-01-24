def _constant_c(self, game):
    rslt = self.init_results()
    for T in powerset(self.N, 0, self.s - 1):
        game_val = game(T)
        t = len(T)
        for S in powerset(self.N, self.min_order, self.s):
            rslt[len(S)][S] += game_val * self.weights[t, len(set(S).intersection(T))]

    for T in powerset(self.N, self.n - self.s + 1, self.n):
        game_val = game(T)
        t = len(T)
        for S in powerset(self.N, self.min_order, self.s):
            rslt[len(S)][S] += game_val * self.weights[t, len(set(S).intersection(T))]
    return rslt


    def constant_budget(self):
        rslt = 0
        for t in range(self.s):
            rslt += 2 * binom(self.n, t)
        return rslt