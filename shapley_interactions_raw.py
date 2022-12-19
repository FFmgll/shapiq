import scipy
import numpy as np
import itertools
import random
from collections import Counter
import copy
from scipy.special import binom
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))




class Game:
    import numpy as np
    def __init__(self,n):
        self.weights = np.random.rand(n)
        self.n = n
        self.intx2 = 0
        self.intx3 = 0
    def call(self,x):
        return np.dot(x,self.weights)+x[1]*x[2]*self.intx2+self.intx3*x[1]*x[2]*x[3]


    def set_call(self,S):
        x=np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)

class NN:
    def __init__(self,n):
        self.weights_1 = np.random.normal(loc=0,scale=10,size=((100,n)))
        self.bias_1 = np.random.normal(loc=0,scale=1,size=(100))
        self.weights_2 = np.random.normal(loc=0,scale=0.5,size=((10,100)))
        self.bias_2 = np.random.normal(loc=0,scale=1)
        self.weights_3 = np.random.normal(loc=0,scale=0.05,size=((1,10)))
        #self.bias_3 = np.random.normal(loc=1,scale=0.05)
        self.n = n
        self.intx2 = 0
        self.intx3 = 0

    def call(self,x):
        return sigmoid(np.dot(self.weights_3,np.maximum(0,np.dot(self.weights_2,np.maximum(0,np.dot(self.weights_1,x)+self.bias_1)) + self.bias_2)))
    def set_call(self,S):
        x=np.zeros(self.n)
        x[list(S)] = 1
        return self.call(x)-self.call(np.zeros(self.n))


n=15
N = set(range(n))


#game = Game(n)
game=NN(n)
game_fun = game.set_call

min_order = 2
s = 2

type = "SII"

def kernel_m(t):
    if type == "SII":
        return np.math.factorial(n - t - s) * np.math.factorial(t) / np.math.factorial(n - s + 1)
    if type == "STI":
        return s * np.math.factorial(n - t - 1) * np.math.factorial(t) / np.math.factorial(n)
    if type == "SFI":
        return np.math.factorial(2 * s - 1) / np.math.factorial(s - 1) ** 2 * np.math.factorial(n - t - 1) * np.math.factorial(t + s - 1) / np.math.factorial(n + s - 1)

def powerset(iterable, min_size=-1, max_size=None):
    if max_size==None and min_size > -1:
        max_size=min_size
    s = list(iterable)
    if max_size==None:
        max_size = len(s)
    else:
        max_size=min(max_size,len(s))
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(max(min_size, 0), max_size + 1))


def init_results():
    results = {}
    for k in range(min_order, s + 1):
        results[k] = np.zeros(np.repeat(n, k))
    return results

def compute_interactions_complete_k(game_fun, k):
    results = init_results()
    for T in powerset(N, k, k):
        game_eval = game_fun(T)
        t = len(T)
        for S in powerset(N, min_order, s):
            s_t = len(set(S).intersection(T))
            results[len(S)][S] += game_eval * weights[t, s_t]
    return results

def kernel_q(t,sampling_kernel):
    if sampling_kernel == "ksh":
        return np.math.factorial(n-t-s)*np.math.factorial(t-s)/np.math.factorial(n-s+1)
    if sampling_kernel == "faith":
        return np.math.factorial(n-t-1)*np.math.factorial(t-1)/np.math.factorial(n-1)
    if sampling_kernel == "unif-size":
        return 1
    if sampling_kernel == "unif-set":
        return binom(n,t)

def init_sampling_weights(sampling_kernel):
    q = np.zeros(n+1)
    for t in range(s,n-s+1):
        q[t] = kernel_q(t,sampling_kernel)
    return q


weights = np.zeros((n + 1, s + 1))
for t in range(0, n + 1):
    for k in range(max(0, s + t - n), min(s, t) + 1):
        weights[t, k] = (-1) ** (s - k) * kernel_m(t - k)


budget = 10000


import matplotlib.pyplot as plt

sampling_kernel_list = ["ksh","faith","unif-size","unif-set"]
#sampling_kernel_list = ["ksh"]
pairing_list = [True,False]

incomplete_subsets = [3,4,5]
incomplete_subsets = [5,6,7,8,9]
q = init_sampling_weights("unif-set")

subset_weight_vector = q / np.sum(q[incomplete_subsets])

exact = {}
exact = init_results()
exact_all = {}
exact_all = init_results()
for k in range(n+1):
    if k in incomplete_subsets:
        exact[s] += compute_interactions_complete_k(game_fun,k)[s]
    else:
        exact_all[s] += compute_interactions_complete_k(game_fun,k)[s]


def constant_R(incomplete_subsets, q):
    R = 0
    for t in incomplete_subsets:
        R += q[t]
    return R

R = constant_R(incomplete_subsets,q)

subset_sizes_samples = random.choices(incomplete_subsets, k=budget, weights=subset_weight_vector[incomplete_subsets])

vars = {}
for k in incomplete_subsets:
    vars[k] = 0



for sampling_kernel in sampling_kernel_list:
    print(sampling_kernel)
    n_evals = 0
    results = init_results()
    q = init_sampling_weights(sampling_kernel)
    subset_weight_vector = q / np.sum(q[incomplete_subsets])

    for pairing in pairing_list:
        errors = []
        errors_step = []
        for k in subset_sizes_samples:
            T = set(np.random.choice(n, k, replace=False))
            p = subset_weight_vector[k]/binom(n,k)
            game_eval = game_fun(T)
            game_eval_c = game_fun(N-T)
            t = len(T)
            for S in powerset(N, min_order, s):
                size_intersection = len(set(S).intersection(T))
                results[len(S)][S] += game_eval * weights[t, size_intersection]/p
                if pairing:
                    results[len(S)][S] += game_eval_c * weights[n-t, s-size_intersection]/p

                #results[len(S)][S] += 1
            if pairing:
                n_evals += 2
                estimate = results[s]/(2*n_evals)
            else:
                n_evals += 1
                estimate = results[s] / (n_evals)
            if (n_evals*100/ budget) % 5 == 0:
                error_val = np.sum((estimate-exact[s])**2)
                error_ref = np.sum((exact[s])**2)
                print("approx-k",error_val," ref:",error_ref)
                #print("approx-all",np.sum((results[s]/n_evals-exact_all[s])**2))
                #print("k-all",np.sum((exact[s]-exact_all[s])**2))
                errors_step.append(error_val)
            errors.append(np.sum((results[s]/n_evals-exact[s])**2))

        results[s] /= n_evals

        plt.figure()
        plt.title(sampling_kernel+"_"+str(pairing))
        #plt.vlines(x=error_ref,ymin=error_ref,ymax=error_ref)
        plt.plot(errors_step)
        plt.show()
