#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:40:53 2021

@author: antonucci
"""

import numpy as np
import pandas as pd
from pulp import *
import copy

VERBOSE = False
LOGGING = True


def t(m, f, v='primal', o='max'):
    d = len(m)
    output = np.zeros(d)
    if len(m.shape) == 3:  # Imprecise
        for problem in range(d):
            model = LpProblem("Credal", LpMaximize if o == 'max' else LpMinimize)
            if v == 'dual':
                x = pulp.LpVariable.dicts('x', range(d))
                model += lpSum(x[i] * f[i] for i in range(d))  # Objective function
                model += lpSum(x[i] for i in range(d)) == 1.0
                for i in range(d):
                    model += x[i] >= m[problem][0][i]
                    model += x[i] <= m[problem][1][i]
                if LOGGING:
                    model.writeLP("Problem" + str(problem) + ".lp")  # optional
                model.solve()
                output[problem] = model.objective.value()  # pulp.value(model.objective)
    else:
        if v == 'primal':
            output = np.dot(m.T, f)
        else:
            output = np.dot(m, f)
    return output


def contaminate(m, eps):
    d = len(m)
    m2 = np.array([[[0. for _ in range(d)] for _ in range(2)] for _ in range(d)])
    for _ in range(d):
        if np.count_nonzero(m[_]) == 1:
            m2[_][0] = m[_]
            m2[_][1] = m[_]
        else:
            m2[_][0] = m[_]*(1.0-eps)
            m2[_][1] = [(0 if m[_][i] == 0 else m[_][i]*(1.0-eps) + eps) for i in range(d)]
    return m2


class Chain:

    def __init__(self, name, n, states):
        self.name = name
        self.size = n
        self.states = states
        self.rewards = []
        self.T = []

    def set_t(self, m):
        self.T = m

    def set_rew(self, v):
        self.rewards = np.array(v)

    def print_name(self):
        for i in range(len(self.states)):
            print('n', self.states[i])
        print(self.name)
        print(self.T)

    def hitting(self, hitting, opt='max', verbose=False):
        results = pd.DataFrame(columns=self.states)
        one = np.array([1.0 for _ in range(len(self.states))])
        not_hitting = one - np.array(hitting)
        h0 = copy.deepcopy(hitting)
        for _ in range(self.size + 1):
            h1 = not_hitting * t(self.T, h0, 'dual', opt) + hitting
            results.loc[_] = h1
            h0 = h1
        if verbose:
            print(results)
        return results

    def conditional_hitting(self, phi1, phi2, opt='max'):
        one = np.array([1.0 for _ in range(len(self.states))])
        phi1_phi2 = phi1 * (one-phi2)
        h0 = copy.deepcopy(phi2)  # initial exp rew
        for _ in range(self.size + 1):
            h1 = phi2 + phi1_phi2 * t(self.T,  h0, 'dual', opt)
            h0 = h1

    def cumulative(self, phi, opt='max'):
        results = pd.DataFrame(columns=self.states)
        one = np.array([1.0 for _ in range(len(self.states))])
        not_phi = one - np.array(phi)
        h0 = copy.deepcopy(self.rewards)  # initial exp rew DEBUG
        for _ in range(self.size + 1):
            h1 = self.rewards + not_phi * t(self.T,  h0, 'dual', opt)
            results.loc[_] = h1
            h0 = h1
        return results

    def bounded_reward(self, phi1, phi2, r, opt='max'):
        one = np.array([1.0 for _ in range(len(self.states))])
        ones = np.array([[1.0 for _ in range(len(self.states))] for _ in range(r)])
        s_rew = [r-_ for _ in self.rewards]
        phi1_phi2 = phi1 * (one - phi2)
        not_phi = one - np.array(phi)
        #h0 = copy.deepcopy(phi)*self.rewards  # initial exp rew
        #for _ in range(self.size + 1):
        #    h1 = Ica * (self.rewards + t(tr,  h0, 'dual')) + Ia * rewards
        #    budget = a[dep] * h1[0] + l[dep] * h1[1]
        #    h0 = h1
        #budget += a[dep] * rewards[0] + l[dep] * rewards[1]
        #print('[T=%d] Budget = %2.5f' % (k + 1, budget))