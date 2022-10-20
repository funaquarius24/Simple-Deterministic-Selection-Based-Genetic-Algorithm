#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 09:49, 17/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieu1995                                                        %
#-------------------------------------------------------------------------------------------------------%

from numpy.random import uniform, normal, randint
from numpy import pi, sin, cos, zeros, minimum, maximum, abs, where, sign, save
from copy import deepcopy
from root import Root
import json, codecs
import random
import pickle

class BasePSO(Root):
    """
        The original version of: Particle Swarm Optimization
    """

    random.seed(100)

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100,
                 c1=1, c2=2, w_min=0.4, w_max=0.9, random_seed=None, terminal_value = None, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size
        self.c1 = c1            # [0-2]  -> [(1.2, 1.2), (0.8, 2.0), (1.6, 0.6)]  Local and global coefficient
        self.c2 = c2
        self.w_min = w_min      # [0-1] -> [0.4-0.9]      Weight of bird
        self.w_max = w_max
        self.best_function = None

        if random_seed is not None:
            random.seed(random_seed)
            self.set_random_seed(random_seed)
        
        self.terminal_value = None
        self.terminal_value_iteration = 0
        if terminal_value is not None:
            self.terminal_value = terminal_value
            

    def train(self, filename, data=None, epoch_count=0, v_list=None, pop=None):
        v_max = 0.5 * (self.ub - self.lb)
        v_min = zeros(self.problem_size)
        if data is None:

            pop = [self.create_solution() for _ in range(self.pop_size)]
            
            v_list = uniform(v_min, v_max, (self.pop_size, self.problem_size))
            epoch_count = 1
        elif 'epoch_count' in data.keys():
            epoch_count = data['epoch_count']
            pop = data['pop']
            v_list = data['v_list']
        else:
            pop_data = data['pop']
            pop = [self.create_readymade_solution(pop_data[i]) for i in range(self.pop_size)]
            v_list = uniform(v_min, v_max, (self.pop_size, self.problem_size))
            epoch_count = 1
            
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(epoch_count, self.epoch):
            # Update weight after each move count  (weight down)
            w = (self.epoch - epoch) / self.epoch * (self.w_max - self.w_min) + self.w_min
            for i in range(self.pop_size):
                v_new = w * v_list[i] + self.c1 * uniform() * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) +\
                            self.c2 * uniform() * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                x_new = pop[i][self.ID_POS] + v_new             # Xi(new) = Xi(old) + Vi(new) * deltaT (deltaT = 1)
                x_new = self.amend_position_random_faster(x_new)
                fit_new = self.get_fitness_position(x_new)
                pop[i] = [x_new, fit_new]

                # Update current position, current velocity and compare with past position, past fitness (local best)
                if fit_new < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_new, fit_new]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop_local, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch+1, g_best[self.ID_FIT]))
            data = {}
            data['pop'] = pop
            data['v_list'] = v_list
            data['epoch_count'] = epoch + 1

            self.best_function = g_best[self.ID_MIN_PROB][self.ID_FIT]

            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            if self.terminal_value is not None:
                if abs(self.best_function - self.terminal_value) <= 2.2204460e-16:
                    self.terminal_value_iteration = epoch - 1
                    break
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train


class PPSO(Root):
    """
        A variant version of PSO: Phasor particle swarm optimization: a simple and efficient variant of PSO
        Matlab code sent by one of the author: Ebrahim Akbari
        I convert matlab to python code
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=750, pop_size=100, **kwargs):
        Root.__init__(self, obj_func, lb, ub, verbose, kwargs)
        self.epoch = epoch
        self.pop_size = pop_size

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        v_max = 0.5 * (self.ub - self.lb)
        v_list = zeros((self.pop_size, self.problem_size))
        delta_list = uniform(0, 2*pi, self.pop_size)
        pop_local = deepcopy(pop)
        g_best = self.get_global_best_solution(pop=pop, id_fit=self.ID_FIT, id_best=self.ID_MIN_PROB)

        for epoch in range(0, self.epoch):
            for i in range(0, self.pop_size):
                aa = 2 * (sin(delta_list[i]))
                bb = 2 * (cos(delta_list[i]))
                ee = abs(cos(delta_list[i])) ** aa
                tt = abs(sin(delta_list[i])) ** bb

                v_list[i, :] = ee * (pop_local[i][self.ID_POS] - pop[i][self.ID_POS]) + tt * (g_best[self.ID_POS] - pop[i][self.ID_POS])
                v_list[i, :] = minimum(maximum(v_list[i], -v_max), v_max)

                x_temp = pop[i][self.ID_POS] + v_list[i, :]
                x_temp = minimum(maximum(x_temp, self.lb), self.ub)
                fit = self.get_fitness_position(x_temp)
                pop[i] = [x_temp, fit]

                delta_list[i] += abs(aa + bb) * (2 * pi)
                v_max = (abs(cos(delta_list[i])) ** 2) * (self.ub - self.lb)

                if fit < pop_local[i][self.ID_FIT]:
                    pop_local[i] = [x_temp, fit]

                ## batch size idea
                if self.batch_idea:
                    if (i + 1) % self.batch_size == 0:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
                else:
                    if (i + 1) % self.pop_size:
                        g_best = self.update_global_best_solution(pop, self.ID_MIN_PROB, g_best)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
            if self.terminal_value is not None:
                if abs(self.best_function - self.terminal_value) <= 2.2204460e-16:
                    t = self.iterate + 1
                    self.terminal_value_iteration = epoch
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train

