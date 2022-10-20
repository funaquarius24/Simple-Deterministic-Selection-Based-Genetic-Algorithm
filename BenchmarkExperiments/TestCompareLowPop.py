import numpy as np
import time
import pandas as pd

IN_COLAB = False
import sys
if IN_COLAB:
  
  scripts_dir = '/drive/My Drive/Colab Notebooks/scripts/'
  sys.path.insert(1, scripts_dir)
# from opfunu.cec_basic.cec2014_nobias import *
# from mealpy.swarm_based.PSO import BasePSO

# insert at 1, 0 is the script path (or '' in REPL)
else:
    sys.path.insert(1, 'scripts')

from geneticalgorithm import geneticalgorithm as ga
from geneticalgorithm1 import geneticalgorithm1 as ga1
from geneticalgorithmOptd import geneticalgorithmOptd as gaOptd

from PSO import BasePSO
from BBO import BaseBBO

import pandas as pd

# import benchmark_func as bf

import importlib
bf = importlib.import_module("benchmark_func")

# class_ = getattr(bf, "Ackley")
# instance = class_(2)

dimension = 2
population_size = 15
max_iter = 10
num_runs = 5
seeds = np.random.randint(0, 1000, num_runs)
varbound=np.array([[-100, 100]]*dimension)

timestr = time.strftime("%Y%m%d-%H%M%S")
filename = f'results/low_pop/Run_Results-{timestr}'

one_args = getattr(bf, "__oneArgument__")
two_args = getattr(bf, "__twoArgument__")

result = {}
result["function"] = None
result["run"] = None
result["seed"] = None
result["algo"] = None
result["pop_size"] = None
result["g_opt_val"] = None
result["func_val"] = None

result["max_iter"] = None
result["best_iter"] = None

results = []

def createPop(dimension=2, population_size=15):
  var=np.zeros(dimension) 

  pop=np.array([np.zeros(dimension)]*population_size)
  for p in range(0, population_size):
      for i in range(0, dimension):
          var[i]=np.random.randint(varbound[i][0],\
                  varbound[i][1]+1) 
      pop[p] = var
  data = {}
  data['pop'] = pop.copy()

  # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
  # print(data['pop'])

  return data
# result = instance.get_func_val(np.array([0.0, 5.0]))

# print(getattr(bf, "__oneArgument__"))

results_df = pd.DataFrame(results)

def normGa(obj_func, pop_data=None, seed = 777):
  algorithm_param = {'max_num_iteration': max_iter,\
                   'population_size':population_size,\
                   'mutation_probability':0.2,\
                   'elit_ratio': 0.1,\
                   'crossover_probability': .7,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
  model=ga(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=seed, algorithm_parameters=algorithm_param)
  model.run('GA_data.dat', data=pop_data)

  data = {}
  data['best_sol'] = model.best_variable
  data['best_fit'] = model.best_function
  data['best_iter'] = model.iterate
  data['max_iter'] = model.iterate
  return data

def imprvGa(obj_func, pop_data=None, seed = 777):
  algorithm_param = {'max_num_iteration': max_iter,\
                   'population_size':population_size,\
                   'mutation_probability':0.4,\
                   'elit_ratio': 0.1,\
                   'crossover_probability': .7,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
  model=gaOptd(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=seed, algorithm_parameters=algorithm_param)
  model.run('GA_data.dat', data=pop_data)

  data = {}
  data['best_sol'] = model.best_variable
  data['best_fit'] = model.best_function
  data['best_iter'] = model.iterate
  data['max_iter'] = model.iterate
  # data['best_sol'] = best_pos1
  # data['best_sol'] = best_pos1
  return data

def pso(obj_func, pop_data=None, seed = 777):
  lb = varbound[:, 0].tolist()
  ub = varbound[:, 1].tolist()

  verbose = False
  
  model = BasePSO(obj_func, lb, ub, verbose, max_iter, population_size, random_seed=seed)  # Remember the keyword "problem_size"
  best_sol, best_fit, list_loss1 = model.train('pso_data.dat', data=pop_data)
  data = {}
  data['best_sol'] = best_sol
  data['best_fit'] = best_fit
  data['best_iter'] = max_iter
  data['max_iter'] = max_iter
  # data['best_sol'] = best_pos1
  # data['best_sol'] = best_pos1
  return data

def bbo(obj_func, pop_data=None, seed = 777):
  lb = varbound[:, 0].tolist()
  ub = varbound[:, 1].tolist()

  verbose = False

  model = BaseBBO(obj_func, lb, ub, verbose, max_iter, population_size, random_seed=seed)  # Remember the keyword "problem_size"
  best_sol, best_fit, list_loss1 = model.train('bbo_data.dat', data=pop_data)
  data = {}
  data['best_sol'] = best_sol
  data['best_fit'] = best_fit
  data['best_iter'] = max_iter
  data['max_iter'] = max_iter
  # data['best_sol'] = best_pos1
  # data['best_sol'] = best_pos1
  return data

for func in two_args:
  class_ = getattr(bf, func)
  instance = class_(dimension)
  count = 1
  for seed in seeds:
    np.random.seed(seed)
    obj_func = instance.get_func_val
    pop_data = createPop(dimension, population_size)

    print("NormGA")
    res = normGa(obj_func, pop_data)

    result = {}
    result["function"] = func
    result["run"] = count
    result["seed"] = seed
    result["algo"] = "NormGA"
    result["func_val"] = res["best_fit"]
    result["pop_size"] = population_size
    result["max_iter"] = max_iter
    result["best_iter"] = max_iter

    results.append(result)

    print("imprvGA")
    res = imprvGa(obj_func, pop_data)

    result = {}
    result["function"] = func
    result["run"] = count
    result["seed"] = seed
    result["algo"] = "imprvGA"
    result["func_val"] = res["best_fit"]
    result["pop_size"] = population_size
    result["max_iter"] = max_iter
    result["best_iter"] = max_iter

    results.append(result)

    print("PSO")
    res = pso(obj_func, pop_data)

    result = {}
    result["function"] = func
    result["run"] = count
    result["seed"] = seed
    result["algo"] = "PSO"
    result["func_val"] = res["best_fit"]
    result["pop_size"] = population_size
    result["max_iter"] = max_iter
    result["best_iter"] = max_iter

    results.append(result)

    print("BBO")
    res = bbo(obj_func, pop_data)

    result = {}
    result["function"] = func
    result["run"] = count
    result["seed"] = seed
    result["algo"] = "BBO"
    result["func_val"] = res["best_fit"]
    result["pop_size"] = population_size
    result["max_iter"] = max_iter
    result["best_iter"] = max_iter

    results.append(result)


    

    count += 1

pd.DataFrame.from_dict(results, orient='columns').to_csv(f'{filename}.csv')


