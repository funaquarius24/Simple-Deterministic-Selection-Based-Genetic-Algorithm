#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


dimension = 2
population_size = 100
max_iter = 50
num_runs = 5
seeds = np.random.randint(0, 1000, num_runs)
varbound=np.array([[-100, 100]]*dimension)


# In[3]:


timestr = time.strftime("%Y%m%d-%H%M%S")
filename = f'results/high_pop/Run_Results-{timestr}'
pivot_filename = "resultFile_d{}_pop{}_iter{}_runs{}--{}.xlsx"    .format(dimension, population_size, max_iter, num_runs, timestr)

one_args = getattr(bf, "__oneArgument__")
two_args = getattr(bf, "__twoArgument__")


# In[4]:


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


# In[5]:


def createPop(dimension=2, population_size=15):
  var=np.zeros(dimension) 

  pop=np.array([np.zeros(dimension)]*population_size)
  for p in range(0, population_size):
      for i in range(0, dimension):
          var[i]=np.random.randint(varbound[i][0],                  varbound[i][1]+1) 
      pop[p] = var
  data = {}
  data['pop'] = pop.copy()

  # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
  # print(data['pop'])

  return data


# In[6]:


def normGa(obj_func, pop_data=None, seed = 777, terminal_value=None):
  algorithm_param = {'max_num_iteration': max_iter,                   'population_size':population_size,                   'mutation_probability':0.2,                   'elit_ratio': 0.1,                   'crossover_probability': .7,                   'parents_portion': 0.3,                   'crossover_type':'uniform',                   'max_iteration_without_improv':None,                   'terminal_value': terminal_value}
  model=ga(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=seed, algorithm_parameters=algorithm_param)
  model.run('GA_data.dat', data=pop_data)

  data = {}
  data['best_sol'] = model.best_variable
  data['best_fit'] = model.best_function
  data['best_iter'] = model.terminal_value_iteration
  # data['max_iter'] = model.iterate
  return data


# In[7]:


def imprvGa(obj_func, pop_data=None, seed = 777, terminal_value=None):
  algorithm_param = {'max_num_iteration': max_iter,                   'population_size':population_size,                   'mutation_probability':0.4,                   'elit_ratio': 0.1,                   'crossover_probability': .7,                   'parents_portion': 0.3,                   'crossover_type':'uniform',                   'max_iteration_without_improv':None,                   'terminal_value': terminal_value}
  model=gaOptd(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=seed, algorithm_parameters=algorithm_param)
  model.run('GA_data.dat', data=pop_data)

  data = {}
  data['best_sol'] = model.best_variable
  data['best_fit'] = model.best_function
  data['best_iter'] = model.terminal_value_iteration
  # data['max_iter'] = model.iterate
  # data['best_sol'] = best_pos1
  # data['best_sol'] = best_pos1
  return data


# In[8]:


def pso(obj_func, pop_data=None, seed = 777, terminal_value=None):
  lb = varbound[:, 0].tolist()
  ub = varbound[:, 1].tolist()

  verbose = False
  
  model = BasePSO(obj_func, lb, ub, verbose, max_iter, population_size, random_seed=seed, terminal_value=terminal_value)  # Remember the keyword "problem_size"
  best_sol, best_fit, list_loss1 = model.train('pso_data.dat', data=pop_data)
  data = {}
  data['best_sol'] = best_sol
  data['best_fit'] = best_fit
  data['best_iter'] = model.terminal_value_iteration
  # data['max_iter'] = max_iter
  # data['best_sol'] = best_pos1
  # data['best_sol'] = best_pos1
  return data


# In[9]:


def bbo(obj_func, pop_data=None, seed = 777, terminal_value=None):
  lb = varbound[:, 0].tolist()
  ub = varbound[:, 1].tolist()

  verbose = False

  model = BaseBBO(obj_func, lb, ub, verbose, max_iter, population_size, random_seed=seed, terminal_value=terminal_value)  # Remember the keyword "problem_size"
  best_sol, best_fit, list_loss1 = model.train('bbo_data.dat', data=pop_data)
  data = {}
  data['best_sol'] = best_sol
  data['best_fit'] = best_fit
  data['best_iter'] = model.terminal_value_iteration
  # data['max_iter'] = max_iter
  # data['best_sol'] = best_pos1
  # data['best_sol'] = best_pos1
  return data


# In[10]:


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
    # result["max_iter"] = max_iter
    result["best_iter"] = res['best_iter']

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
    # result["max_iter"] = max_iter
    result["best_iter"] = res['best_iter']

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
    # result["max_iter"] = max_iter
    result["best_iter"] = res['best_iter']

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
    # result["max_iter"] = max_iter
    result["best_iter"] = res['best_iter']

    results.append(result)

    

    count += 1


# In[11]:


pd.DataFrame.from_dict(results, orient='columns').to_csv(f'{filename}.csv')


# In[12]:


results_df = pd.DataFrame(results)


# In[13]:


algos = results_df.algo.unique()
functions = results_df.function.unique()


# In[ ]:





# In[14]:


describe_result = {}
describe_results = []
describe_results_df = None

count = 0
for i in range(len(algos)):
    for j in range(len(functions)):
        describe_result = {}
        new_df = results_df.loc[(results_df['function'] == functions[j]) & (results_df['algo'] == algos[i])]
        describe = new_df.describe().loc[['mean','std'], 'func_val'].tolist()
        describe_result['function'] = functions[j]
        describe_result['algo'] = algos[i]
        describe_result['mean'] = describe[0]
        describe_result['std'] = describe[1]
        describe_results.append(describe_result.copy())
    


# In[15]:


func_global_opts = {}
for func in two_args:
    
    class_ = getattr(bf, func)
    instance = class_(dimension)
    func_global_opts[instance.func_name] = instance.global_optimum_solution


# In[16]:


for data in describe_results:
    data['global_val'] = func_global_opts[data['function']]


# In[17]:


describe_results_df = pd.DataFrame(describe_results)


# In[18]:


describe_results_df.head(50)


# In[19]:


pivot_table_df = pd.pivot_table(describe_results_df,

        index=['function'],

        columns=['algo'],

        values=['mean', 'std'],

        aggfunc=sum)
pivot_table_df


# In[20]:


func_global_opts


# In[21]:


pivot_table_df.keys()


# In[22]:


pivot_table_df.to_excel(pivot_filename)


# In[23]:


aa = "dsj{}-{}={}".format(2, 3, 4)


# In[24]:


aa


# In[25]:


# describe_results_df.to_excel("globals.xlsx")


# In[ ]:





# In[ ]:




