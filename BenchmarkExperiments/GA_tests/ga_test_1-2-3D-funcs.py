import numpy as np
import math
# import test
import time

# test.load_dataset()
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
from geneticalgorithmOptd import geneticalgorithmOptd as ga1
# from geneticalgorithmOptd import geneticalgorithmOptd as gaOptd
import benchmark_func as bf

population_size = 100

algorithm_param = {'max_num_iteration': 100,\
                   'population_size':population_size,\
                   'mutation_probability':0.4,\
                   'elit_ratio': 0.1,\
                   'crossover_probability': .9,\
                   'parents_portion': 0.667,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

def f1(X):

    dim=len(X)
    

    t1=0
    t2=0
    for i in range (0,dim):
        t1+=X[i]**2
        t2+=math.cos(2*math.pi*X[i])     

    OF=20+math.e-20*math.exp((t1/dim)*-0.2)-math.exp(t2/dim)

    return OF

def run_train_model(agent):
    # print(agent)
    return sum(agent**4)
# np.random.seed(427)
dimension = 1

test_objs = []

# test_objs.append(bf.Zakharov(dimension))
test_objs.append(bf.Ackley(dimension))
test_objs.append(bf.Sphere(dimension))
test_objs.append(bf.Rosenbrock(dimension))
test_objs.append(bf.Michalewicz(dimension))

# print(test_objs[0])
# exit()
# test = bf.Zakharov(dimension)
# test = bf.Ackley(dimension)
# test = bf.Rastrigin(dimension)
# test = bf.Booth()
# test = bf.Eggholder()
# test = bf.StyblinskiTang(dimension)
# test = bf.WeightedSphere(dimension)

# obj_func = test.get_func_val



# varbound=np.array([[0, 4]]*2)
# print(varbound)
min = -100
max = 100
varbound=np.array([[min, max]]*dimension)
print("len varbound: ", len(varbound))
print(" varbound: ", (varbound))

# for i in range(0, 1):

results = {}
results["name"] = []
results["best_function"] = []
# results["model"] = []
var=np.zeros(dimension) 

pop=np.array([np.zeros(dimension)]*population_size)
for p in range(0, population_size):
    for i in range(0, dimension):
        var[i]=np.random.randint(varbound[i][0],\
                varbound[i][1]+1) 
    pop[p] = var
data = {}
data['pop'] = pop.copy()
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
# print(data['pop'])

# print("dfkdfjkdjfk: ", pop) 

# model=ga(function=obj_func,dimension=2,variable_type='real',variable_boundaries=varbound, random_seed=777, algorithm_parameters=algorithm_param)
# model.run('GA_data.dat')
rand_var = 80

# before = time.time()
# model=ga(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=rand_var, algorithm_parameters=algorithm_param, convergence_curve=False)
# model.run('GA_data.dat', data=data)
# total_time = time.time() - before

# print("Time to run model: ", total_time)


# before = time.time()
# model1=ga1(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=rand_var, algorithm_parameters=algorithm_param, convergence_curve=False)
# model1.run('GA_data.dat', data=data)
# total_time = time.time() - before

# print("Time to run model1: ", total_time)

import pandas as pd

for i in range(len(test_objs)):
    obj_func = test_objs[i].get_func_val

    # before = time.time()
    model=ga1(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=rand_var, algorithm_parameters=algorithm_param, convergence_curve=False)
    model.run('GA_data.dat', data=data)

    results["name"].append(test_objs[i].func_name)
    results["best_function"].append(model.best_function)

pd.DataFrame.from_dict(results, orient='columns').to_csv(f'test_{dimension}d_result.csv')

# print("best_function: ", model2.best_function)
# print("best_variable: ", model2.best_variable)
# total_time = time.time() - before

# print("Time to run modelOpt: ", total_time)


# for i in range(0, 5):
#     rand_var = np.random.randint(0, 1000)
#     before = time.time()
#     model2=gaOptd(function=obj_func,dimension=dimension,variable_type='real',variable_boundaries=varbound, random_seed=rand_var, algorithm_parameters=algorithm_param, convergence_curve=False)
#     model2.run('GA_data.dat', data=data)
#     total_time = time.time() - before

# print("Time to run modelOpt: ", total_time)

# results["random"].append(rand_var)
# results["model"].append(model.best_function)
# results["model1"].append(model1.best_function)

import pandas as pd

results_df = pd.DataFrame(results)
# print(test)

# print(results_df)

# print(obj_func([-5, 5]))

# test.plot()