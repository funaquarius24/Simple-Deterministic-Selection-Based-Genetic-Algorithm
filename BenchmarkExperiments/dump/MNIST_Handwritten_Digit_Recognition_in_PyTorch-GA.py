#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

import torchvision
import os
from pathlib import Path
import time

# Import libraries
import os
import time
import random
import numpy as np
from numpy import argmax
import pandas as pd
import json
from collections import OrderedDict, namedtuple
from itertools import product
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# In[2]:


try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False


# In[3]:


if IN_COLAB:
    get_ipython().system('pip install mealpy')
    get_ipython().system('pip install geneticalgorithm')
from mealpy.evolutionary_based.GA import BaseGA
import math
from geneticalgorithm import geneticalgorithm as ga


# In[4]:


# Enable GPU processing
if IN_COLAB:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Device type: {device}')


# In[5]:


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

results_folder = 'results'
Path("results").mkdir(parents=True, exist_ok=True)


# In[6]:


# Set random seed
seed = 777
torch.manual_seed(seed)
model = None


# In[7]:


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

data = {
    'train': 
    torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    'val': 
    torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
}


# In[8]:


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# In[9]:


example_data.shape


# In[10]:


import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])


# In[11]:


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[12]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# In[13]:


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

# If we were using a GPU for training, we should have also sent 
# the network parameters to the GPU using e.g. network.cuda(). 
# It is important to transfer the network's parameters to the appropriate 
# device before passing them to the optimizer, otherwise the optimizer will 
# not be able to keep track of them in the right way.


# In[14]:


print(network)


# In[15]:


# Define a class to build run execution sets based on a dictionary of hyperparameters
class RunBuilder():
  @staticmethod
  def get_runs(params):
    Run = namedtuple('Run', params.keys())
    runs = []
    for v in product(*params.values()):
      runs.append(Run(*v))
    return runs


# In[16]:


# Create a class to manage the training / hyperparameter runs
class RunManager():
  def __init__(self):
    self.epoch_count = 0
    self.train_loss = 0
    self.train_num_correct = 0
    self.val_loss = 0
    self.val_num_correct = 0

    self.run_params = None
    self.run_count = 0
    self.run_data = []

    self.model = None
    self.train_loader = None
    self.val_loader = None
    self.tb = None
    
    #---
    self.results = None

  def begin_run(self, run, model, train_loader, val_loader):
    self.run_params = run
    self.run_count += 1
    self.model = model.to(device)
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.tb = SummaryWriter(log_dir='/runs', max_queue=20, comment=f'-{run}')
    images, labels = next(iter(self.train_loader))
    images, labels = images.to(device), labels.to(device)
    self.tb.add_graph(self.model, images)
    

  def end_run(self):
    self.tb.close()
    self.epoch_count = 0

  def begin_epoch(self):
    self.epoch_count += 1
    self.train_loss = 0
    self.train_num_correct = 0
    self.val_loss = 0
    self.val_num_correct = 0

  def end_epoch(self):
    train_loss = self.train_loss / len(self.train_loader.dataset)
    train_accuracy = self.train_num_correct / len(self.train_loader.dataset)
    val_loss = self.val_loss / len(self.val_loader.dataset)
    val_accuracy = self.val_num_correct / len(self.val_loader.dataset)

    self.tb.add_scalar('Train Loss', train_loss, self.epoch_count)
    self.tb.add_scalar('Train Accuracy', train_accuracy, self.epoch_count)
    self.tb.add_scalar('Val Loss', val_loss, self.epoch_count)
    self.tb.add_scalar('Val Accuracy', val_accuracy, self.epoch_count)

    for name, param in self.model.named_parameters():
      self.tb.add_histogram(name, param, self.epoch_count)
      #self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

    print(f'Epoch: {self.epoch_count}, Train Loss: {train_loss:.3f}, Train Acc: {train_accuracy:.3f}')
    print(f'Epoch: {self.epoch_count}, Valid Loss: {val_loss:.3f}, Valid Acc: {val_accuracy:.3f}')
    
    results = OrderedDict()
    results['run'] = self.run_count
    results['epoch'] = self.epoch_count
    results['train loss'] = train_loss
    results['train acc'] = train_accuracy
    results['valid loss'] = val_loss
    results['valid acc'] = val_accuracy
    
    # ---
    self.results = results

    for k, v in self.run_params.items():
      results[k] = v

    self.run_data.append(results)

  def track_loss(self, loss, mode):
    if mode == 'train':
      self.train_loss += loss.item() * self.train_loader.batch_size
    elif mode == 'val':
      self.val_loss += loss.item() * self.val_loader.batch_size

  def track_num_correct(self, preds, labels, mode):
    if mode == 'train':
      self.train_num_correct += preds.argmax(dim=1).eq(labels).sum().item()
    elif mode == 'val':
      self.val_num_correct += preds.argmax(dim=1).eq(labels).sum().item()

  def save_output(self, filename):
    if filename:
      filename = filename
      pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')
      
      # with open(f'{filename}.json', 'w', encoding='utf-8') as f:
      #   json.dump(self.run_data, f, ensure_ascii=False, indent=4)

      print('Results saved to disk')

    return pd.DataFrame.from_dict(self.run_data, orient='columns')


# In[17]:


# Define training loop execution
def execution_loop(filename, model, args): # args is given by the optimizer
    agent = {}
    agents = []
    power = args[0].astype(int)
    num = args[0] - power 
    tmp = num * 10.0**-power
    agent['lr'] = round(tmp, 6)
    agent['batch_size'] = int(args[1])
    agents.append(agent)
    m = RunManager()
    for run in agents: # this should be one. a particle in pso

        # instantiate the neural network model
        #     model = CNN_model(run.hidden_units, run.dropout, run.num_classes) 
        optimizer = Adam(model.parameters(), lr=run['lr'])

        # Define the data loaders
        dataloaders = {
            'train': DataLoader(data['train'], batch_size=run['batch_size'], shuffle=True, num_workers=1),
            'val': DataLoader(data['val'], batch_size=run['batch_size'], shuffle=False, num_workers=1)
        }

        train_loader = dataloaders['train']
        val_loader = dataloaders['val']  

        print(f'Run Params: {run}')

        m.begin_run(run, model, train_loader, val_loader)
        for epoch in range(params['n_epochs'][0]):
            m.begin_epoch()
            for batch in train_loader:
                with torch.set_grad_enabled(True):
                    # get inputs/targets and move tensors to GPU
                    images, labels = batch[0].to(device), batch[1].to(device)
                    # clear previous gradients
                    optimizer.zero_grad()
                    # make prediction
                    yhat = model(images)
                    # calculate the loss
                    loss = F.nll_loss(yhat, labels)
                    # perform back prop
                    loss.backward()
                    # update model weights
                    optimizer.step()

                    m.track_loss(loss, 'train')
                    m.track_num_correct(yhat, labels, 'train')

            else:
                with torch.no_grad():
                    for batch in val_loader:
                        images, labels = batch[0].to(device), batch[1].to(device)
                        output = model(images)
                        loss = F.nll_loss(output, labels)

                        m.track_loss(loss, 'val')
                        m.track_num_correct(output, labels, 'val')

            m.end_epoch()
    m.end_run()
    return model, m.save_output(filename), m.results['valid acc']


# In[18]:


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


# In[19]:


# Define training run hyperparameters
params = {
    'hidden_units' : [256],
    'dropout' : [0.5],
    'num_classes' : [10],
    'lr' : [0, 1],
    'batch_size' : [20, 2000],
    'n_epochs' : [4]
}

# params = OrderedDict(
#     lr = [0, 1],
#     batch_size = [20 2000],
#     n_epochs = [3]
# )


# In[20]:


def run_train_model(agent):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model, history, valid_acc = execution_loop(f'Run_Results-GA-{timestr}', network, agent)
    print("valid acc: ", valid_acc)
    return valid_acc * -1


# In[ ]:


verbose = True
pop_size = 2    
max_iter = 2 

obj_func = run_train_model

lb = [0, params['batch_size'][0]]
ub = [4, params['batch_size'][1]]

algorithm_param = {'max_num_iteration': 4,                   'population_size':4,                   'mutation_probability':0.06,                   'elit_ratio': 0.0,                   'crossover_probability': 0.3,                   'parents_portion': 0.1,                   'crossover_type':'uniform',                   'max_iteration_without_improv':None}

problem_size = len(ub)

pc = 0.95
pm = 1 - pc

varbound = np.array([ [ lb[0], ub[0] ], [ lb[1], ub[1] ] ] )

model=ga(function=run_train_model,dimension=2,variable_type='real',variable_boundaries=varbound, algorithm_parameters=algorithm_param, function_timeout=2000)
model.run()


# In[ ]:


gff


# In[ ]:


# Function to evaluate the model on the test set
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc


# In[ ]:





# In[ ]:


# Determine model accuracy on the test set
test_dl = DataLoader(data['test'], batch_size=params['batch_size'], shuffle=False, num_workers=1)
test_acc = evaluate_model(test_dl, model)

