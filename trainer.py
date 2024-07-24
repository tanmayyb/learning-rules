import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime, timezone
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# !pip install wandb --quiet
import wandb
wandb.login()


entity="adorable-lantanas"
project="learning-rules"

configs = dict(
  entity=entity,
  project=project,
  rule_select = 'hebb',
  epochs = 2,
  batch_size = 32,
  num_inputs = 784,
  num_hidden = 100,
  num_outputs = 10,
  activation_type = 'relu',
  bias = False,
  lr=1e-4,
  momentum=0.9,
  weight_decay=0.001,
  nesterov=True,

  ############
  # specific #
  ############

  # hebbian
  clamp_output = False,
)


train_loader = DataLoader(
  datasets.MNIST(
    './data', train=True, download=True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          (0.1307,),(0.3081,)
      )
    ])
  ),
  batch_size=configs['batch_size'],
  shuffle=True,
)

test_loader = DataLoader(
  datasets.MNIST(
    './data', train=False, download=True,
    transform=transforms.Compose([
      transforms.ToTensor(),
        transforms.Normalize(
          (0.1307,),(0.3081,)
          )
      ]),
  ),
  batch_size=configs['batch_size'],
  shuffle=True,
)


def generate_experiment_name( # write this function to generate your custom name
    name,
    # ... any args you want
):
  # ... your experiment name generator
  return name

experiment_name = generate_experiment_name(
    # "<dev>-<experiment>-<counter>",
    "tbishnoi-modeltestlocal-0",
    # any other args you want
)

from rules.classes.BasicOptim import BasicOptimizer
from rules.classes.BasicOptim import BasicOptimizer

def select_model(configs)-> torch.nn.Module:
  if configs['rule_select'] == 'backprop':
    from rules.classes.MLP import MultiLayerPerceptron
    model = MultiLayerPerceptron(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      bias=configs['bias'],
      activation_type=configs['activation_type'],
    ).to(device)

  elif configs['rule_select'] == 'hebb':
    from rules.Hebbian import HebbianNetwork
    model = HebbianNetwork(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      clamp_output=configs['clamp_output'],
      bias=configs['bias'],
    ).to(device)

  elif configs['rule_select'] == 'wp':
    from rules.WP import WeightPerturbMLP
    model = WeightPerturbMLP(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      bias=configs['bias'],
      activation_type=configs['activation_type'],
    ).to(device)

  elif configs['rule_select'] == 'np':
    from rules.NP import NodePerturbMLP
    model = NodePerturbMLP(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      bias=configs['bias'],
      activation_type=configs['activation_type'],
    ).to(device)

  elif configs['rule_select'] == 'fa':
    from rules.FA import FeedbackAlignmentPerceptron
    model = FeedbackAlignmentPerceptron(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      bias=configs['bias'],
      activation_type=configs['activation_type'],
    ).to(device)

  else:
      raise NotImplementedError("Selected Rule does not exist!")

  return model

model = select_model(configs)
optimizer = BasicOptimizer(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])



from utils.training_utils import train_model

print(configs)

# TODO: add validation loader
train_model(
  model, 
  train_loader, 
  test_loader, 
  optimizer, 
  experiment_name, 
  configs, 
  log_results=False, 
  device=device
)


