import torch
import copy
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime, timezone

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

entity="adorable-lantanas"
project="learning-rules"
prefix="tbishnoi-learn"
log_results=True
num_seed=10
seed=0

_configs = dict(
  entity=entity,
  project=project,
  epochs = 20,
  batch_size = 32,
  num_inputs = 784,
  num_hidden = 100,
  num_outputs = 10,

  activation_type = 'relu',

  bias=True,
  # lr=1e-4,
  momentum=0.9,
  # weight_decay=0.001,
  nesterov=True,

  # hebbian-specific
  # clamp_output=True,

)


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   /$$                        /$$           /$$                                     /$$                    
  | $$                       |__/          |__/                                    | $$                    
 /$$$$$$    /$$$$$$  /$$$$$$  /$$ /$$$$$$$  /$$ /$$$$$$$   /$$$$$$         /$$$$$$ | $$  /$$$$$$  /$$$$$$$ 
|_  $$_/   /$$__  $$|____  $$| $$| $$__  $$| $$| $$__  $$ /$$__  $$       /$$__  $$| $$ |____  $$| $$__  $$
  | $$    | $$  \__/ /$$$$$$$| $$| $$  \ $$| $$| $$  \ $$| $$  \ $$      | $$  \ $$| $$  /$$$$$$$| $$  \ $$
  | $$ /$$| $$      /$$__  $$| $$| $$  | $$| $$| $$  | $$| $$  | $$      | $$  | $$| $$ /$$__  $$| $$  | $$
  |  $$$$/| $$     |  $$$$$$$| $$| $$  | $$| $$| $$  | $$|  $$$$$$$      | $$$$$$$/| $$|  $$$$$$$| $$  | $$
   \___/  |__/      \_______/|__/|__/  |__/|__/|__/  |__/ \____  $$      | $$____/ |__/ \_______/|__/  |__/
                                                          /$$  \ $$      | $$                              
                                                         |  $$$$$$/      | $$                              
                                                          \______/       |__/                              
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def generate_experiment_name( # write this function to generate your custom name
  prefix, info,
) -> str:
  return f"{prefix}-{info}"

def main():

  # set initial seed
  set_seed(seed)

  #create loaders
  from utils.data_utils import set_seed, download_mnist
  train_set, valid_set, test_set = download_mnist()
  configs = copy.deepcopy(_configs)
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=configs['batch_size'], shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=configs['batch_size'], shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=configs['batch_size'], shuffle=False)


  from utils.training_utils import select_model, train_model
  from rules.classes.BasicOptim import BasicOptimizer
  for rule in [
    'backprop', 
    # 'hebb',
    'wp',
    # 'np',
  ]:

    # for each rule we copy base config
    configs = copy.deepcopy(_configs)

    if rule=='backprop':
      configs['lr'] = 1e-4


    for seed in tqdm(range(num_seed)):
      # rule
      configs['rule_select'] = rule
      
      # experiment name
      experiment_name = generate_experiment_name(prefix, rule, seed)
      configs['experiment_name'] = experiment_name

      # create and save model name
      model_filepath = f"models/model-{datetime.now(timezone.utc).strftime('%y%m%d-%H%M%S')}.pth"
      configs['model_filepath'] = model_filepath

      # set seed
      set_seed(seed)
      configs['seed'] = seed
      
      # select model
      model = select_model(configs, device)
      optimizer = BasicOptimizer(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])

      print(
        "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        f"rule: {rule}\n"
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
      )

      train_model(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        experiment_name, 
        configs, 
        log_results=log_results, 
        device=device
      )

    del model
    del optimizer


if __name__ == '__main__':
  main()
