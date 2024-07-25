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
project="bp-vs-wp"
prefix="tbishnoi-testrun"
log_results=True
num_ramdom_seed=1
random_seed=0 # initial random_seed

_configs = dict(
  entity=entity,
  project=project,

  epochs = 5,
  batch_size = 32,
  num_inputs = 784,
  num_hidden = 100,
  num_outputs = 10,
  activation_type = 'relu',

  bias=False,
  momentum=0.9,
  weight_decay=0.000,
  nesterov=True,

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

def main():
  global random_seed
  from utils.data_utils import set_seed, download_mnist

  # set initial random_seed
  set_seed(random_seed)

  #create loaders
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

    # conditionally modify configs
    if rule=='backprop':
      configs['lr'] = 1e-2
    elif rule=='wp':
      configs['lr'] = 1e-4
      configs['sigma'] = 1e-4
    else:
      raise NotImplementedError(f"rule: {rule} outside implementation of training plan!")

    for random_seed in tqdm(range(num_ramdom_seed)):
      # rule
      configs['rule_select'] = rule
      
      # experiment name
      experiment_name = f"{prefix}-{rule}-{random_seed}"
      configs['experiment_name'] = experiment_name

      # create and save model name
      model_filepath = f"models/model-{datetime.now(timezone.utc).strftime('%y%m%d-%H%M%S')}.pth"
      configs['model_filepath'] = model_filepath

      # overwrites the random_seed that was used before
      set_seed(random_seed)  
      configs['random_seed'] = random_seed
      
      # select model
      model = select_model(configs, device)
      optimizer = BasicOptimizer(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])

      print()
      print(f"\texp name:\t{experiment_name}")
      print(f"\ttrain rule:\t{rule}")
      print(f"\tnum epochs:\t{configs['epochs']}")
      print(f"\trandom seed:\t{random_seed}")
      print(f"\tmodel filepath:\t{model_filepath}")
      print()





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
