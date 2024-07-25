import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime, timezone

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

entity="adorable-lantanas"
project="learning-rules"
log_results=True

configs = dict(
  entity=entity,
  project=project,
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

  # hebbian-specific
  clamp_output=True,
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

  from utils.training_utils import select_model, train_model
  from rules.classes.BasicOptim import BasicOptimizer

  def generate_experiment_name( # write this function to generate your custom name
    name,
  ):
    name = f"tbishnoi-test-{name}"
    return name

  for rule in [
    'backprop', 
    # 'hebb',
    # 'wp',
    # 'np',
  ]:
    
    configs['rule_select'] = rule
    experiment_name = generate_experiment_name(rule)

    # create and save model name
    model_filepath = f"models/model-{datetime.now(timezone.utc).strftime('%y%m%d-%H%M%S')}.pth"
    configs['model_filepath'] = model_filepath

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
