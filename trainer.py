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
    "<dev>-<experiment>-<counter>",
    # "tbishnoi-colabtraintest-0",
    # any other args you want
)



def select_model(configs)-> torch.nn.Module:
  if configs['rule_select'] == 'hebb':
    from rules.Hebbian import HebbianNetwork
    model = HebbianNetwork(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      clamp_output=configs['clamp_output'],
      bias=configs['bias'],
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


def trainer(experiment_name, configs):
  # initialize run
  run = wandb.init(
    entity=entity,
    project=project,
    name=experiment_name,
    config=configs,
    resume="never",
  )

  optimizer = torch.optim.SGD(
    model.parameters(),
    lr=configs['lr'], 
    momentum=configs['momentum'], 
    weight_decay=configs['weight_decay'], 
    nesterov=configs['nesterov'],
  )
  loss_crossentropy = torch.nn.CrossEntropyLoss()

  # training loop
  for epoch in tqdm(range(configs['epochs'])):
    for idx_batch, (inputs, targets) in enumerate(train_loader):
      inputs = inputs.view(configs['batch_size'], -1)

      inputs, targets = Variable(inputs), Variable(targets)
      outputs = model(inputs.to(device))
      loss = loss_crossentropy(outputs, targets.to(device))

      model.zero_grad()
      loss.backward()
      optimizer.step()

      # training history
      run.log({"loss": loss.item(), "epoch": epoch})


  # save model info
  os.makedirs("models", exist_ok=True)
  torch.save(model.state_dict(), "models/model.pth")
  wandb.save("models/model.pth")
  
  # important: finish experiment
  run.finish()

trainer(experiment_name, configs)

