from IPython.display import Image, SVG, display
import os
from pathlib import Path

import random
from tqdm import tqdm
import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch
import torchvision
import contextlib
import io
import wandb

def train_model(
  model, 
  train_loader, 
  valid_loader, 
  optimizer, 
  experiment_name=None, 
  configs=None, 
  log_results=False, 
  device='cpu',
):
  MLP = model
  num_epochs = configs['epochs']
  perturbation_update = configs['rule_select'] in ['wp', 'np']

  if log_results:
    # initialize run
    run = wandb.init(
      entity=configs['entity'],
      project=configs['project'],
      name=experiment_name,
      config=configs,
      resume="never",
    )
  else:
    print('WARNING: training without logging to wandb')
    run = None

  results_dict = {
      "avg_train_losses": list(),
      "avg_valid_losses": list(),
      "avg_train_accuracies": list(),
      "avg_valid_accuracies": list(),
      "weight_stats": list(), #stat
      "activation_stats": list(), #stat
  }

  # run training
  for epoch in tqdm(range(num_epochs)):
    no_train = True if epoch == 0 else False # to get a baseline
    
    # train the epoch
    latest_epoch_results_dict = train_epoch(
      MLP, 
      train_loader, 
      valid_loader, 
      optimizer=optimizer, 
      no_train=no_train, 
      perturbation_update=perturbation_update, 
      run=run, 
      epoch=epoch,
      device=device,
    )

    # store results of the epoch
    for key, result in latest_epoch_results_dict.items():
      if key in results_dict.keys() and isinstance(results_dict[key], list):
        results_dict[key].append(latest_epoch_results_dict[key])
      else:
        results_dict[key] = result # copy latest

    # wandb logging for each epoch
    if log_results:
      
      per_epoch_info = dict(
        dict(epoch=epoch), 
        **latest_epoch_results_dict
      )
      run.log(per_epoch_info)
  
  # save model weights
  if log_results:
    model_filepath = configs['model_filepath']
    dirpath = os.path.dirname(model_filepath)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(model.state_dict(), model_filepath)
    wandb.save(model_filepath)

    run.finish()
  
  return results_dict


def train_epoch(
  MLP, 
  train_loader, 
  valid_loader, 
  optimizer, 
  no_train=False, 
  perturbation_update=False, 
  run=None, 
  epoch=None, 
  device=None,
):

  criterion = torch.nn.NLLLoss()
  #Negative Log-Likelihood Loss, torch.log(softmax(output))
  # the NLLLoss function encourages the model to assign higher probabilities to the correct classes and lower probabilities to the incorrect classes

  # if perturbation_update:
  #   for params in MLP.parameters():
  #     params.requires_grad = False

  epoch_results_dict = dict() #stat
  
  # for dataset in ["train", "valid"]:
  #   for sub_str in ["correct_by_class", "seen_by_class"]:
  #     epoch_results_dict[f"{dataset}_{sub_str}"] = {
  #       i:0 for i in range(MLP.num_outputs)
  #     }

  MLP.train()
  train_losses, train_acc = list(), list() # for each epoch
  
  # entire history of weights and activations
  all_weight_stats = []
  all_activation_stats = []
  all_loss_stats = []
  
  for batch_idx, (X, y) in enumerate(train_loader): #stat
    X = X.to(device)
    y = y.to(device)
    
    # unperturbed pass
    if perturbation_update:
      y_pred = MLP(X, y=y)
      loss = criterion(torch.log(y_pred), y) # unperturbed loss
      acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
      train_losses.append(loss.item() * len(y))
      train_acc.append(acc.item() * len(y))

      # # how did the model do w.r.t each class in this batch?
      # update_results_by_class_in_place(
      #   y, y_pred.detach(), epoch_results_dict, dataset="train",
      #   num_classes=MLP.num_outputs
      # )

      w_stats, a_stats = collect_statistics(MLP, X)
      all_weight_stats.append(w_stats)
      all_activation_stats.append(a_stats)
      all_loss_stats.append(loss)

      # perturbed pass
      y_pred_p, perturbs, activations = MLP.forward_p(X, y=y)
      loss_p = criterion(torch.log(y_pred_p), y)

      optimizer.zero_grad()
      if not no_train:
        MLP.accumulate_grads(X, perturbs, activations, loss, loss_p)
        optimizer.step()
        
    else: # update model as usual
      y_pred = MLP(X, y=y)

      loss = criterion(torch.log(y_pred), y) # loss function
      acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
      train_losses.append(loss.item() * len(y))
      train_acc.append(acc.item() * len(y))

      # #  how did the model do w.r.t each class in this batch?
      # update_results_by_class_in_place(
      #   y, y_pred.detach(), epoch_results_dict, dataset="train",
      #   num_classes=MLP.num_outputs
      # )

      w_stats, a_stats = collect_statistics(MLP, X)
      all_weight_stats.append(w_stats)
      all_activation_stats.append(a_stats)
      all_loss_stats.append(loss)

      optimizer.zero_grad()
      if not no_train:
        loss.backward()
        optimizer.step()

    # logging at end of batch 
    if run is not None:
        run.log({"train_loss": loss.item(), "epoch": epoch,})

  # evaluation of model after end of batch
  MLP.eval()
  valid_losses, valid_acc = list(), list()
  with torch.no_grad():
    for X, y in valid_loader:
      X = X.to(device)
      y = y.to(device)

      y_pred = MLP(X)
      loss = criterion(torch.log(y_pred), y)
      acc = (torch.argmax(y_pred, axis=1) == y).sum() / len(y)
      valid_losses.append(loss.item() * len(y))
      valid_acc.append(acc.item() * len(y))

      # stat
      # update_results_by_class_in_place(
      #   y, y_pred.detach(), 
      #   epoch_results_dict, 
      #   dataset="valid",
      # )

  num_items = len(train_loader.dataset)
  epoch_results_dict["avg_train_losses"] = np.sum(train_losses) / num_items
  epoch_results_dict["avg_train_accuracies"] = np.sum(train_acc) / num_items * 100
  epoch_results_dict['weight_stats'] = compute_aggregate_stats(all_weight_stats)
  # epoch_results_dict['activation_stats'] = compute_aggregate_stats(all_activation_stats)

  num_items = len(valid_loader.dataset)
  epoch_results_dict["avg_valid_losses"] = np.sum(valid_losses) / num_items
  epoch_results_dict["avg_valid_accuracies"] = np.sum(valid_acc) / num_items * 100

  return epoch_results_dict

# # stat related functions
# def update_results_by_class_in_place(
#   _y, _y_pred, 
#   result_dict, 
#   dataset="training_classes",
#   num_classes=10, #TODO: static why?
# ):
#   """
#   Update training and validation accuracy 
#   e.g., train accuracy = train_correct_by_class/train_seen_by_class
#   result_dict = epoch_results_dict
#   dataset="train" or "valid"
#   """
#   y = _y.cpu()
#   y_pred = _y_pred.cpu()
  
#   y_pred = np.argmax(y_pred, axis=1)
#   if len(y) != len(y_pred):
#     raise RuntimeError("Number of predictions does not match number of targets.")

#   for i in result_dict[f"{dataset}_seen_by_class"].keys():
#     idxs = np.where(y == int(i))[0]
#     result_dict[f"{dataset}_seen_by_class"][int(i)] += len(idxs)

#     num_correct = int(sum(y[idxs] == y_pred[idxs]))
#     result_dict[f"{dataset}_correct_by_class"][int(i)] += num_correct

def collect_statistics(model, data):
  """
  Collect weight_stats and activation_stats
  """
  # Collect weight statistics
  weight_stats = {
      'lin1': model.lin1.weight.data.clone().cpu(),
      'lin2': model.lin2.weight.data.clone().cpu(),
  }

  # print(weight_stats['lin1'])

  # Collect activation statistics
  model.eval()
  with torch.no_grad():
    h = model(data)

  activation_stats = {
      'output': h.clone().cpu(),
  }

  # We can't collect layer-wise activity changes without modifying the model
  # So we'll skip this for now

  return weight_stats, activation_stats

def compute_aggregate_stats(stats_list):
  """
  calculate mean, std, min, max, median for stats in stats_list
  stats_list = all_weight_stats or all_activation_stats
  """
  all_stats = []
  for stats in stats_list:
    flat_stats = torch.cat([s.flatten() for s in stats.values()])
    all_stats.append(flat_stats.numpy())

  # Ensure all elements have the same shape before creating the array  
  try:
    all_stats = np.vstack(all_stats)
  except:
    print("im broken")
    import sys
    sys.exit(0)
  # all_stats = np.array(all_stats)

  return {
    'mean': np.mean(all_stats, axis=1),
    'std': np.std(all_stats, axis=1),
    'min': np.min(all_stats, axis=1),
    'max': np.max(all_stats, axis=1),
    'median': np.median(all_stats, axis=1)
  }


# select model
def select_model(configs, device)-> torch.nn.Module:
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

  elif configs['rule_select'] == 'wp': # change to accepting kwargs
    from rules.WP import WeightPerturbMLP
    model = WeightPerturbMLP(
      num_inputs=configs['num_inputs'],
      num_hidden=configs['num_hidden'],
      num_outputs=configs['num_outputs'],
      bias=configs['bias'],
      activation_type=configs['activation_type'],
      sigma=configs['sigma'],
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
