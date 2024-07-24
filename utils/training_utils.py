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

def train_model(model, train_loader, valid_loader, optimizer, experiment_name=None, configs=None, log_results=False):
  """
  Train a model for several epochs.

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - num_epochs (int, optional): Number of epochs to train model.

  Returns:
  - results_dict (dict): Dictionary storing results across epochs on training
    and validation data.
  """
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

  results_dict = {
      "avg_train_losses": list(),
      "avg_valid_losses": list(),
      "avg_train_accuracies": list(),
      "avg_valid_accuracies": list(),
  }

  for epoch in tqdm(range(num_epochs)):
    no_train = True if epoch == 0 else False # to get a baseline
    latest_epoch_results_dict = train_epoch(
        MLP, train_loader, valid_loader, optimizer=optimizer, no_train=no_train, perturbation_update=perturbation_update
        )

    for key, result in latest_epoch_results_dict.items():
      if key in results_dict.keys() and isinstance(results_dict[key], list):
        results_dict[key].append(latest_epoch_results_dict[key])
      else:
        results_dict[key] = result # copy latest
    if log_results:
      run.log({"avg_train_losses": results_dict['avg_train_losses'], "epoch": epoch})
  
  if log_results:
    # save model info
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    wandb.save("models/model.pth")
  
    # important: finish experiment
    run.finish()
  
  return results_dict


def train_epoch(MLP, train_loader, valid_loader, optimizer, no_train=False, perturbation_update=False):
  """
  Train a model for one epoch.

  Arguments:
  - MLP (torch model): Model to train.
  - train_loader (torch dataloader): Dataloader to use to train the model.
  - valid_loader (torch dataloader): Dataloader to use to validate the model.
  - optimizer (torch optimizer): Optimizer to use to update the model.
  - no_train (bool, optional): If True, the model is not trained for the
    current epoch. Allows a baseline (chance) performance to be computed in the
    first epoch before training starts.

  Returns:
  - epoch_results_dict (dict): Dictionary storing epoch results on training
    and validation data.
  """

  criterion = torch.nn.NLLLoss()
  #Negative Log-Likelihood Loss, torch.log(softmax(output))
  # the NLLLoss function encourages the model to assign higher probabilities to the correct classes and lower probabilities to the incorrect classes

  # if perturbation_update:
  #   for params in MLP.parameters():
  #     params.requires_grad = False

  epoch_results_dict = dict()
  for dataset in ["train", "valid"]:
    for sub_str in ["correct_by_class", "seen_by_class"]:
      epoch_results_dict[f"{dataset}_{sub_str}"] = {
          i:0 for i in range(MLP.num_outputs)
          }

  MLP.train()
  train_losses, train_acc = list(), list()
  for X, y in train_loader:
    if perturbation_update:
      # Update model using perturbation

      # Unperturbed pass
      y_pred = MLP(X, y=y)
      loss = criterion(torch.log(y_pred), y) # unperturbed loss
      acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
      train_losses.append(loss.item() * len(y))
      train_acc.append(acc.item() * len(y))

      update_results_by_class_in_place(
          y, y_pred.detach(), epoch_results_dict, dataset="train",
          num_classes=MLP.num_outputs
          )

      # Perturbed pass
      y_pred_p, perturbs, activations = MLP.forward_p(X, y=y)
      loss_p = criterion(torch.log(y_pred_p), y)

      optimizer.zero_grad()

      # Calculate gradients

      if not no_train:
        MLP.accumulate_grads(X, perturbs, activations, loss, loss_p)
        #loss.backward()
        optimizer.step()
        #pass
    else:
      # Update model as usual
      y_pred = MLP(X, y=y)

      loss = criterion(torch.log(y_pred), y) # loss function
      acc = (torch.argmax(y_pred.detach(), axis=1) == y).sum() / len(y)
      train_losses.append(loss.item() * len(y))
      train_acc.append(acc.item() * len(y))

      update_results_by_class_in_place(
          y, y_pred.detach(), epoch_results_dict, dataset="train",
          num_classes=MLP.num_outputs
          )
      optimizer.zero_grad()

      if not no_train:
        loss.backward()
        optimizer.step()

  num_items = len(train_loader.dataset)
  epoch_results_dict["avg_train_losses"] = np.sum(train_losses) / num_items
  epoch_results_dict["avg_train_accuracies"] = np.sum(train_acc) / num_items * 100

  MLP.eval()
  valid_losses, valid_acc = list(), list()
  with torch.no_grad():
    for X, y in valid_loader:
      y_pred = MLP(X)
      loss = criterion(torch.log(y_pred), y)
      acc = (torch.argmax(y_pred, axis=1) == y).sum() / len(y)
      valid_losses.append(loss.item() * len(y))
      valid_acc.append(acc.item() * len(y))
      update_results_by_class_in_place(
          y, y_pred.detach(), epoch_results_dict, dataset="valid"
          )

  num_items = len(valid_loader.dataset)
  epoch_results_dict["avg_valid_losses"] = np.sum(valid_losses) / num_items
  epoch_results_dict["avg_valid_accuracies"] = np.sum(valid_acc) / num_items * 100

  return epoch_results_dict

def update_results_by_class_in_place(y, y_pred, result_dict, dataset="train",
                                     num_classes=10):
  """
  Updates results dictionary in place during a training epoch by adding data
  needed to compute the accuracies for each class.

  Arguments:
  - y (torch Tensor): target labels
  - y_pred (torch Tensor): predicted targets
  - result_dict (dict): Dictionary storing epoch results on training
    and validation data.
  - dataset (str, optional): Dataset for which results are being added.
  - num_classes (int, optional): Number of classes.
  """

  correct_by_class = None
  seen_by_class = None

  y_pred = np.argmax(y_pred, axis=1)
  if len(y) != len(y_pred):
    raise RuntimeError("Number of predictions does not match number of targets.")

  for i in result_dict[f"{dataset}_seen_by_class"].keys():
    idxs = np.where(y == int(i))[0]
    result_dict[f"{dataset}_seen_by_class"][int(i)] += len(idxs)

    num_correct = int(sum(y[idxs] == y_pred[idxs]))
    result_dict[f"{dataset}_correct_by_class"][int(i)] += num_correct
