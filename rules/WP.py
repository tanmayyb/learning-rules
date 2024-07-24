import torch
from .classes.MLP import MultiLayerPerceptron

class WeightPerturbMLP(MultiLayerPerceptron):
  """
  A multilayer perceptron that is capable of learning through weight perturbation
  """

  def __init__(self, sigma=1e-4, **kwargs):
    """
    NOTE: working hyperparameters seem to be lr=1e-4, sigma=1e-4, activation='relu'
    """

    self.sigma = sigma
    super().__init__(**kwargs)

  def forward(self, X, y=None):
    """
    Runs a forward pass through the network.

    Arguments:
    - X (torch.Tensor): Batch of input images.
    - y (torch.Tensor, optional): Batch of targets, stored for the backward
      pass to compute the gradients for the last layer.

    Returns:
    - y_pred (torch.Tensor): Predicted targets.
    """

    h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
    y_pred = self.softmax(self.lin2(h)) #probability distribution

    return y_pred

  def forward_p(self, X, y=None):
    perturbs = []

    # perturb params
    with torch.no_grad():
      for params in self.parameters():
          perturb = self.sigma * torch.randn(params.shape, device=params.device)
          params.add_(perturb)
          perturbs.append(perturb)

    y_pred_p = self.forward(X)

    # unperturb params
    with torch.no_grad():
      for i, params in enumerate(self.parameters()):
          params.sub_(perturbs[i])

    return y_pred_p, perturbs, None

  def grad_w(self, perturb, loss, loss_p):
    return (loss_p - loss)*perturb/(self.sigma**2)

  def accumulate_grads(self, X, perturbs, activations, loss, loss_p):
    for i, params in enumerate(self.parameters()):
          dw = self.grad_w(perturbs[i], loss, loss_p)

          if params.grad is None:
            params.grad = torch.zeros_like(params)

          params.grad = params.grad.add_(dw)
