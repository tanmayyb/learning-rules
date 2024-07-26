import torch
from .classes.MLP import MultiLayerPerceptron

class NodePerturbMLP(MultiLayerPerceptron):
  """
  A multilayer perceptron that is capable of learning through node perturbation
  """

  def __init__(self, sigma=1e-3, **kwargs):
    """
    Initializes a node perturbation multilayer perceptron object

    NOTE: working hyperparameters seem to be lr=1e-3, sigma=1e-3

    Arguments:
    - clamp_output (bool, optional): if True, outputs are clamped to targets,
      if available, when computing weight updates.
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
    activations = []

    perturb_h = self.sigma * torch.randn((X.shape[0], self.num_hidden), device=X.device)
    h_p_pre = self.lin1(X.reshape(-1, self.num_inputs)) + perturb_h
    h_p = self.activation(h_p_pre)
    perturbs.append(perturb_h), activations.append(h_p_pre)

    perturb_y = self.sigma * torch.randn((X.shape[0], self.num_outputs), device=X.device)
    # TODO: save preactivations or postactivations for final layer?
    y_pred_pre = self.lin2(h_p) + perturb_y
    perturbs.append(perturb_y), activations.append(y_pred_pre)

    y_pred_p = self.softmax(y_pred_pre)

    return y_pred_p, perturbs, activations

  def grad_w(self, input_activation, perturb, loss, loss_p, is_bias):
    if is_bias:
      return (loss_p - loss) * perturb.mean(dim=0) / (self.sigma**2)
    else:
      outer_product = torch.einsum('bi,bj->bij', perturb, input_activation).mean(dim=0) # get outer product
      return (loss_p - loss)*outer_product/(self.sigma**2)

  def accumulate_grads(self, X, perturbs, activations, loss, loss_p):
    N_batch = X.shape[0]
    
    input_activations = [X.reshape(N_batch, -1), *activations]
    for i, params in enumerate(self.parameters()):
        is_bias = params.dim() == 1
        dw = self.grad_w(input_activations[i//2], perturbs[i//2], loss, loss_p, is_bias=is_bias)
        # print(is_bias)
        # print(dw)

        if params.grad is None:
            params.grad = torch.zeros_like(params)

        # Add gradients for optimizer to step params
        params.grad = params.grad.add_(dw)
      
        # Manually step params
        # with torch.no_grad():
        #   params.add_(1e-3*dw)
