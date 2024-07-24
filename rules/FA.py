import torch
from .classes.MLP import MultiLayerPerceptron


#@Josh Tindall's implementation

class FeedbackAlignmentFunction(torch.autograd.Function):
  """
  Gradient computing function class for Hebbian learning.
  """

  @staticmethod
  def forward(context, input, weight, backwards_weight, bias=None, nonlinearity=None, nonlinearity_deriv=None, target=None):
    """
    Forward pass method for the layer. Computes the output of the layer and
    stores variables needed for the backward pass.

    Arguments:
    - context (torch context): context in which variables can be stored for
      the backward pass.
    - input (torch tensor): input to the layer.
    - weight (torch tensor): layer weights.
    - bias (torch tensor, optional): layer biases.
    - nonlinearity (torch functional, optional): nonlinearity for the layer.
    - target (torch tensor, optional): layer target, if applicable.

    Returns:
    - output (torch tensor): layer output.
    """
    activation_deriv = None

    # compute the output for the layer (linear layer with non-linearity)
    preactivations = input.mm(weight.t())

    if bias is not None:
      preactivations += bias.unsqueeze(0).expand_as(preactivations)

    if nonlinearity_deriv is not None:
      if target is not None:
        activation_deriv = nonlinearity_deriv(preactivations, target)
      else:
        activation_deriv = nonlinearity_deriv(preactivations)

    if nonlinearity is not None:
      output = nonlinearity(preactivations)
    else:
      output = preactivations

    # store variables in the context for the backward pass
    context.save_for_backward(input, weight, backwards_weight, bias, activation_deriv, target)

    return output

  @staticmethod
  def backward(context, grad_output=None):
    """
    Backward pass method for the layer. Computes and returns the gradients for
    all variables passed to forward (returning None if not applicable).

    Arguments:
    - context (torch context): context in which variables can be stored for
      the backward pass.
    - input (torch tensor): input to the layer.
    - weight (torch tensor): layer weights.
    - bias (torch tensor, optional): layer biases.
    - nonlinearity (torch functional, optional): nonlinearity for the layer.
    - target (torch tensor, optional): layer target, if applicable.

    Returns:
    - grad_input (None): gradients for the input (None, since gradients are not
      backpropagated in Hebbian learning).
    - grad_weight (torch tensor): gradients for the weights.
    - grad_bias (torch tensor or None): gradients for the biases, if they aren't
      None.
    - grad_nonlinearity (None): gradients for the nonlinearity (None, since
      gradients do not apply to the non-linearities).
    - grad_target (None): gradients for the targets (None, since
      gradients do not apply to the targets).
    """

    input, weight, backwards_weight, bias, activation_deriv, target = context.saved_tensors
    grad_input = None
    grad_weight = None
    grad_bias = None
    grad_nonlinearity = None
    grad_target = None

    grad_backwards_weight = None
    grad_nonlinearity_deriv = None

    # Calculate gradient with respect to inputs (for passing error signals to upstream layers)
    input_needs_grad = context.needs_input_grad[0]
    if input_needs_grad:
      grad_input = (grad_output * activation_deriv).mm(backwards_weight.t()) #np.dot(backwards_weight, grad_output * activation_deriv)

    # Calculate gradient with respect to weights
    weight_needs_grad = context.needs_input_grad[1]
    if weight_needs_grad:
      grad_weight = (input.t()).mm(grad_output * activation_deriv) #np.dot(grad_output * activation_deriv, input.transpose())

      grad_weight = grad_weight / len(input) # average across batch

    # Calculate gradient with respect to biases
    if bias is not None:
      bias_needs_grad = context.needs_input_grad[2]
      if bias_needs_grad:
        grad_bias = (grad_output * activation_deriv).sum(dim=0) / len(input)

    return grad_input, grad_weight.t(), grad_backwards_weight, grad_bias, grad_nonlinearity, grad_nonlinearity_deriv, grad_target


class FeedbackAlignmentPerceptron(MultiLayerPerceptron):
  """
  Hebbian multilayer perceptron with one hidden layer.
  """

  def __init__(self, **kwargs):
    """
    Initializes a Hebbian multilayer perceptron object

    Arguments:
    - clamp_output (bool, optional): if True, outputs are clamped to targets,
      if available, when computing weight updates.
    """
    super().__init__(**kwargs)

    self.lin1_B = torch.nn.Linear(self.num_hidden, self.num_inputs, bias=self.bias)
    self.lin2_B = torch.nn.Linear(self.num_outputs, self.num_hidden, bias=self.bias)

  def activation_deriv(self, x):
    """
    Sets the activation function used for the hidden layers.
    """
    if self.activation_type.lower() == "relu":
      derivative = 1.0 * (x > 0) # maps to positive
    elif self.activation_type.lower() == "sigmoid":
      activations = self.activation(x)
      derivative = activations * (1 - activations) # maps to same
    elif self.activation_type.lower() == "identity":
      derivative = torch.ones_like(x) # maps to same
    else:
      raise NotImplementedError(
          f"{self.activation_type} activation type not recognized. Only "
          "'relu' and 'identity' have been implemented so far."
          )
    return derivative

  def softmax_deriv(self, x, target):
    """
    Sets the activation function used for the output layer.
    """
    output = self.softmax(x)
    derivative = output * (target - output)
    return derivative


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

    h = FeedbackAlignmentFunction.apply(
        X.reshape(-1, self.num_inputs),
        self.lin1.weight,
        self.lin1_B.weight,
        self.lin1.bias,
        self.activation,
        self.activation_deriv
    )

    # if targets are provided, they can be used instead of the last layer's
    # output to train the last layer.
    if y is None:
      targets = None
      output_deriv = None
    else:
      targets = torch.nn.functional.one_hot(
          y, num_classes=self.num_outputs
          ).float()
      output_deriv = self.softmax_deriv

    y_pred = FeedbackAlignmentFunction.apply(
        h,
        self.lin2.weight,
        self.lin2_B.weight,
        self.lin2.bias,
        self.softmax,
        output_deriv, #self.softmax_deriv
        targets
    )

    return y_pred


