import torch

NUM_INPUTS = 10
NUM_HIDDEN = 100
NUM_OUTPUTS = 2
ACTIVATION_TYPE = 'sigmoid'
BIAS = False

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(
        self,
        num_inputs=NUM_INPUTS,
        num_hidden=NUM_HIDDEN,
        num_outputs=NUM_OUTPUTS,
        activation_type=ACTIVATION_TYPE,
        bias=BIAS,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.activation_type = activation_type
        self.bias = bias

        self.lin1 = torch.nn.Linear(num_inputs, num_hidden, bias=bias)
        self.lin2 = torch.nn.Linear(num_hidden, num_outputs, bias=bias)        

        self._store_initial_weights_biases()

        self._set_activation()
        self.softmax = torch.nn.Softmax(dim=1)


    def _store_initial_weights_biases(self):

        self.init_lin1_weight = self.lin1.weight.data.clone()
        self.init_lin2_weight = self.lin2.weight.data.clone()
        if self.bias:
            self.init_lin1_bias = self.lin1.bias.data.clone()
            self.init_lin2_bias = self.lin2.bias.data.clone()

    def _set_activation(self):
        if self.activation_type.lower() == "sigmoid":
            self.activation = torch.nn.Sigmoid() # maps to [0, 1]
        elif self.activation_type.lower() == "tanh":
            self.activation = torch.nn.Tanh() # maps to [-1, 1]
        elif self.activation_type.lower() == "relu":
            self.activation = torch.nn.ReLU() # maps to positive
        elif self.activation_type.lower() == "identity":
            self.activation = torch.nn.Identity() # maps to same
        else:
            raise NotImplementedError(
                f"{self.activation_type} activation type not recognized. Only "
                "'sigmoid', 'relu' and 'identity' have been implemented so far."
            )
        
    def forward(self, X, y=None):
        h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
        y_pred = self.softmax(self.lin2(h))
        return y_pred
    
    def forward_backprop(self, X):
        h = self.activation(self.lin1(X.reshape(-1, self.num_inputs)))
        y_pred = self.softmax(self.lin2(h))
        return y_pred

    def list_parameters(self) -> list:
        params_list = list()

        for layer_str in ['lin1', 'lin2']:
            params_list.append(f'{layer_str}_weight')
            if self.bias:
                params_list.append(f'{layer_str}_bias')

        return params_list
    
    def gather_gradient_dict(self) -> dict:
        
        params_list = self.list_parameters()
        
        gradient_dict = dict()
        for param_name in params_list:
            layer_str, param_str = param_name.split('_')
            layer = getattr(self, layer_str)
            grad = getattr(layer, param_str)
            if grad is None:
                raise RuntimeError("No gradient was computed")
            gradient_dict[param_name] = grad.detach().clone().numpy()
        
        return gradient_dict

