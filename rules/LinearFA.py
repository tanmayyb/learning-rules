import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable


#Feedback alignment: https://github.com/L0SG/feedback-alignment-pytorch/blob/master/lib/fa_linear.py
class LinearFAFunction(autograd.Function):

    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_fa.to(grad_output.device))

        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFAModule(nn.Module):

    def __init__(self, input_features, output_features, bias=True):
        super().__init__()

        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features,))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weight_fa = Variable(torch.FloatTensor(output_features, input_features), requires_grad=False)

        torch.nn.init.kaiming_uniform(self.weight)
        torch.nn.init.kaiming_uniform(self.weight_fa)
        torch.nn.init.constant(self.bias, 1)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)


class LinearFANetwork(nn.Module):

    def __init__(self, in_features, num_layers, num_hidden_list):
        super().__init__()

        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list

        # first hiddent layer
        self.linear = [LinearFAModule(self.in_features, self.num_hidden_list[0])]
        for idx in range(self.num_layers - 1):
            # middle hidden layers
            self.linear.append(LinearFAModule(self.num_hidden_list[idx], self.num_hidden_list[idx+1]))

        # register module list 
        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):

        # first hidden layer
        linear1 = self.linear[0](inputs)
        # other hidden layers        
        linear2 = self.linear[1](linear1)

        return linear2