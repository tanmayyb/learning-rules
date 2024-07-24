import torch
from .classes.MLP import MultiLayerPerceptron


class HKPNetwork(MultiLayerPerceptron):
    def __init__(context, input, weight, weight_backward, bias=None, nonlinearity=None, target=None):

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        if nonlinearity is not None:
            output = nonlinearity(output)

        # calculate the output to use for the backward pass
        output_for_update = output if target is None else target

        # store variables in the context for the backward pass
        context.save_for_backward(input, weight, bias, output_for_update)

        return output
