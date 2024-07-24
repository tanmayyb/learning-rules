import torch
from .classes.MLP import MultiLayerPerceptron



# from @NMA Project notebook implementation: https://neuroai.neuromatch.io/projects/project-notebooks/Microlearning.html
class HebbianFunction(torch.autograd.Function):


    @staticmethod
    def forward(context, input, weight, bias=None, nonlinearity=None, target=None):
        
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        if nonlinearity is not None:
            output = nonlinearity(output)

        output_for_update = output if target is None else target
        context.save_for_backward(input, weight, bias, output_for_update) 

        return output

    @staticmethod
    def backward(context, grad_output=None):
        input, weight, bias, output_for_update = context.saved_tensors

        grad_input = grad_weight = grad_bias = grad_nonlinearity = grad_target = None

        input_needs_grad = context.needs_input_grad[0]
        if input_needs_grad:
            pass

        weight_needs_grad = context.needs_input_grad[1]
        if weight_needs_grad:
            grad_weight = output_for_update.t().mm(input)
            grad_weight = grad_weight / len(input) # average across batch

            # center around 0
            grad_weight = grad_weight - grad_weight.mean(axis=0) # center around 0

            ## or apply Oja's rule (not compatible with clamping outputs to the targets!)
            # oja_subtract = output_for_update.pow(2).mm(grad_weight).mean(axis=0)
            # grad_weight = grad_weight - oja_subtract

            # take the negative, as the gradient will be subtracted
            grad_weight = -grad_weight

        if bias is not None:
            bias_needs_grad = context.needs_input_grad[2]
            if bias_needs_grad:
                grad_bias = output_for_update.mean(axis=0) # average across batch

                # center around 0
                grad_bias = grad_bias - grad_bias.mean()

                ## or apply an adaptation of Oja's rule for biases
                ## (not compatible with clamping outputs to the targets!)
                # oja_subtract = (output_for_update.pow(2) * bias).mean(axis=0)
                # grad_bias = grad_bias - oja_subtract

                # take the negative, as the gradient will be subtracted
                grad_bias = -grad_bias

        return grad_input, grad_weight, grad_bias, grad_nonlinearity, grad_target


class HebbianNetwork(MultiLayerPerceptron):

    def __init__(self, clamp_output=True, **kwargs):
        self.clamp_output = clamp_output
        super().__init__(**kwargs)

    def forward(self, X, y=None):
        # return super().forward(X, y)

        h = HebbianFunction.apply(
            X.reshape(-1, self.num_inputs),
            self.lin1.weight,
            self.lin1.bias,
            self.activation,
        )    

        if y is None or not self.clamp_output:
            targets = None
        else:
            targets = torch.nn.functional.one_hot(
                y, num_classes=self.num_outputs
            ).float()

        y_pred = HebbianFunction.apply(
            h,
            self.lin2.weight,
            self.lin2.bias,
            self.softmax,
            targets
        )

        return y_pred    

