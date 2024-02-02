import torch
from torch import nn

class Swish(nn.Module):
    def __init__(self, beta = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return SwishImplementation.apply(x, self.beta)
             
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i, beta):
        #Check if beta is a tensor because sometimes it is a float
        if not isinstance(beta, torch.Tensor):
            beta = torch.tensor(beta, dtype=i.dtype, device=i.device)
        result = i * torch.sigmoid(beta * i)
        ctx.save_for_backward(i, beta)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, beta = ctx.saved_tensors
        sigmoid_i = torch.sigmoid(beta * i)
        return grad_output * (sigmoid_i * (1 + beta * i * (1 - sigmoid_i))), None
