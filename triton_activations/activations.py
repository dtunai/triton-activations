import torch
from triton_activations import functions
from typing import Any, Optional, Union

def apply_activation(x: torch.Tensor, activation_fn: Any, *args, **kwargs):
    """
    applies the specified activation function element-wise to the input tensor
    """
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    if 'axis_ld' in kwargs:
        axis_ld = kwargs.pop('axis_ld')
        activation_fn[grid](x, output, axis_ld, n_elements, BLOCK_SIZE=1024, *args, **kwargs)
    else:
        activation_fn[grid](x, output, n_elements, BLOCK_SIZE=1024, *args, **kwargs)
    return output

def tanh_activation(x: torch.Tensor):
    """
    applies the hyperbolic tangent (tanh) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.tanh_activation_kernel)

def relu_activation(x: torch.Tensor):
    """
    applies the rectified linear unit (ReLU) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.relu_activation_kernel)

def softplus_activation(x: torch.Tensor):
    """
    applies the softplus activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.softplus_activation_kernel)

def softsign_activation(x: torch.Tensor):
    """
    applies the softsign activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.softsign_activation_kernel)

def sigmoid_activation(x: torch.Tensor):
    """
    applies the sigmoid activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.sigmoid_activation_kernel)

def silu_activation(x: torch.Tensor):
    """
    applies the Sigmoid-weighted Linear Unit (SiLU) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.silu_activation_kernel)

def gelu_activation(x: torch.Tensor, approximate: bool = True):
    """
    applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.gelu_activation_kernel, approximate)

def softmax_activation(x: torch.Tensor, axis_ld: Optional[Union[int, tuple[int, ...]]] = -1):
    """
    applies the softmax activation function to the input tensor along the specified axis
    """
    if axis_ld is None:
        axis_ld = 0
    return apply_activation(x, functions.softmax_activation_kernel, axis_ld)
