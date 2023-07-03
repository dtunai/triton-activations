import torch
import triton
import triton.language as tl
from triton_activations import functions
from typing import Any, Optional, Union


def apply_activation(x: torch.Tensor, activation_fn: Any, *args, **kwargs):
    """
    Applies the specified activation function element-wise to the input tensor
    """
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    if "axis_ld" in kwargs:
        axis_ld = kwargs.pop("axis_ld")
        activation_fn[grid](
            x, output, axis_ld, n_elements, BLOCK_SIZE=1024, *args, **kwargs
        )
    else:
        activation_fn[grid](x, output, n_elements, BLOCK_SIZE=1024, *args, **kwargs)
    return output


def tanh_activation(x: torch.Tensor):
    """
    Applies the hyperbolic tangent (tanh) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.tanh_activation_kernel)


def hard_tanh_activation(x: torch.Tensor):
    """
    Applies the hard tanh activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.hard_tanh_activation_kernel)


def relu_activation(x: torch.Tensor):
    """
    Applies the rectified linear unit (ReLU) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.relu_activation_kernel)


def relu6_activation(x: torch.Tensor):
    """
    Applies the rectified linear unit 6 (ReLU 6) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.relu6_activation_kernel)


def leaky_relu_activation(x: torch.Tensor, alpha: float = 0.2):
    """
    Applies the LeakyReLU activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.leaky_relu_activation_kernel, alpha=alpha)


def softplus_activation(x: torch.Tensor):
    """
    Applies the softplus activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.softplus_activation_kernel)


def softsign_activation(x: torch.Tensor):
    """
    Applies the softsign activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.softsign_activation_kernel)


def sigmoid_activation(x: torch.Tensor):
    """
    Applies the sigmoid activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.sigmoid_activation_kernel)


def hard_sigmoid_activation(x: torch.Tensor):
    """
    Applies the hard sigmoid activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.hard_sigmoid_activation_kernel)


def silu_activation(x: torch.Tensor):
    """
    Applies the Sigmoid-weighted Linear Unit (SiLU) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.silu_activation_kernel)


def hard_silu_activation(x: torch.Tensor,):
    """
    Applies the hard SiLU activation function to element-wise to the input tensor
    """
    return apply_activation(x, functions.hard_silu_activation_kernel)


def gelu_activation(x: torch.Tensor, approximate: bool = True):
    """
    Applies the Gaussian Error Linear Unit (GELU) activation function element-wise to the input tensor
    """
    return apply_activation(x, functions.gelu_activation_kernel, approximate)


def softmax_activation(
    x: torch.Tensor, axis_ld: Optional[Union[int, tuple[int, ...]]] = -1
):
    """
    Applies the softmax activation function to the input tensor along the specified axis
    """
    if axis_ld is None:
        axis_ld = 0
    return apply_activation(x, functions.softmax_activation_kernel, axis_ld)
