import torch

from triton_activations import funcs
from typing import Any, Optional, Union

def tanh_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.tanh_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def relu_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def softplus_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.softplus_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def softsign_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.softsign_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def sigmoid_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.sigmoid_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

def silu_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.silu_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def gelu_activation(x: torch.Tensor, approximate: bool = True):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.gelu_activation_kernel[grid](x, output, approximate, n_elements, BLOCK_SIZE=1024)
    return output

def softmax_activation(x: torch.Tensor, axis_ld: Optional[Union[int, tuple[int, ...]]] = -1):
    if axis_ld is None:
        axis_ld = 0
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    funcs.softmax_activation_kernel[grid](x, output, axis_ld, approximate, n_elements, BLOCK_SIZE=1024)
    return output