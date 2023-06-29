import time
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

# create rand array for example usages
torch.manual_seed(0)
size = 98432
x = torch.rand(size)

# Tanh
start_time = time.time()
output_triton_tanh = tanh_activation(x)
end_time = time.time()
triton_execution_time_tanh = end_time - start_time

# ReLU
start_time = time.time()
output_triton_relu = relu_activation(x)
end_time = time.time()
triton_execution_time_relu = end_time - start_time

# Softplus
start_time = time.time()
output_triton_softplus = softplus_activation(x)
end_time = time.time()
triton_execution_time_softplus = end_time - start_time

# Softsign
start_time = time.time()
output_triton_softsign = softsign_activation(x)
end_time = time.time()
triton_execution_time_softsign = end_time - start_time

# SiLU
start_time = time.time()
output_triton_silu = silu_activation(x)
end_time = time.time()
triton_execution_time_silu = end_time - start_time

# GeLU
start_time = time.time()
output_triton_gelu_approximate_false = gelu_activation(x, approximate=False)
end_time = time.time()
triton_execution_time_gelu_approximate_false = end_time - start_time

# Softmax
start_time = time.time()
output_triton_softmax = softmax_activation(x, axis_ld=0)
end_time = time.time()
triton_execution_time_softmax = end_time - start_time

print(f'Output triton (Tanh): {output_triton_tanh}\n')
print(f'Triton execution time (Tanh): {triton_execution_time_tanh} seconds\n')

print("---------------")

print(f'Output triton (ReLU): {output_triton_relu}\n')
print(f'Triton execution time (ReLU): {triton_execution_time_relu} seconds\n')

print("---------------")

print(f'Output triton (Softplus): {output_triton_softplus}\n')
print(f'Triton execution time (Softplus): {triton_execution_time_softplus} seconds\n')

print("---------------")

print(f'Output triton (Softsign): {output_triton_softsign}\n')
print(f'Triton execution time (Softsign): {triton_execution_time_softsign} seconds\n')

print("---------------")

print(f'Output triton (SiLU): {output_triton_silu}\n')
print(f'Triton execution time (SiLU): {triton_execution_time_silu} seconds\n')

print("---------------")

print(f'Output triton (GeLU Approximate False): {output_triton_gelu_approximate_false}\n')
print(f'Triton execution time (GeLU Approximate False): {triton_execution_time_gelu_approximate_false} seconds\n')

print("---------------")

print(f'Output triton (Softmax): {output_triton_softmax}\n')
print(f'Triton execution time (Softmax): {triton_execution_time_softmax} seconds\n')