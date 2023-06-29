import time
import math
import torch
import triton
import triton.language as tl

@triton.jit
def tanh_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Tanh activation function kernel
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.libdevice.tanh(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def tanh_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def relu_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    ReLU activation function kernel
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(0, x)
    tl.store(output_ptr + offsets, output, mask=mask)


def relu_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def softplus_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Softplus activation function kernel
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output= tl.log(1 + tl.exp(x))
    tl.store(output_ptr + offsets, output, mask=mask)


def softplus_activation(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    softplus_activation_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


torch.manual_seed(0)
size = 98432

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
output_triton_softplus= softplus_activation(x)
end_time = time.time()
triton_execution_time_softplus = end_time - start_time

print(f'Output triton (Tanh): {output_triton_tanh}\n')
print(f'Triton execution time (Tanh): {triton_execution_time_tanh} seconds\n')

print("---------------")

print(f'Output triton (ReLU): {output_triton_relu}\n')
print(f'Triton execution time (ReLU): {triton_execution_time_relu} seconds\n')

print("---------------")

print(f'Output triton (Softplus): {output_triton_softplus}\n')
print(f'Triton execution time (Softplus): {triton_execution_time_softplus} seconds\n')