import time
import torch
from triton_activations import activations
from typing import Any, Optional, Union

# create rand array for example usages
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')

# Tanh
start_time = time.time()
output_triton_tanh = activations.tanh_activation(x)
end_time = time.time()
triton_execution_time_tanh = end_time - start_time

# ReLU
start_time = time.time()
output_triton_relu = activations.relu_activation(x)
end_time = time.time()
triton_execution_time_relu = end_time - start_time

# Softplus
start_time = time.time()
output_triton_softplus = activations.softplus_activation(x)
end_time = time.time()
triton_execution_time_softplus = end_time - start_time

# Softsign
start_time = time.time()
output_triton_softsign = activations.softsign_activation(x)
end_time = time.time()
triton_execution_time_softsign = end_time - start_time

# SiLU
start_time = time.time()
output_triton_silu = activations.silu_activation(x)
end_time = time.time()
triton_execution_time_silu = end_time - start_time

# GeLU
start_time = time.time()
output_triton_gelu_approximate_true = activations.gelu_activation(x, approximate=True)
end_time = time.time()
triton_execution_time_gelu_approximate_true = end_time - start_time

# Softmax
start_time = time.time()
output_triton_softmax = activations.softmax_activation(x, axis_ld=10)
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

print(f'Output triton (GeLU Approximate True): {output_triton_gelu_approximate_true}\n')
print(f'Triton execution time (GeLU Approximate True): {triton_execution_time_gelu_approximate_true} seconds\n')

print("---------------")

print(f'Output triton (Softmax): {output_triton_softmax}\n')
print(f'Triton execution time (Softmax): {triton_execution_time_softmax} seconds\n')
