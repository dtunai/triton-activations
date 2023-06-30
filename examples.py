import time
import torch
from triton_activations import activations
from typing import Any, Optional, Union

# Create rand array for example usages, select device cuda
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device="cuda")

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

# ReLU 6
start_time = time.time()
output_triton_relu6 = activations.relu6_activation(x)
end_time = time.time()
triton_execution_time_relu6 = end_time - start_time

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

# Sigmoid
start_time = time.time()
output_triton_sigmoid = activations.sigmoid_activation(x)
end_time = time.time()
triton_execution_time_sigmoid = end_time - start_time

# Hard Sigmoid
start_time = time.time()
output_triton_hard_sigmoid = activations.hard_sigmoid_activation(x)
end_time = time.time()
triton_execution_time_hard_sigmoid = end_time - start_time

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

print(f"Output triton (Tanh): {output_triton_tanh}\n")
print(f"Triton execution time (Tanh): {triton_execution_time_tanh} seconds\n")

print("---------------")

print(f"Output triton (ReLU): {output_triton_relu}\n")
print(f"Triton execution time (ReLU): {triton_execution_time_relu} seconds\n")

print("---------------")

print(f"Output triton (ReLU 6): {output_triton_relu6}\n")
print(f"Triton execution time (ReLU): {triton_execution_time_relu6} seconds\n")

print("---------------")

print(f"Output triton (Softplus): {output_triton_softplus}\n")
print(f"Triton execution time (Softplus): {triton_execution_time_softplus} seconds\n")

print("---------------")

print(f"Output triton (Softsign): {output_triton_softsign}\n")
print(f"Triton execution time (Softsign): {triton_execution_time_softsign} seconds\n")

print("---------------")

print(f"Output triton (Sigmoid): {output_triton_sigmoid}\n")
print(f"Triton execution time (Sigmoid): {triton_execution_time_sigmoid} seconds\n")

print("---------------")

print(f"Output triton (Hard Sigmoid): {output_triton_hard_sigmoid}\n")
print(
    f"Triton execution time (Hard Sigmoid): {triton_execution_time_hard_sigmoid} seconds\n"
)

print("---------------")

print(f"Output triton (SiLU): {output_triton_silu}\n")
print(f"Triton execution time (SiLU): {triton_execution_time_silu} seconds\n")

print("---------------")

print(f"Output triton (GeLU Approximate True): {output_triton_gelu_approximate_true}\n")
print(
    f"Triton execution time (GeLU Approximate True): {triton_execution_time_gelu_approximate_true} seconds\n"
)

print("---------------")

print(f"Output triton (Softmax): {output_triton_softmax}\n")
print(f"Triton execution time (Softmax): {triton_execution_time_softmax} seconds\n")
