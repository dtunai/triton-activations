import time
import torch
from triton_activations import activations
from typing import Any, Optional, Union


# Measurement execution method
def measure_execution_time(activation_func, x):
    start_time = time.time()
    output = activation_func(x)
    end_time = time.time()
    execution_time = end_time - start_time
    return output, execution_time


# Random array for example usages and select device "cuda"
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device="cuda")

# Activation functions dict object
activation_functions = {
    "Tanh": activations.tanh_activation,
    "Hard Tanh": activations.hard_tanh_activation,
    "ReLU": activations.relu_activation,
    "ReLU 6": activations.relu6_activation,
    "Leaky ReLU": lambda x: activations.leaky_relu_activation(x, alpha=0.2),
    "Softplus": activations.softplus_activation,
    "Softsign": activations.softsign_activation,
    "Sigmoid": activations.sigmoid_activation,
    "Hard Sigmoid": activations.hard_sigmoid_activation,
    "SiLU": activations.silu_activation,
    "Hard SiLU": activations.hard_silu_activation,
    "GeLU Approximate True": lambda x: activations.gelu_activation(x, approximate=True),
    "Softmax": lambda x: activations.softmax_activation(x, axis_ld=10),
}

# Measure execution time for each activation function
execution_times = {}
output_results = {}

# Process execution time
for activation_name, activation_func in activation_functions.items():
    output, execution_time = measure_execution_time(activation_func, x)
    execution_times[activation_name] = execution_time
    output_results[activation_name] = output

# Drop results
for activation_name, execution_time in execution_times.items():
    print(f"Output triton ({activation_name}): {output_results[activation_name]}\n")
    print(f"Triton execution time ({activation_name}): {execution_time} seconds\n")
    print("---------------")
