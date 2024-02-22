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


@triton.jit
def hard_tanh_activation_kernel(
    x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    Hard Tanh activation function kernel

    Computes the element-wise function: hard\_tanh}(x) = -1, & x < -1\\x, & -1 \le x \le 1\\1, & 1 < x
    """

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    shape_condition = tl.where(x < -1, -1, x)
    output = tl.where(x > 1, 1, shape_condition)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def relu_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    ReLU activation function kernel

    Computes the element-wise function: {relu}(x) = \max(x, 0)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(0, x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def relu6_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    ReLU 6 activation function kernel

    Computes the element-wise function: {relu6}(x) = \min(\max(x, 0), 6)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.libdevice.min(tl.libdevice.max(x, 0), 6.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def leaky_relu_activation_kernel(
    x_ptr, output_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr
):
    """
    Leaky ReLU activation function kernel

    Computes the element-wise function: {leaky_relu}(x) = \max(x, \alpha * x) with a given alpha slope value
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.maximum(x, alpha * x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def softplus_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Softplus activation function kernel

    Computes the element-wise function: {softplus}(x) = \log(1 + e^x)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.log(1 + tl.exp(x))
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def softsign_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Softsign activation function kernel

    Computes the element-wise function: {soft\_sign}(x) = \frac{x}{|x| + 1}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x / (tl.libdevice.abs(x) + 1)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def sigmoid_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Sigmoid activation function kernel

    Computes the element-wise function: {sigmoid}(x) = \frac{1}{1 + e^{-x}}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = 1 / (1 + tl.exp(-x))
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def hard_sigmoid_activation_kernel(
    x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    Hard sigmoid activation function kernel

    Computes the element-wise function: {hard\_sigmoid}(x) = \frac{\mathrm{relu6}(x + 3)}{6}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_plus_3 = x + 3.0
    relu6_result = tl.libdevice.min(tl.libdevice.max(x_plus_3, 0), 6.0)
    output = relu6_result / 6.0
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def silu_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    SiLU activation function kernel

    Computes the element-wise function: {silu}(x) = x \cdot \mathrm{sigmoid}(x) = \frac{x}{1 + e^{-x}}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * (1 / (1 + tl.exp(-x)))
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def hard_silu_activation_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Hard SiLU activation function kernel

    Computes the element-wise function: {hard\_silu}(x) = x \cdot \mathrm{hard\_sigmoid}(x)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    x_plus_3 = x + 3.0
    relu6_result = tl.libdevice.min(tl.libdevice.max(x_plus_3, 0), 6.0)
    hard_sigmoid_output = relu6_result / 6.0
    output = x * hard_sigmoid_output
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def gelu_activation_kernel(
    x_ptr, output_ptr, approximation, n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    GeLU activation function kernel

    Computes the element-wise function: {gelu}(x) = \frac{x}{2} \left(1 + \mathrm{erf} \left(\frac{x}{\sqrt{2}} \right) \right)

    If ``approximate=True``, uses the approximate formulation of GELU:

    Computes the approximate formulation of GeLU: {gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left( \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    if approximation == True:
        output = (
            0.5
            * x
            * (
                1
                + tl.libdevice.tanh(
                    tl.libdevice.sqrt(2.0 / 3.141592653589793)
                    * (x + 0.044715 * x * x * x)
                )
            )
        )
        tl.store(output_ptr + offsets, output, mask=mask)
    else:
        output = x * 0.5 * (1.0 + tl.libdevice.erf(x / tl.libdevice.sqrt(2.0)))
        tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def softmax_activation_kernel(
    x_ptr, output_ptr, axis_ld, n_elements, BLOCK_SIZE: tl.constexpr
):
    """
    Softmax activation function kernel

    Computes the function which rescales elements to the range`[0, 1]`: {softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    max_x = tl.libdevice.max(x, axis_ld)
    x -= max_x
    exp_x = tl.libdevice.exp(x)
    sum_exp_x = exp_x + axis_ld
    output = exp_x / sum_exp_x
    tl.store(output_ptr + offsets, output, mask=mask)
