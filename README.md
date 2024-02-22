# Triton Activations

Expanded collection of Neural Network activation functions and other function kernels in [Triton by OpenAI.](https://github.com/openai/triton)

## Usage

```bash
git clone github.com/SCALEDSL/triton-activations
cd triton-activations
python3 setup.py install
python3 examples.py
```

## List of the activation functions in Triton Activations

#### **Hyperbolic Tangent Activation Function Kernel**

Applies the hyperbolic tangent activation function to transform input data. It squashes the input values to the range [-1, 1], providing non-linear behavior with outputs centered around zero.

#### **Hard Tanh Activation Function Kernel**

Applies the hard tanh activation function to transform input data. It is a piecewise linear approximation of the hyperbolic tangent function and maps input values to the range [-1, 1]. Unlike the hyperbolic tangent function, the hard tanh function has a flat region where the output saturates, resulting in a thresholding effect. 

#### **Rectified Linear Unit Activation Function Kernel**

Uses the rectified linear unit (ReLU) activation function to process input data. It sets negative values to zero while passing positive values unchanged, resulting in a piecewise linear activation function.

#### **ReLU 6 Activation Function Kernel**

Similar to the Rectified Linear Unit (ReLU) kernel, this kernel also uses the ReLU activation function. However, it additionally clips the output values at 6, limiting them to the range [0, 6].

#### **Leaky ReLU Activation Function Kernel**
Computes the element-wise function leaky_relu(x) = max(x, alpha * x), where x is the input tensor and alpha is a given slope parameter.

#### **Softplus Activation Function Kernel**

Applies the softplus activation function to the input data. It produces smoothed and continuously differentiable outputs, with positive values increasing linearly and negative values converging to zero.

#### **Softsign Activation Function Kernel**

Transforms the input data using the softsign activation function. It maps the input to the range [-1, 1], allowing the output to be sensitive to small changes around zero.

#### **Sigmoid Activation Function Kernel**

Applies the sigmoid activation function to the input data. It squashes the input values between 0 and 1, interpreting them as probabilities.

#### **Hard Sigmoid Activation Function Kernel**

Approximates the sigmoid function with a piecewise linear function. It speeds up computation by sacrificing some accuracy, mapping input values to the range [0, 1].

#### **Sigmoid-weighted Linear Unit Activation Function Kernel**

Combines the sigmoid and linear activation functions. It multiplies the linear component by the sigmoid output, emphasizing or de-emphasizing the linear contribution based on the sigmoid value.

#### **Hard SiLU Activation Function Kernel**

Piecewise linear function that approximates the SiLU (sigmoid-weighted linear unit) activation function. It is designed to provide a non-linear activation while being computationally efficient.

#### **Gaussian Error Linear Unit Activation Function Kernel**

Applies a smooth approximation of the Gaussian cumulative distribution function to the input data. It introduces non-linearity while preserving certain properties of the Gaussian distribution.

#### **Softmax Activation Function Kernel**

Applies the softmax activation function, which converts a vector of real values into a probability distribution. It exponentiates and normalizes the input values, ensuring that the resulting outputs sum up to 1.
