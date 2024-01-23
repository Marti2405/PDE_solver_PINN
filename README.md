# N-Dimensional Partial Differential Equation (PDE) Solver with Physics-Informed Neural Networks (PINNs)

[![GitHub](https://img.shields.io/badge/GitHub-Repository-brightgreen.svg)](https://github.com/Marti2405/PDE_solver_PINN)

## 

**Author:** Marti JIMENEZ  
**Date:** 01/08/2023  
**License:** MIT License 

## Introduction

This repository contains Python code to solve n-Dimensional Partial Differential Equations (PDEs) with Physics-Informed Neural Networks (PINNs). The code leverages TensorFlow, an open-source deep learning library, to build and train the neural network.

This Framework harnesses the flexibility of neural networks to efficiently approximate complex PDE solutions while incorporating known physics-based constraints, eliminating the need for large labeled datasets.

Solving complex, high-dimensional PDEs can be challenging, especially when analytical solutions are not feasible. PINNs can achieve accurate and robust solutions without increasing the complexity exponentially for every dimension added in the PDE like other methods (e.g. Finite Difference Method).


## Dependencies

- Python 3.x
- TensorFlow (>=2.0)
- NumPy
- Matplotlib

Table of Contents
=================

- [How to Use](#how-to-use)
  - [Dimensions and Variable Boundaries](#dimensions-and-variable-boundaries)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Data-set](#data-set)
  - [Training Parameters](#training-parameters)
  - [Saving the Model](#saving-the-model)
  - [PDE Loss](#pde-loss)
  - [Initial Condition Loss](#initial-condition-loss)
  - [Neumann Boundary Condition Loss](#neumann-boundary-condition-loss)
  - [Dirichlet Boundary Condition Loss](#dirichlet-boundary-condition-loss)
- [Data Preparation](#data-preparation)
- [Neural Network Model](#neural-network-model)
- [Loss Function](#loss-function)
- [Training](#training)
- [Result Visualization, Model Loading](#result-visualization-model-loading)

## How to Use

As an example, let's consider the Fisher-KPP equation in 2 dimensions:

$\frac{{\partial u}}{{\partial t}} = D\left(\frac{{\partial^2 u}}{{\partial x^2}} + \frac{{\partial^2 u}}{{\partial y^2}}\right) + r \cdot u(1 - u)$

Where \(u\) is the unknown function, \(D\) is the diffusion coefficient, and \(r\) is the growth rate.


### Dimensions and Variable Boundaries

You can define the number of dimensions of the PDE and the boundaries for each variable. For the Fisher-KPP equation, we have:

```python
DIMENSIONS = 3  # Take into account the time dimension
BOUNDARIES = [
    [0.0, 5.0],  # Time boundaries
    [0.0, 1.0],  # x boundaries
    [0.0, 1.0],  # y boundaries
    # Add more boundaries for additional dimensions if needed
]
```

### Neural Network Architecture

You can customize the Neural Network architecture with the number of hidden layers, neurons per layer, and activation function.

```python
NUMBER_HIDDEN_LAYERS = 2
NUMBER_NEURONS_PER_LAYER = 20
ACTIVATION = tf.keras.activations.tanh
```

### Data-set

Set the number of data points for the PDE, initial condition, and boundary condition. (Please note that the number of boundary points will be multiplied by the number of dimensions)

```python
NUMBER_DATA_POINTS_PDE = 10000
NUMBER_DATA_POINTS_INITIAL_CONDITION = 1000
NUMBER_DATA_POINTS_BOUNDARY_CONDITION = 1000
```

### Training Parameters

Define the number of training epochs. Set the interval to print the loss during training. Set the PLOT_HISTORY_LOSS to true if you want to see the plot of the loss history after training.

```python
NUMBER_TRAINING_EPOCHS = 20000
PRINT_LOSS_INTERVAL = 10
PLOT_LOSS_HISTORY = True
```

### Saving the Model

Set the path to save the model checkpoints and the interval at which the model will be saved.

```python
CHECKPOINT_PATH = 'model_checkpoint'
CHECKPOINT_ITERATIONS = 100
```

### PDE Loss

To define the residual of the PDE, modify the return value of the function `comp_r`.

The inputs of the function will be automatically computed and parsed to the function.
- var is the list of input variables [t, x, y, z, ...]
- u is the output of the neural network, representing the predicted value of the PDE
- first deriv is a list containing the first derivatives [ut, ux, uy, uz , ...]
- second deriv is a list containing the second derivatives [utt, uxx, uyy, uzz , ...]

Use the necessary terms for your PDE inside this function and return the PDE residual. Here
you can see the residual for the Fisher-KPP equation:
```python
def comp_r(var, u, first_deriv, second_deriv):
    u_t = first_deriv[0] 
    u_xx, u_yy = second_deriv[1], second_deriv[2]
    return u_t - D*(u_xx+u_yy) - r*u*(1-u)
```

### Initial Condition Loss

To define the initial condition Loss, modify the return of the function `comp_i(X)`. The input of this function is a matrix containing the initial condition points with one column per variable (starting with the time variable) `[t, x, y, z, ...]`.

Here you can see the residual for the Fisher-KPP equation:

```python
def comp_i(X):
    x = X[:, 1:2]
    y = X[:, 2:3]
    return tf.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.08)
```

### Neumann Boundary Condition Loss

To impose the Neumann boundary condition, set the variable `CHOOSE_NEUMANN_BOUNDARY_CONDITION` to `True` and set the variable `NEUMANN_BOUNDARY_CONDITION` to the desired value.

Here you can see the Neumann Boundary Condition $\nabla u\cdot n =0$:

```python
CHOOSE_NEUMANN_BOUNDARY_CONDITION = True
NEUMANN_BOUNDARY_CONDITION = 0
```

### Dirichlet Boundary Condition Loss

To impose the Dirichlet boundary condition, set the variable `CHOOSE_NEUMANN_BOUNDARY_CONDITION` to `False` and modify the expected value in the `DC_comp_b(model, x_b)` function.

Here you can see the Dirichlet Boundary Condition $u(t,x,y) = \sin(\pi.x)+sin(\pi.y)$:

```python
CHOOSE_NEUMANN_BOUNDARY_CONDITION = False

def DC_comp_b(prediction, x_b):
    t = x_b[:, 0:1]  # Variables
    x = x_b[:, 1:2]  # Variables
    y = x_b[:, 2:3]  # Variables
    expected = tf.sin(pi*x) + tf.sin(pi*y)  # Expected output at the boundary
    return prediction - expected
```

## Data Preparation

The framework generates the wanted number of data points for initial conditions, boundary conditions, and PDE data.

## Neural Network Model

The framework initializes the Neural Network model with the specified architecture and compiles it with an Adam optimizer using the defined learning rates.

## Loss Function

The loss function combines three components: PDE residual loss, boundary condition residual loss, and initial condition loss. The PDE residual loss is obtained by computing the residuals of the PDE equation using the neural network's predicted values and their derivatives. The boundary condition residual loss ensures that the neural network satisfies the boundary conditions. The initial condition loss enforces the initial condition.

## Training

The model is trained using Adam optimizer. The loss is computed using the three components mentioned above. During training, the model parameters are updated to minimize the loss.

## Result Visualization, Model Loading

After training, the model is saved. You can use the model to predict the PDE solution by calling `model.predict(X)` with `X` representing the desired points. (There is an example in the code to show the solution of the 2D Fisher-KPP equation).

To load a model, you have to initialize a model with the same neural network architecture 
```
model1 = init_model()
``` 
and then load the weights previously saved in a .h5 file 
```
model1.load_weights(filepath.h5)
```



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

This code is developed by Marti JIMENEZ as part of research conducted at the Laboratory of Pathogen Host Interactions (LPHI) of Montpellier, France. If you find this code useful for your work, we appreciate acknowledgment in your publications.

Feel free to reach out to the author if you have any questions or need further assistance with the code.


