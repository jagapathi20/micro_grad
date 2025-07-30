# Micrograd Implementation

A minimal neural network library with automatic differentiation, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). This is a learning implementation that demonstrates the core concepts of automatic differentiation and neural network training from scratch.

## 🚀 Features

- **Automatic Differentiation**: Scalar-valued autograd engine with backward pass
- **Neural Network Components**: Neurons, layers, and multi-layer perceptrons (MLPs)
- **Activation Functions**: ReLU and Tanh support
- **Training Loop**: Complete gradient-based optimization
- **Educational**: Clean, readable code perfect for understanding backpropagation

## 📁 Repository Structure

```
micrograd-implementation/
├── engine.py           # Core Value class with autograd
├── neural_network.py   # Neural network components (Neuron, Layer, MLP)
├── demo.ipynb         # Training example on breast cancer dataset
└── README.md          # This file
```

## 🔧 Core Components

### Value Class (`engine.py`)
The heart of the autograd engine - a scalar value that tracks gradients:

```python
from engine import Value

# Create values
a = Value(2.0)
b = Value(4.0)

# Operations automatically build computation graph
c = a * b + a.tanh()

# Backpropagation
c.backward()
print(f"dc/da = {a.grad}")  # Gradient of c with respect to a
```

**Supported Operations:**
- Addition: `a + b`
- Multiplication: `a * b`
- Power: `a ** n`
- Division: `a / b`
- Tanh activation: `a.tanh()`
- ReLU activation: `a.relu()`

### Neural Network (`neural_network.py`)

**Neuron**: Single perceptron with weights, bias, and activation
```python
neuron = Neuron(nin=3, nonlin=True)  # 3 inputs, ReLU activation
output = neuron([1.0, 2.0, 3.0])
```

**Layer**: Collection of neurons
```python
layer = Layer(nin=3, nout=5)  # 3 inputs, 5 outputs
```

**MLP**: Multi-layer perceptron
```python
model = MLP(nin=30, nouts=[16, 16, 1])  # 30->16->16->1 architecture
```

## 🎯 Example Usage

### Binary Classification on Breast Cancer Dataset

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from neural_network import MLP
from engine import Value

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create model
model = MLP(30, [16, 16, 1])

# Training loop
for epoch in range(100):
    # Forward pass
    inputs = [list(map(Value, row)) for row in X_train]
    predictions = [model(x) for x in inputs]
    
    # Loss calculation (MSE + L2 regularization)
    loss = sum((pred - target)**2 for pred, target in zip(predictions, y_train))
    loss = loss * (1.0 / len(predictions))
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Parameter update
    learning_rate = 0.01
    for param in model.parameters():
        param.data -= learning_rate * param.grad
```

## 📊 Demo Results

The included Jupyter notebook (`demo.ipynb`) demonstrates:
- Loading and preprocessing the breast cancer dataset
- Training a neural network from scratch
- Achieving ~95% accuracy on binary classification
- Visualizing training progress

## 🛠️ Installation & Requirements

```bash
# Clone the repository
git clone https://github.com/yourusername/micrograd-implementation.git
cd micrograd-implementation

# Install dependencies
pip install numpy scikit-learn matplotlib jupyter
```

**Dependencies:**
- `numpy` - For numerical operations
- `scikit-learn` - For datasets and preprocessing
- `matplotlib` - For plotting (demo only)
- `jupyter` - For running the demo notebook

## 🎓 Learning Objectives

This implementation helps understand:

1. **Automatic Differentiation**: How computational graphs track gradients
2. **Backpropagation**: The chain rule in action
3. **Neural Network Architecture**: Building networks from basic components
4. **Training Process**: Forward pass, loss calculation, backward pass, parameter updates
5. **Gradient-based Optimization**: How neural networks learn

## 🔍 Key Implementation Details

- **Numerical Stability**: Uses `math.tanh()` to avoid overflow errors
- **Memory Efficiency**: Scalar-based operations (not vectorized)
- **Educational Focus**: Prioritizes clarity over performance
- **Gradient Accumulation**: Proper handling of parameter gradients

## 🚨 Limitations

- **Performance**: Not optimized for large-scale training (use PyTorch/TensorFlow for real projects)
- **Scalar Operations**: No vectorization - slow on large datasets
- **Limited Functionality**: Basic operations only
- **Educational Purpose**: Designed for learning, not production use

## 🙏 Acknowledgments

This implementation is inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and his excellent educational content. This project was created as a learning exercise to understand the fundamentals of automatic differentiation and neural networks.

## 📖 Further Reading

- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Original micrograd repository](https://github.com/karpathy/micrograd)
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)

---

**Note**: This is an educational implementation. For production neural networks, use established frameworks like PyTorch, TensorFlow, or JAX.