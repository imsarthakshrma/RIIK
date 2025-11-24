# RIIK: My Neural Network Implementation from Scratch

(I have named it RIIK because I wanted to.) 

Welcome to my implementation of a neural network built entirely from scratch using just NumPy! This project represents my journey into understanding the inner workings of deep learning frameworks by building one myself.

##  What I've Built?

I've implemented a fully functional neural network with all the bells and whistles you'd expect from a modern deep learning framework:

- **Optimization**: Adam and SGD with momentum for efficient training
- **Regularization**: Dropout, L1/L2 regularization, and batch normalization
- **Features**: Early stopping, learning rate scheduling, and mini-batch training
- **Versatility**: Supports both classification and regression tasks
- **Production-ready**: Model saving/loading and comprehensive evaluation metrics

##  Installation

Getting started is simple - just install the requirements:

```bash
pip install -r requirements.txt
```

##  Project Structure

Here's how I've organized the code:

```
neural_networks/
├── core/               # Core neural network implementation
│   ├── __init__.py     
│   ├── network.py     # Main neural network class
│   ├── layer.py        # Layer implementation
│   ├── activation.py   # Activation functions
│   └── optimizers.py   # Optimization algorithms
├── datasets/           # Data handling
│   ├── __init__.py
│   └── loaders.py      # Data loading utilities
└── examples/           # Example scripts
```

<!-- ##  Quick Start

Here's how you can use my neural network implementation:

```python
from neural_networks.core import NeuralNetwork
from neural_networks.datasets.loaders import DataLoader

# Load some data
data = DataLoader.load_classification_dataset('breast_cancer')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Define the network architecture
architecture = [
    (30, 64, 'relu', 0.3, True),    # Input layer
    (64, 32, 'relu', 0.3, True),     # Hidden layer 1
    (32, 16, 'relu', 0.2, False),    # Hidden layer 2
    (16, 1, 'sigmoid', 0.0, False)   # Output layer
]

# Create and train the model
model = NeuralNetwork(architecture, task='classification', optimizer='adam', learning_rate=0.001)

model.fit(
    X_train, y_train, X_val, y_val,
    epochs=500, 
    batch_size=32,
    l2_reg=0.001, 
    early_stopping=True,
    patience=30, 
    lr_schedule=True
) -->

# Evaluate on test set
results = model.evaluate(X_test, y_test)
model.plot_history()

# Save model
model.save_model('breast_cancer_model.pkl')

# Load model
loaded_model = NeuralNetwork.load_model('breast_cancer_model.pkl')
```

## Examples

Check out the `examples/` directory for complete working examples, including:

- Binary classification (Breast Cancer dataset)
- Regression (California Housing prices)
- Multi-class classification (Synthetic data)

## Why I Built This?

I created this project to deepen my understanding of how neural networks work under the hood. By implementing everything from scratch, I've gained valuable insights into the mathematics and computational aspects of deep learning.

## 📊 Performance

I'm proud to say that my implementation achieves comparable performance to scikit-learn's MLP models while offering more flexibility and control. The modular design makes it easy to experiment with different architectures and techniques.

## What I Learned

- The importance of numerical stability in deep learning
- How different optimization techniques affect training
- The impact of various regularization methods
- Best practices in software design for machine learning

## Contributing??

Feel free to fork this repository and submit pull requests. I'm always open to suggestions and improvements!

## What's Next? 

This is just the beginning. My ultimate aim with RIIK is to go beyond basic neural networks and push toward more advanced deep learning architectures. 

Here's what I'm planning next:

- Implement CNNs, RNNs, Transformers, and Attention mechanisms
- Build an autograd engine to support dynamic computation graphs
- Optimize for scalability and efficiency to handle large-scale training tasks
- Work toward building a research-grade engine capable of benchmarks near GPT-3–level models (within practical constraints) if possible. 

## 📜 License

MIT - Feel free to use this code for your own learning or projects!
