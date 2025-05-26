# RIIK Neural Network from Scratch

A high-performance neural network implementation built from scratch in Python with NumPy. This implementation includes advanced features for real-world production use.

## Features

- ✅ Advanced Optimizers (Adam, SGD with Momentum)
- ✅ Batch Normalization for stable training
- ✅ Dropout for regularization
- ✅ L1/L2 Regularization
- ✅ Early Stopping with patience
- ✅ Learning Rate Scheduling
- ✅ Mini-batch Training
- ✅ Model Saving/Loading
- ✅ Comprehensive Evaluation Metrics
- ✅ Multiple Activation Functions
- ✅ Binary & Multi-class Classification
- ✅ Regression Support
- ✅ Numerical Stability
- ✅ Training History Tracking
- ✅ Real-world Dataset Compatibility
- ✅ Professional Code Structure
- ✅ Production-ready Performance

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
neural_networks/
├── core/
│   ├── __init__.py         # Package exports
│   ├── network.py          # Neural Network implementation
│   ├── layer.py            # Layer implementation
│   ├── activation.py       # Activation functions
│   └── optimizers.py       # Optimization algorithms
├── datasets/
│   ├── __init__.py
│   └── loaders.py          # Data loading utilities
└── utils/                  # Additional utilities
```

## Quick Start

```python
from neural_networks.core import NeuralNetwork
from neural_networks.datasets.loaders import DataLoader

# Load dataset
data = DataLoader.load_classification_dataset('breast_cancer')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Define architecture
architecture = [
    (30, 64, 'relu', 0.3, True),    # Input layer with dropout and batch norm
    (64, 32, 'relu', 0.3, True),    # Hidden layer
    (32, 16, 'relu', 0.2, False),   # Hidden layer
    (16, 1, 'sigmoid', 0.0, False)  # Output layer
]

# Create and train model
model = NeuralNetwork(architecture, task='classification', optimizer='adam', learning_rate=0.001)

model.fit(
    X_train, y_train, X_val, y_val,
    epochs=500, batch_size=32,
    l2_reg=0.001, early_stopping=True,
    patience=30, lr_schedule=True
)

# Evaluate
results = model.evaluate(X_test, y_test)
model.plot_history()

# Save model
model.save_model('breast_cancer_model.pkl')

# Load model
loaded_model = NeuralNetwork.load_model('breast_cancer_model.pkl')
```

## Examples

See the `examples/` directory for complete examples:

- Binary classification (Breast Cancer dataset)
- Regression (California Housing dataset)
- Multi-class classification (Synthetic dataset)

## Performance

This implementation has been benchmarked against scikit-learn's MLPClassifier and MLPRegressor, showing comparable accuracy with more flexibility and additional features.

## License

MIT
