import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Import neural network components
import sys
import os

# Add the parent directory to the path so we can import the neural_networks package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  
from neural_networks.core import NeuralNetwork


def run_real_world_examples():
    """Demonstrate on real-world datasets"""
    print("=== PRODUCTION NEURAL NETWORK - REAL WORLD EXAMPLES ===\n")
    
    # Example 1: Breast Cancer Classification (Binary)
    print("1. BREAST CANCER CLASSIFICATION (Binary)")
    print("-" * 50)
    
    # Load and preprocess data
    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Define architecture
    architecture = [
        (30, 64, 'relu', 0.3, True),    # Input layer with dropout and batch norm
        (64, 32, 'relu', 0.3, True),    # Hidden layer
        (32, 16, 'relu', 0.2, False),   # Hidden layer
        (16, 1, 'sigmoid', 0.0, False)  # Output layer
    ]
    
    # Create and train model
    model = NeuralNetwork(architecture, task='classification', optimizer='adam', learning_rate=0.001)
    
    training_info = model.fit(
        X_train, y_train, X_val, y_val,
        epochs=500, batch_size=32,
        l2_reg=0.001, early_stopping=True,
        patience=30, lr_schedule=True
    )
    
    # Evaluate
    results = model.evaluate(X_test, y_test)
    model.plot_history()
    
    # Save the classification model
    model.save_model('breast_cancer_model.pkl')
    
    # Example 2: California Housing (Regression)
    print("\n2. CALIFORNIA HOUSING PREDICTION (Regression)")
    print("-" * 50)
    
    # Load and preprocess data
    housing = fetch_california_housing()
    X_reg, y_reg = housing.data, housing.target.reshape(-1, 1)
    
    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_reg_scaled = scaler_X.fit_transform(X_reg)
    y_reg_scaled = scaler_y.fit_transform(y_reg)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_scaled, y_reg_scaled, test_size=0.2, random_state=42
    )
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
        X_train_r, y_train_r, test_size=0.2, random_state=42
    )
    
    # Architecture for regression
    architecture_reg = [
        (8, 128, 'relu', 0.3, True),     # Input layer
        (128, 64, 'relu', 0.3, True),    # Hidden layer
        (64, 32, 'relu', 0.2, True),     # Hidden layer
        (32, 16, 'leaky_relu', 0.1, False),  # Hidden layer
        (16, 1, 'linear', 0.0, False)    # Output layer
    ]
    
    # Create and train regression model
    model_reg = NeuralNetwork(architecture_reg, task='regression', optimizer='adam', learning_rate=0.001)
    
    training_info_reg = model_reg.fit(
        X_train_r, y_train_r, X_val_r, y_val_r,
        epochs=1000, batch_size=64,
        l1_reg=0.0001, l2_reg=0.001,
        early_stopping=True, patience=50,
        lr_schedule=True, lr_decay=0.95
    )
    
    # Evaluate regression model
    results_reg = model_reg.evaluate(X_test_r, y_test_r)
    model_reg.plot_history()
    
    # Transform predictions back to original scale for interpretation
    y_pred_scaled = model_reg.predict(X_test_r)
    y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
    y_test_original = scaler_y.inverse_transform(y_test_r)
    
    # Calculate RMSE in original scale
    rmse = np.sqrt(np.mean((y_test_original - y_pred_original)**2))
    print(f"RMSE in original scale: ${rmse*100000:.0f}")
    
    # Example 3: Multi-class Classification
    print("\n3. MULTI-CLASS CLASSIFICATION (Iris-like synthetic data)")
    print("-" * 50)
    
    # Generate multi-class dataset
    from sklearn.datasets import make_classification
    X_multi, y_multi = make_classification(
        n_samples=5000, n_features=20, n_informative=15, 
        n_redundant=5, n_classes=5, n_clusters_per_class=1, random_state=42
    )
    
    # One-hot encode labels
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    y_multi_encoded = encoder.fit_transform(y_multi.reshape(-1, 1))
    
    # Scale features
    scaler_multi = StandardScaler()
    X_multi_scaled = scaler_multi.fit_transform(X_multi)
    
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi_scaled, y_multi_encoded, test_size=0.2, random_state=42
    )
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(
        X_train_m, y_train_m, test_size=0.2, random_state=42
    )
    
    # Architecture for multi-class classification
    architecture_multi = [
        (20, 128, 'relu', 0.4, True),    # Input layer
        (128, 64, 'relu', 0.3, True),    # Hidden layer
        (64, 32, 'relu', 0.2, True),     # Hidden layer
        (32, 5, 'softmax', 0.0, False)   # Output layer (5 classes)
    ]
    
    # Create and train multi-class model
    model_multi = NeuralNetwork(architecture_multi, task='classification', optimizer='adam', learning_rate=0.001)
    
    training_info_multi = model_multi.fit(
        X_train_m, y_train_m, X_val_m, y_val_m,
        epochs=300, batch_size=64,
        l2_reg=0.01, early_stopping=True,
        patience=25, lr_schedule=True
    )
    
    # Evaluate multi-class model
    results_multi = model_multi.evaluate(X_test_m, y_test_m)
    model_multi.plot_history()
    
    return model, model_reg, model_multi