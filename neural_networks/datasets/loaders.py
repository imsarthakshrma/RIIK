import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union, Any
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_california_housing, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class DataLoader:
    """Data loading and preprocessing utilities for neural networks"""
    
    @staticmethod
    def load_classification_dataset(dataset_name: str = 'breast_cancer', test_size: float = 0.2, 
                                  val_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]:
        """Load and preprocess classification datasets
        
        Args:
            dataset_name: Name of dataset ('breast_cancer', 'synthetic_binary', 'synthetic_multiclass')
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
        """
        if dataset_name == 'breast_cancer':
            # Load breast cancer dataset
            data = load_breast_cancer()
            X, y = data.data, data.target.reshape(-1, 1)
            feature_names = data.feature_names
            
        elif dataset_name == 'synthetic_binary':
            # Generate synthetic binary classification dataset
            X, y = make_classification(
                n_samples=2000, n_features=20, n_informative=15, 
                n_redundant=5, n_classes=2, random_state=random_state
            )
            y = y.reshape(-1, 1)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        elif dataset_name == 'synthetic_multiclass':
            # Generate synthetic multi-class dataset
            X, y = make_classification(
                n_samples=5000, n_features=20, n_informative=15, 
                n_redundant=5, n_classes=5, random_state=random_state
            )
            # One-hot encode labels for multi-class
            encoder = OneHotEncoder(sparse=False)
            y = encoder.fit_transform(y.reshape(-1, 1))
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Split train into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
    
    @staticmethod
    def load_regression_dataset(dataset_name: str = 'california_housing', test_size: float = 0.2, 
                              val_size: float = 0.2, random_state: int = 42) -> Dict[str, np.ndarray]:
        """Load and preprocess regression datasets
        
        Args:
            dataset_name: Name of dataset ('california_housing', 'diabetes')
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
        """
        if dataset_name == 'california_housing':
            # Load California housing dataset
            data = fetch_california_housing()
            X, y = data.data, data.target.reshape(-1, 1)
            feature_names = data.feature_names
            
        elif dataset_name == 'diabetes':
            # Load diabetes dataset
            data = load_diabetes()
            X, y = data.data, data.target.reshape(-1, 1)
            feature_names = data.feature_names
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Scale target for better training
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y)
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=random_state
        )
        
        # Split train into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
    
    @staticmethod
    def load_from_csv(filepath: str, target_column: str, test_size: float = 0.2, 
                     val_size: float = 0.2, random_state: int = 42, 
                     task: str = 'classification') -> Dict[str, np.ndarray]:
        """Load dataset from CSV file
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            random_state: Random seed for reproducibility
            task: 'classification' or 'regression'
            
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Load data
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values.reshape(-1, 1)
        feature_names = df.drop(columns=[target_column]).columns.tolist()
        
        # Scale features
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        
        # Handle target based on task
        if task == 'classification':
            # Check if multi-class (more than 2 unique values)
            if len(np.unique(y)) > 2:
                # One-hot encode labels for multi-class
                encoder = OneHotEncoder(sparse=False)
                y = encoder.fit_transform(y)
                scaler_y = None
            else:
                # Binary classification
                scaler_y = None
        else:  # regression
            # Scale target for regression
            scaler_y = StandardScaler()
            y = scaler_y.fit_transform(y)
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Split train into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=random_state
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y
        }
    
    @staticmethod
    def create_batches(X: np.ndarray, y: np.ndarray, batch_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create mini-batches from data
        
        Args:
            X: Input features
            y: Target values
            batch_size: Size of each batch
            
        Returns:
            List of (X_batch, y_batch) tuples
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        batches = []
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]
            batches.append((X_batch, y_batch))
            
        return batches