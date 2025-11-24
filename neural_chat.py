import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import re
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, fetch_california_housing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Import our neural network
from neural_networks.core import NeuralNetwork
from neural_networks.datasets.loaders import DataLoader

class NeuralNetworkChatbot:
    def __init__(self):
        self.model = None
        self.data = None
        self.trained = False
        self.dataset_name = None
        self.task_type = None
        self.history = []
        self.available_datasets = {
            'iris': 'Multi-class classification (3 classes of flowers)',
            'breast_cancer': 'Binary classification (malignant/benign)',
            'diabetes': 'Regression (disease progression)',
            'california_housing': 'Regression (housing prices)',
            'xor': 'Binary classification (XOR logical operation)'
        }
        
        # Welcome message
        print("\n" + "="*60)
        print("🧠 Welcome to NumpyNet - Neural Network Chat Interface 🧠")
        print("="*60)
        print("Type 'help' to see available commands")
        print("="*60 + "\n")
    
    def process_command(self, command):
        """Process user commands"""
        command = command.strip().lower()
        
        # Add command to history
        self.history.append(command)
        
        # Help command
        if command == 'help':
            self.show_help()
            
        # List available datasets
        elif command == 'list datasets':
            self.list_datasets()
            
        # Train on a dataset
        elif command.startswith('train on '):
            dataset = command[9:].strip()
            self.train_on_dataset(dataset)
            
        # Create custom architecture
        elif command.startswith('create architecture'):
            self.create_architecture(command)
            
        # Make predictions
        elif command.startswith('predict'):
            self.make_prediction(command)
            
        # Evaluate model
        elif command == 'evaluate':
            self.evaluate_model()
            
        # Show model details
        elif command == 'model info':
            self.show_model_info()
            
        # Plot training history
        elif command == 'plot history':
            self.plot_history()
            
        # Save model
        elif command.startswith('save model'):
            parts = command.split(' ')
            filename = 'model.pkl' if len(parts) <= 2 else parts[2]
            self.save_model(filename)
            
        # Load model
        elif command.startswith('load model'):
            parts = command.split(' ')
            filename = parts[2] if len(parts) > 2 else None
            if filename:
                self.load_model(filename)
            else:
                print("❌ Please specify a filename: load model <filename>")
                
        # Run XOR example
        elif command == 'run xor example':
            self.run_xor_example()
            
        # Exit
        elif command in ['exit', 'quit', 'bye']:
            print("👋 Thank you for using NumpyNet! Goodbye!")
            return False
            
        # Unknown command
        else:
            print(f"❓ Unknown command: '{command}'. Type 'help' to see available commands.")
        
        return True
    
    def show_help(self):
        """Show available commands"""
        print("\n📚 Available Commands:")
        print("  help                    - Show this help message")
        print("  list datasets           - Show available datasets")
        print("  train on <dataset>      - Train on a specific dataset")
        print("  create architecture     - Create a custom network architecture")
        print("  predict <data>          - Make predictions with trained model")
        print("  evaluate                - Evaluate model on test data")
        print("  model info              - Show information about the current model")
        print("  plot history            - Plot training history")
        print("  save model <filename>   - Save the current model")
        print("  load model <filename>   - Load a saved model")
        print("  run xor example         - Run the XOR example")
        print("  exit/quit/bye           - Exit the program\n")
    
    def list_datasets(self):
        """List available datasets"""
        print("\n📊 Available Datasets:")
        for name, description in self.available_datasets.items():
            print(f"  • {name}: {description}")
        print()
    
    def train_on_dataset(self, dataset_name):
        """Train on a specific dataset"""
        if dataset_name not in self.available_datasets:
            print(f"❌ Dataset '{dataset_name}' not found. Use 'list datasets' to see available options.")
            return
            
        print(f"🔄 Loading {dataset_name} dataset...")
        
        # Load the dataset
        if dataset_name == 'iris':
            iris = load_iris()
            X = iris.data
            encoder = OneHotEncoder(sparse=False)
            y = encoder.fit_transform(iris.target.reshape(-1, 1))
            self.task_type = 'classification'
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create architecture
            architecture = [
                (4, 10, 'relu', 0.2, True),
                (10, 3, 'softmax')
            ]
            
        elif dataset_name == 'breast_cancer':
            data = DataLoader.load_classification_dataset('breast_cancer')
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            self.task_type = 'classification'
            
            # Create architecture
            architecture = [
                (30, 16, 'relu', 0.3, True),
                (16, 8, 'relu', 0.2, False),
                (8, 1, 'sigmoid')
            ]
            
        elif dataset_name == 'diabetes':
            data = DataLoader.load_regression_dataset('diabetes')
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            self.task_type = 'regression'
            
            # Create architecture
            architecture = [
                (10, 32, 'relu', 0.2, True),
                (32, 16, 'relu', 0.1, True),
                (16, 1, 'linear')
            ]
            
        elif dataset_name == 'california_housing':
            data = DataLoader.load_regression_dataset('california_housing')
            X_train, y_train = data['X_train'], data['y_train']
            X_test, y_test = data['X_test'], data['y_test']
            self.task_type = 'regression'
            
            # Create architecture
            architecture = [
                (8, 64, 'relu', 0.3, True),
                (64, 32, 'relu', 0.2, True),
                (32, 1, 'linear')
            ]
            
        elif dataset_name == 'xor':
            # XOR problem
            X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            y = np.array([[0], [1], [1], [0]])
            
            # No need to split for XOR
            X_train, X_test, y_train, y_test = X, X, y, y
            self.task_type = 'classification'
            
            # Create architecture
            architecture = [
                (2, 4, 'relu'),
                (4, 1, 'sigmoid')
            ]
        
        # Store dataset info
        self.data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        self.dataset_name = dataset_name
        
        # Create and train model
        print(f"Creating neural network with architecture: {architecture}")
        self.model = NeuralNetwork(architecture, task=self.task_type)
        
        # Ask for training parameters
        epochs = int(input("Enter number of epochs (default: 300): ") or 300)
        batch_size = int(input("Enter batch size (default: 32): ") or 32)
        learning_rate = float(input("Enter learning rate (default: 0.001): ") or 0.001)
        
        # Set optimizer with custom learning rate
        if input("Use Adam optimizer? (y/n, default: y): ").lower() != 'n':
            self.model.optimizer = self.model.optimizer.__class__(learning_rate)
        else:
            self.model.optimizer = self.model.optimizer.__class__(learning_rate)
        
        print(f"\n🚀 Training model on {dataset_name} dataset...")
        print(f"   Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            l2_reg=0.001,
            early_stopping=True,
            patience=50,
            verbose=True
        )
        
        self.trained = True
        print(f" Training completed! Use 'evaluate' to test the model.")
    
    def create_architecture(self, command):
        """Create a custom architecture"""
        print("\n🏗️ Let's create a custom neural network architecture!")
        
        # Get input size
        input_size = int(input("Enter input size (number of features): "))
        
        # Get task type
        task = input("Enter task type (classification/regression): ").lower()
        if task not in ['classification', 'regression']:
            print(" Invalid task type. Using classification as default.")
            task = 'classification'
        
        self.task_type = task
        
        # Get number of layers
        num_layers = int(input("Enter number of layers (including output layer): "))
        
        architecture = []
        prev_size = input_size
        
        # Get layer details
        for i in range(num_layers):
            is_output = (i == num_layers - 1)
            
            # For output layer, determine size based on task
            if is_output:
                if task == 'classification':
                    output_classes = int(input("Enter number of output classes (1 for binary, >1 for multi-class): "))
                    size = output_classes if output_classes > 1 else 1
                    activation = 'sigmoid' if output_classes == 1 else 'softmax'
                else:  # regression
                    size = 1
                    activation = 'linear'
            else:
                # Hidden layer
                size = int(input(f"Enter size for hidden layer {i+1}: "))
                activation = input(f"Enter activation for layer {i+1} (relu/leaky_relu/tanh/sigmoid): ") or 'relu'
            
            # Get dropout and batch norm for non-output layers
            if not is_output:
                dropout = float(input(f"Enter dropout rate for layer {i+1} (0-1, 0 for none): ") or 0)
                batch_norm = input(f"Use batch normalization for layer {i+1}? (y/n): ").lower() == 'y'
                
                # Create layer config
                layer_config = (prev_size, size, activation, dropout, batch_norm)
            else:
                # Output layer typically doesn't use dropout or batch norm
                layer_config = (prev_size, size, activation)
            
            architecture.append(layer_config)
            prev_size = size
        
        # Create the model
        print(f"\n Creating neural network with architecture: {architecture}")
        self.model = NeuralNetwork(architecture, task=self.task_type)
        
        print("✅ Custom architecture created! Use 'train on <dataset>' to train the model.")
    
    def make_prediction(self, command):
        """Make predictions with the trained model"""
        if not self.trained or self.model is None:
            print(" No trained model available. Train a model first.")
            return
            
        if self.dataset_name == 'xor':
            # Special case for XOR
            print("\n🔮 Predictions for XOR:")
            inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
            for input_data in inputs:
                x = np.array([input_data])
                pred = self.model.predict(x)[0][0]
                print(f"  Input: {input_data}, Prediction: {pred:.4f}, Class: {1 if pred > 0.5 else 0}")
            return
            
        # For other datasets, allow custom input
        try:
            # Extract input data from command or ask for it
            if len(command) > 8 and '[' in command:
                # Parse input from command
                input_str = command[command.find('['):command.rfind(']')+1]
                input_data = eval(input_str)
                input_array = np.array([input_data])
            else:
                # Ask for input features
                if self.dataset_name == 'iris':
                    print("\nEnter 4 features (sepal length, sepal width, petal length, petal width):")
                    features = [float(input(f"Feature {i+1}: ")) for i in range(4)]
                    input_array = np.array([features])
                elif self.dataset_name == 'breast_cancer':
                    print(" Breast cancer dataset has 30 features. Use a sample from test data? (y/n)")
                    if input().lower() == 'y':
                        idx = np.random.randint(0, len(self.data['X_test']))
                        input_array = self.data['X_test'][idx:idx+1]
                        print(f"Using sample #{idx} from test data")
                    else:
                        print("Manual input for 30 features not supported in chat mode")
                        return
                else:
                    print(" This dataset has many features. Use a sample from test data? (y/n)")
                    if input().lower() == 'y':
                        idx = np.random.randint(0, len(self.data['X_test']))
                        input_array = self.data['X_test'][idx:idx+1]
                        print(f"Using sample #{idx} from test data")
                    else:
                        print(" Manual input for multiple features not supported in chat mode")
                        return
            
            # Make prediction
            prediction = self.model.predict(input_array)
            
            # Format and display result
            if self.task_type == 'classification':
                if prediction.shape[1] == 1:  # Binary classification
                    prob = prediction[0][0]
                    pred_class = 1 if prob > 0.5 else 0
                    print(f"\n Prediction: Class {pred_class} (Probability: {prob:.4f})")
                else:  # Multi-class
                    probs = prediction[0]
                    pred_class = np.argmax(probs)
                    print(f"\n Prediction: Class {pred_class}")
                    print(f"   Probabilities: {probs}")
                    
                    if self.dataset_name == 'iris':
                        classes = ['setosa', 'versicolor', 'virginica']
                        print(f"   Predicted flower: {classes[pred_class]}")
            else:  # Regression
                value = prediction[0][0]
                print(f"\n Predicted value: {value:.4f}")
                
                # If we have a scaler for the target, convert back to original scale
                if hasattr(self.data, 'scaler_y'):
                    original_value = self.data['scaler_y'].inverse_transform([[value]])[0][0]
                    print(f"   Original scale: {original_value:.4f}")
        
        except Exception as e:
            print(f" Error making prediction: {str(e)}")
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        if not self.trained or self.model is None:
            print(" No trained model available. Train a model first.")
            return
            
        print(f"\n Evaluating model on {self.dataset_name} test data...")
        
        # Evaluate the model
        results = self.model.evaluate(self.data['X_test'], self.data['y_test'])
        
        # Display additional metrics based on task type
        if self.task_type == 'classification':
            if self.model.layers[-1].output_size == 1:  # Binary classification
                y_pred = (self.model.predict(self.data['X_test']) > 0.5).astype(int)
                y_true = self.data['y_test'].astype(int)
                
                # Calculate metrics
                tp = np.sum((y_pred == 1) & (y_true == 1))
                tn = np.sum((y_pred == 0) & (y_true == 0))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                print(f"\n Additional Metrics:")
                print(f"   Precision: {precision:.4f}")
                print(f"   Recall: {recall:.4f}")
                print(f"   F1 Score: {f1:.4f}")
                print(f"   True Positives: {tp}, True Negatives: {tn}")
                print(f"   False Positives: {fp}, False Negatives: {fn}")
            else:
                # Multi-class metrics
                y_pred = np.argmax(self.model.predict(self.data['X_test']), axis=1)
                y_true = np.argmax(self.data['y_test'], axis=1)
                
                # Calculate accuracy
                accuracy = np.mean(y_pred == y_true)
                print(f"\n Additional Metrics:")
                print(f"   Accuracy: {accuracy:.4f}")
                
                # Class-wise accuracy
                for cls in range(self.model.layers[-1].output_size):
                    cls_acc = np.mean((y_pred == cls) == (y_true == cls))
                    print(f"   Class {cls} Accuracy: {cls_acc:.4f}")
        else:
            # Regression metrics
            y_pred = self.model.predict(self.data['X_test'])
            y_true = self.data['y_test']
            
            # Calculate metrics
            mse = np.mean((y_pred - y_true) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_pred - y_true))
            
            print(f"\n Additional Metrics:")
            print(f"   Mean Squared Error: {mse:.4f}")
            print(f"   Root Mean Squared Error: {rmse:.4f}")
            print(f"   Mean Absolute Error: {mae:.4f}")
    
    def show_model_info(self):
        """Show information about the current model"""
        if self.model is None:
            print(" No model created yet. Create or train a model first.")
            return
            
        print("\n Neural Network Information:")
        print(f"   Task Type: {self.task_type}")
        print(f"   Dataset: {self.dataset_name or 'None'}")
        print(f"   Trained: {'Yes' if self.trained else 'No'}")
        
        print("\n   Architecture:")
        for i, layer in enumerate(self.model.layers):
            layer_type = "Output" if i == len(self.model.layers) - 1 else "Hidden"
            print(f"   - Layer {i+1} ({layer_type}): {layer.input_size} → {layer.output_size}, Activation: {layer.activation}")
            if hasattr(layer, 'dropout_rate') and layer.dropout_rate > 0:
                print(f"     Dropout: {layer.dropout_rate}")
            if hasattr(layer, 'batch_norm') and layer.batch_norm:
                print(f"     Batch Normalization: Yes")
        
        print(f"\n   Optimizer: {self.model.optimizer.__class__.__name__}")
        print(f"   Learning Rate: {self.model.optimizer.lr}")
        
        if self.trained:
            print(f"\n   Training History:")
            print(f"   - Final Training Loss: {self.model.history['train_loss'][-1]:.6f}")
            if self.model.history['val_loss']:
                print(f"   - Final Validation Loss: {self.model.history['val_loss'][-1]:.6f}")
            print(f"   - Training Epochs: {len(self.model.history['train_loss'])}")
    
    def plot_history(self):
        """Plot training history"""
        if not self.trained or self.model is None:
            print(" No trained model available. Train a model first.")
            return
            
        print("\n Plotting training history...")
        self.model.plot_history()
    
    def save_model(self, filename):
        """Save the model to a file"""
        if not self.trained or self.model is None:
            print(" No trained model available. Train a model first.")
            return
            
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Add .pkl extension if not present
            if not filename.endswith('.pkl'):
                filename += '.pkl'
                
            filepath = os.path.join('models', filename)
            
            # Save the model
            self.model.save_model(filepath)
            
            print(f" Model saved to {filepath}")
        except Exception as e:
            print(f" Error saving model: {str(e)}")
    
    def load_model(self, filename):
        """Load a model from a file"""
        try:
            # Add .pkl extension if not present
            if not filename.endswith('.pkl'):
                filename += '.pkl'
                
            filepath = os.path.join('models', filename)
            
            # Check if file exists
            if not os.path.exists(filepath):
                print(f" Model file {filepath} not found.")
                return
                
            # Load the model
            self.model = NeuralNetwork.load_model(filepath)
            self.trained = True
            
            # Try to determine task type from model
            if self.model.layers[-1].activation == 'sigmoid' or self.model.layers[-1].activation == 'softmax':
                self.task_type = 'classification'
            else:
                self.task_type = 'regression'
                
            print(f" Model loaded from {filepath}")
            
            # Show model info
            self.show_model_info()
        except Exception as e:
            print(f" Error loading model: {str(e)}")
    
    def run_xor_example(self):
        """Run the XOR example"""
        print("\n Running XOR Example...")
        
        # XOR data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        
        # Create architecture
        architecture = [
            (2, 4, 'relu'),
            (4, 1, 'sigmoid')
        ]
        
        # Create and train model
        print(" Creating neural network for XOR problem")
        self.model = NeuralNetwork(architecture, task='classification')
        
        print(" Training model on XOR data...")
        self.model.fit(X, y, epochs=1000, batch_size=4, verbose=True)
        
        # Make predictions
        print("\n XOR Predictions:")
        for inputs in X:
            pred = self.model.predict(inputs.reshape(1, -1))[0][0]
            print(f"  Input: {inputs}, Target: {y[np.where((X == inputs).all(axis=1))[0][0]][0]}, Prediction: {pred:.4f}, Class: {1 if pred > 0.5 else 0}")
        
        # Store data and update state
        self.data = {'X_train': X, 'y_train': y, 'X_test': X, 'y_test': y}
        self.dataset_name = 'xor'
        self.task_type = 'classification'
        self.trained = True
        
        print("\n XOR example completed successfully!")

def main():
    chatbot = NeuralNetworkChatbot()
    
    while True:
        try:
            command = input("\n> ")
            if not chatbot.process_command(command):
                break
        except KeyboardInterrupt:
            print("\n Exiting...")
            break
        except Exception as e:
            print(f" Error: {str(e)}")

if __name__ == "__main__":
    main()