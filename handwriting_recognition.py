"""
handwriting_recognition.py
Main module for EMNIST handwriting recognition system
"""

import numpy as np
import scipy.io
from scipy.ndimage import rotate, shift
import pickle
from pathlib import Path


class NeuralNetwork:
    """A neural network implementation from scratch for EMNIST classification"""
    
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize the neural network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights with He initialization
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            self.activations.append(a)
        
        # Output layer with softmax
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        # Output layer gradient
        delta = output - y
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        """Update weights using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
        """
        Train the neural network
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot encoded)
            X_val: Validation data
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Mini-batch size
        """
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                output = self.forward(X_batch)
                gradients_w, gradients_b = self.backward(X_batch, y_batch, output)
                self.update_weights(gradients_w, gradients_b)
            
            # Calculate accuracy on validation set
            if (epoch + 1) % 5 == 0:
                train_acc = self.evaluate(X_train, y_train)
                val_acc = self.evaluate(X_val, y_val)
                print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        """Evaluate accuracy"""
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)
    
    def save(self, filepath):
        """Save model to file"""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases,
            'learning_rate': self.learning_rate
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['layer_sizes'], model_data['learning_rate'])
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        return model


class EMNISTDataLoader:
    """Load and preprocess EMNIST data"""
    
    def __init__(self, mat_file_path):
        """Load EMNIST .mat file"""
        self.data = scipy.io.loadmat(mat_file_path)
        self.class_mapping = self._create_class_mapping()
    
    def _create_class_mapping(self):
        """Create mapping from class indices to characters"""
        # EMNIST ByClass: 0-9 (digits), 10-35 (uppercase A-Z), 36-61 (lowercase a-z)
        mapping = {}
        for i in range(10):
            mapping[i] = str(i)
        for i in range(26):
            mapping[i + 10] = chr(65 + i)  # A-Z
        for i in range(26):
            mapping[i + 36] = chr(97 + i)  # a-z
        return mapping
    
    def load_data(self, augment=True):
        """
        Load and preprocess the EMNIST data
        
        Args:
            augment: Whether to apply data augmentation
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        # Extract data from .mat file
        X_train = self.data['dataset'][0][0][0][0][0][0].astype(np.float32)
        y_train = self.data['dataset'][0][0][0][0][0][1].astype(np.int32)
        X_test = self.data['dataset'][0][0][1][0][0][0].astype(np.float32)
        y_test = self.data['dataset'][0][0][1][0][0][1].astype(np.int32)
        
        # Reshape images (EMNIST images are stored transposed)
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        
        # Transpose and flip to correct orientation
        X_train = np.array([np.rot90(np.fliplr(img)) for img in X_train])
        X_test = np.array([np.rot90(np.fliplr(img)) for img in X_test])
        
        # Normalize pixel values
        X_train = X_train / 255.0
        X_test = X_test / 255.0
        
        # Flatten images
        X_train = X_train.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        # Data augmentation for training set
        if augment:
            print("Applying data augmentation...")
            X_train, y_train = self._augment_data(X_train.reshape(-1, 28, 28), y_train)
            X_train = X_train.reshape(-1, 784)
        
        # One-hot encode labels
        n_classes = len(self.class_mapping)
        y_train_onehot = self._one_hot_encode(y_train, n_classes)
        y_test_onehot = self._one_hot_encode(y_test, n_classes)
        
        return X_train, y_train_onehot, X_test, y_test_onehot
    
    def _augment_data(self, X, y, samples_per_image=2):
        """
        Apply data augmentation (rotation and shift)
        
        Args:
            X: Images (n_samples, 28, 28)
            y: Labels
            samples_per_image: Number of augmented samples per original image
        """
        augmented_X = [X]
        augmented_y = [y]
        
        # Only augment a subset to keep training time reasonable
        n_samples = min(len(X), 50000)
        indices = np.random.choice(len(X), n_samples, replace=False)
        
        for _ in range(samples_per_image):
            X_aug = []
            for idx in indices:
                img = X[idx]
                # Random rotation (-10 to 10 degrees)
                angle = np.random.uniform(-10, 10)
                img_rot = rotate(img, angle, reshape=False, mode='constant', cval=0)
                
                # Random shift (-2 to 2 pixels)
                shift_x = np.random.randint(-2, 3)
                shift_y = np.random.randint(-2, 3)
                img_shifted = shift(img_rot, [shift_y, shift_x], mode='constant', cval=0)
                
                X_aug.append(img_shifted)
            
            augmented_X.append(np.array(X_aug))
            augmented_y.append(y[indices])
        
        return np.concatenate(augmented_X), np.concatenate(augmented_y)
    
    def _one_hot_encode(self, y, n_classes):
        """Convert labels to one-hot encoding"""
        one_hot = np.zeros((len(y), n_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def get_character(self, class_idx):
        """Convert class index to character"""
        return self.class_mapping.get(class_idx, '?')


class HandwritingRecognizer:
    """Main class for handwriting recognition"""
    
    def __init__(self, model_path=None):
        """
        Initialize recognizer
        
        Args:
            model_path: Path to saved model file (optional)
        """
        self.model = None
        self.class_mapping = self._create_class_mapping()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def _create_class_mapping(self):
        """Create mapping from class indices to characters"""
        mapping = {}
        for i in range(10):
            mapping[i] = str(i)
        for i in range(26):
            mapping[i + 10] = chr(65 + i)
        for i in range(26):
            mapping[i + 36] = chr(97 + i)
        return mapping
    
    def train(self, mat_file_path, model_save_path='model.pkl'):
        """
        Train the model on EMNIST data
        
        Args:
            mat_file_path: Path to emnist-byclass.mat file
            model_save_path: Path to save trained model
        """
        print("Loading EMNIST data...")
        loader = EMNISTDataLoader(mat_file_path)
        X_train, y_train, X_test, y_test = loader.load_data(augment=True)
        
        # Split validation set from training data
        val_split = int(0.9 * len(X_train))
        X_val = X_train[val_split:]
        y_val = y_train[val_split:]
        X_train = X_train[:val_split]
        y_train = y_train[:val_split]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        # Create neural network (deeper network for better accuracy)
        print("\nInitializing neural network...")
        self.model = NeuralNetwork(
            layer_sizes=[784, 512, 256, 128, 62],  # 62 classes in EMNIST ByClass
            learning_rate=0.01
        )
        
        # Train the model
        print("\nTraining model...")
        self.model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=128)
        
        # Evaluate on test set
        test_acc = self.model.evaluate(X_test, y_test)
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        
        # Save model
        self.model.save(model_save_path)
        print(f"\nModel saved to {model_save_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        self.model = NeuralNetwork.load(model_path)
        print(f"Model loaded from {model_path}")
    
    def preprocess_image(self, image_array):
        """
        Preprocess image for prediction
        
        Args:
            image_array: 28x28 numpy array
            
        Returns:
            Preprocessed image ready for prediction
        """
        # Ensure correct shape
        if image_array.shape != (28, 28):
            raise ValueError("Image must be 28x28 pixels")
        
        # Normalize
        image_array = image_array.astype(np.float32) / 255.0
        
        # Flatten
        image_array = image_array.reshape(1, 784)
        
        return image_array
    
    def predict(self, image_array):
        """
        Predict character from image
        
        Args:
            image_array: 28x28 numpy array (grayscale)
            
        Returns:
            Predicted character and confidence scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Preprocess
        processed_image = self.preprocess_image(image_array)
        
        # Get prediction
        output = self.model.forward(processed_image)
        class_idx = np.argmax(output[0])
        confidence = output[0][class_idx]
        
        character = self.class_mapping[class_idx]
        
        # Get top 5 predictions
        top5_indices = np.argsort(output[0])[-5:][::-1]
        top5_predictions = [(self.class_mapping[idx], output[0][idx]) for idx in top5_indices]
        
        return {
            'character': character,
            'confidence': float(confidence),
            'top5': top5_predictions
        }


if __name__ == "__main__":
    # Training phase
    recognizer = HandwritingRecognizer()
    
    # Train the model (this will take some time)
    print("Starting training...")
    recognizer.train('emnist-byclass.mat', 'emnist_model.pkl')
    
    # Prediction phase (after training or loading existing model)
    # recognizer = HandwritingRecognizer('emnist_model.pkl')
    
    # Example: Load and predict on an image
    # from PIL import Image
    # img = Image.open('test_image.png').convert('L')
    # img_array = np.array(img)
    # result = recognizer.predict(img_array)
    # print(f"Predicted: {result['character']} (confidence: {result['confidence']:.2%})")