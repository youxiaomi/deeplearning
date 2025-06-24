"""
Multi-Layer Perceptron from Scratch
从零开始的多层感知机

This module implements a complete MLP with backpropagation using only NumPy.
本模块仅使用NumPy实现带反向传播的完整MLP。

Mathematical Foundation:
数学基础：

Forward Pass (前向传播):
z^(l) = W^(l) * a^(l-1) + b^(l)
a^(l) = f(z^(l))

Backward Pass (反向传播):
δ^(L) = ∇_a C ⊙ f'(z^(L))
δ^(l) = ((W^(l+1))^T δ^(l+1)) ⊙ f'(z^(l))

Weight Updates (权重更新):
W^(l) := W^(l) - η * δ^(l) * (a^(l-1))^T
b^(l) := b^(l) - η * δ^(l)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import pickle
import os


class ActivationFunction:
    """
    Activation Function Base Class
    激活函数基类
    """
    
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        """Forward pass"""
        raise NotImplementedError
    
    @staticmethod
    def backward(z: np.ndarray) -> np.ndarray:
        """Backward pass (derivative)"""
        raise NotImplementedError


class ReLU(ActivationFunction):
    """
    ReLU Activation Function
    ReLU激活函数
    
    f(x) = max(0, x)
    f'(x) = 1 if x > 0, 0 otherwise
    """
    
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        """ReLU forward pass: f(z) = max(0, z)"""
        return np.maximum(0, z)
    
    @staticmethod
    def backward(z: np.ndarray) -> np.ndarray:
        """ReLU derivative: f'(z) = 1 if z > 0, 0 otherwise"""
        return (z > 0).astype(float)


class Sigmoid(ActivationFunction):
    """
    Sigmoid Activation Function
    Sigmoid激活函数
    
    f(x) = 1 / (1 + e^(-x))
    f'(x) = f(x) * (1 - f(x))
    """
    
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        """Sigmoid forward: f(z) = 1 / (1 + exp(-z))"""
        # Clip z to prevent overflow (裁剪z以防止溢出)
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))
    
    @staticmethod
    def backward(z: np.ndarray) -> np.ndarray:
        """Sigmoid derivative: f'(z) = f(z) * (1 - f(z))"""
        s = Sigmoid.forward(z)
        return s * (1 - s)


class Softmax:
    """
    Softmax Activation Function (for output layer)
    Softmax激活函数（用于输出层）
    
    f(z_i) = e^(z_i) / Σ_j e^(z_j)
    """
    
    @staticmethod
    def forward(z: np.ndarray) -> np.ndarray:
        """
        Softmax forward pass
        Softmax前向传播
        
        Args:
            z: Input logits (batch_size, num_classes)
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability (减去最大值以提高数值稳定性)
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class CrossEntropyLoss:
    """
    Cross-Entropy Loss Function
    交叉熵损失函数
    
    L = -Σ_i t_i * log(y_i)
    ∂L/∂z = y - t (when combined with Softmax)
    """
    
    @staticmethod
    def forward(predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss
        计算交叉熵损失
        
        Args:
            predictions: Model predictions (batch_size, num_classes)
            targets: One-hot encoded targets (batch_size, num_classes)
        Returns:
            Average loss
        """
        # Add small epsilon to prevent log(0) (添加小的epsilon防止log(0))
        epsilon = 1e-15
        predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
        
        # Compute cross-entropy (计算交叉熵)
        loss = -np.sum(targets * np.log(predictions_clipped)) / predictions.shape[0]
        return loss
    
    @staticmethod
    def backward(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute cross-entropy gradient (combined with Softmax)
        计算交叉熵梯度（与Softmax结合）
        
        Returns:
            Gradient: y - t
        """
        return (predictions - targets) / predictions.shape[0]


class Layer:
    """
    Fully Connected Layer
    全连接层
    """
    
    def __init__(self, input_size: int, output_size: int, activation: ActivationFunction):
        """
        Initialize layer
        初始化层
        
        Args:
            input_size: Number of input features (输入特征数)
            output_size: Number of output features (输出特征数)
            activation: Activation function (激活函数)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases (初始化权重和偏置)
        # Xavier/Glorot initialization for better convergence
        # Xavier/Glorot初始化以获得更好的收敛性
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))
        
        # Cache for backward pass (反向传播缓存)
        self.last_input = None
        self.last_z = None
        self.last_activation = None
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer
        层的前向传播
        
        Args:
            inputs: Input data (batch_size, input_size)
        Returns:
            Layer output (batch_size, output_size)
        """
        # Cache input for backward pass (缓存输入用于反向传播)
        self.last_input = inputs
        
        # Linear transformation: z = W^T * x + b
        # 线性变换: z = W^T * x + b
        self.last_z = np.dot(inputs, self.weights) + self.biases
        
        # Apply activation function: a = f(z)
        # 应用激活函数: a = f(z)
        self.last_activation = self.activation.forward(self.last_z)
        
        return self.last_activation
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer
        层的反向传播
        
        Args:
            grad_output: Gradient from next layer (batch_size, output_size)
        Returns:
            Gradient w.r.t. input (batch_size, input_size)
        """
        # Compute activation derivative (计算激活函数导数)
        activation_grad = self.activation.backward(self.last_z)
        
        # Element-wise product with activation derivative
        # 与激活函数导数逐元素相乘
        delta = grad_output * activation_grad
        
        # Compute gradients (计算梯度)
        self.grad_weights = np.dot(self.last_input.T, delta)
        self.grad_biases = np.sum(delta, axis=0, keepdims=True)
        
        # Compute gradient w.r.t. input (计算关于输入的梯度)
        grad_input = np.dot(delta, self.weights.T)
        
        return grad_input
    
    def update_parameters(self, learning_rate: float):
        """
        Update layer parameters using gradients
        使用梯度更新层参数
        
        Args:
            learning_rate: Learning rate for gradient descent
        """
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases


class MLPClassifier:
    """
    Multi-Layer Perceptron Classifier
    多层感知机分类器
    """
    
    def __init__(self, layer_sizes: List[int], activations: List[str] = None):
        """
        Initialize MLP
        初始化MLP
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: List of activation functions for each hidden layer
        """
        self.layer_sizes = layer_sizes
        self.num_classes = layer_sizes[-1]
        
        # Default activations (默认激活函数)
        if activations is None:
            activations = ['relu'] * (len(layer_sizes) - 2)  # ReLU for hidden layers
        
        # Create layers (创建层)
        self.layers = []
        activation_map = {
            'relu': ReLU(),
            'sigmoid': Sigmoid()
        }
        
        for i in range(len(layer_sizes) - 1):
            if i < len(layer_sizes) - 2:  # Hidden layers
                activation = activation_map[activations[i]]
            else:  # Output layer (no activation, will use Softmax in loss)
                activation = ReLU()  # Placeholder, not used for output
            
            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)
        
        # Loss function (损失函数)
        self.loss_fn = CrossEntropyLoss()
        
        # Training history (训练历史)
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network
        整个网络的前向传播
        
        Args:
            X: Input data (batch_size, input_features)
        Returns:
            Network output (batch_size, num_classes)
        """
        current_input = X
        
        # Pass through all layers except the last
        # 通过除最后一层外的所有层
        for layer in self.layers[:-1]:
            current_input = layer.forward(current_input)
        
        # Last layer (output layer)
        # 最后一层（输出层）
        logits = np.dot(current_input, self.layers[-1].weights) + self.layers[-1].biases
        
        # Cache for backward pass (缓存用于反向传播)
        self.layers[-1].last_input = current_input
        self.layers[-1].last_z = logits
        
        # Apply Softmax to get probabilities (应用Softmax获得概率)
        probabilities = Softmax.forward(logits)
        
        return probabilities
    
    def backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backward pass through the entire network
        整个网络的反向传播
        
        Args:
            X: Input data
            y_true: True labels (one-hot encoded)
            y_pred: Predicted probabilities
        """
        # Start with loss gradient (从损失梯度开始)
        grad_output = self.loss_fn.backward(y_pred, y_true)
        
        # Backward through output layer (输出层反向传播)
        # For output layer, gradient is directly from loss
        # 对于输出层，梯度直接来自损失
        output_layer = self.layers[-1]
        output_layer.grad_weights = np.dot(output_layer.last_input.T, grad_output)
        output_layer.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input of output layer
        # 输出层输入的梯度
        grad_input = np.dot(grad_output, output_layer.weights.T)
        
        # Backward through hidden layers (隐藏层反向传播)
        for layer in reversed(self.layers[:-1]):
            grad_input = layer.backward(grad_input)
    
    def update_parameters(self, learning_rate: float):
        """
        Update all layer parameters
        更新所有层参数
        """
        for layer in self.layers:
            layer.update_parameters(learning_rate)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        进行预测
        
        Args:
            X: Input data
        Returns:
            Predicted class labels
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        获取预测概率
        
        Args:
            X: Input data
        Returns:
            Class probabilities
        """
        return self.forward(X)
    
    def compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy
        计算分类准确率
        
        Args:
            X: Input data
            y: True labels (class indices)
        Returns:
            Accuracy (0-1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def compute_loss(self, X: np.ndarray, y_onehot: np.ndarray) -> float:
        """
        Compute loss
        计算损失
        
        Args:
            X: Input data
            y_onehot: One-hot encoded labels
        Returns:
            Loss value
        """
        predictions = self.forward(X)
        return self.loss_fn.forward(predictions, y_onehot)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01,
            verbose: bool = True) -> Dict:
        """
        Train the MLP
        训练MLP
        
        Args:
            X_train: Training data
            y_train: Training labels (class indices)
            X_val: Validation data (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            learning_rate: Learning rate
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        print(f"Training MLP with architecture: {self.layer_sizes}")
        print(f"训练MLP，架构为: {self.layer_sizes}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        print("-" * 60)
        
        # Convert labels to one-hot encoding (将标签转换为独热编码)
        y_train_onehot = self._to_onehot(y_train)
        if y_val is not None:
            y_val_onehot = self._to_onehot(y_val)
        
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle training data (打乱训练数据)
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_onehot_shuffled = y_train_onehot[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training (小批量训练)
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                y_batch = y_train_onehot_shuffled[i:batch_end]
                
                # Forward pass (前向传播)
                y_pred = self.forward(X_batch)
                
                # Compute loss (计算损失)
                batch_loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass (反向传播)
                self.backward(X_batch, y_batch, y_pred)
                
                # Update parameters (更新参数)
                self.update_parameters(learning_rate)
            
            # Compute epoch metrics (计算epoch指标)
            avg_epoch_loss = epoch_loss / num_batches
            train_accuracy = self.compute_accuracy(X_train, y_train)
            
            # Store training metrics (存储训练指标)
            self.train_losses.append(avg_epoch_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validation metrics (验证指标)
            if X_val is not None:
                val_loss = self.compute_loss(X_val, y_val_onehot)
                val_accuracy = self.compute_accuracy(X_val, y_val)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_accuracy)
            
            # Print progress (打印进度)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs}: ", end="")
                print(f"Loss: {avg_epoch_loss:.4f}, Train Acc: {train_accuracy:.4f}", end="")
                if X_val is not None:
                    print(f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print()
        
        print("✅ Training completed! (训练完成！)")
        
        # Return training history (返回训练历史)
        history = {
            'train_loss': self.train_losses,
            'train_accuracy': self.train_accuracies,
            'val_loss': self.val_losses if X_val is not None else [],
            'val_accuracy': self.val_accuracies if X_val is not None else []
        }
        
        return history
    
    def _to_onehot(self, y: np.ndarray) -> np.ndarray:
        """
        Convert class indices to one-hot encoding
        将类别索引转换为独热编码
        """
        onehot = np.zeros((y.shape[0], self.num_classes))
        onehot[np.arange(y.shape[0]), y] = 1
        return onehot
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        绘制训练历史
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss (绘制损失)
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training Loss (训练损失)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy (绘制准确率)
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        if self.val_accuracies:
            ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training Accuracy (训练准确率)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save trained model
        保存训练好的模型
        """
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': [layer.weights for layer in self.layers],
            'biases': [layer.biases for layer in self.layers],
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to: {filepath}")
        print(f"✅ 模型保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model
        加载训练好的模型
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.layer_sizes = model_data['layer_sizes']
        
        # Restore weights and biases (恢复权重和偏置)
        for i, layer in enumerate(self.layers):
            layer.weights = model_data['weights'][i]
            layer.biases = model_data['biases'][i]
        
        # Restore training history (恢复训练历史)
        self.train_losses = model_data['train_losses']
        self.train_accuracies = model_data['train_accuracies']
        self.val_losses = model_data['val_losses']
        self.val_accuracies = model_data['val_accuracies']
        
        print(f"✅ Model loaded from: {filepath}")
        print(f"✅ 模型从以下位置加载: {filepath}")


def main():
    """
    Demo function for MLP from scratch
    MLP从零实现的演示函数
    """
    print("MLP from Scratch Demo (从零开始MLP演示)")
    print("=" * 50)
    
    # Create synthetic data for testing (创建合成数据进行测试)
    np.random.seed(42)
    
    # Generate sample data (生成样本数据)
    n_samples = 1000
    n_features = 20
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Split data (分割数据)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create and train model (创建并训练模型)
    model = MLPClassifier(
        layer_sizes=[n_features, 64, 32, n_classes],
        activations=['relu', 'relu']
    )
    
    # Train model (训练模型)
    history = model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32,
        learning_rate=0.01,
        verbose=True
    )
    
    # Evaluate model (评估模型)
    train_acc = model.compute_accuracy(X_train, y_train)
    test_acc = model.compute_accuracy(X_test, y_test)
    
    print(f"\nFinal Results (最终结果):")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Plot training history (绘制训练历史)
    model.plot_training_history()
    
    print("✅ Demo completed! (演示完成！)")


if __name__ == "__main__":
    main() 