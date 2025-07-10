import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifierPyTorch(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classifier implemented in PyTorch.
    使用PyTorch实现的多层感知机（MLP）分类器。

    This model is designed for handwritten digit recognition, specifically for the MNIST dataset.
    该模型专为手写数字识别设计，特别是针对MNIST数据集。
    """
    def __init__(self, layer_sizes, activations):
        """
        Initializes the MLP model.
        初始化MLP模型。

        Args:
            layer_sizes (list): A list of integers specifying the number of neurons in each layer.
                                 第一个元素是输入层大小，最后一个是输出层大小。
                                 Example: [784, 128, 64, 10]
            activations (list): A list of strings specifying the activation function for each hidden layer.
                                Should be one of 'relu', 'sigmoid', 'softmax'.
                                Example: ['relu', 'relu']
        """
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.layers = nn.ModuleList()

        # Create hidden layers (创建隐藏层)
        for i in range(len(layer_sizes) - 2):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Create output layer (创建输出层)
        self.layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    def forward(self, x):
        """
        Forward pass through the network.
        网络的前向传播。

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (logits before softmax).
        """
        # Flatten input (将输入展平)
        x = x.view(x.size(0), -1) # Reshape (batch_size, 1, 28, 28) to (batch_size, 784)

        # Apply hidden layers with activations (应用隐藏层和激活函数)
        for i, layer in enumerate(self.layers[:-1]): # Iterate through all but the last layer
            x = layer(x)
            if self.activations[i] == 'relu':
                x = F.relu(x)
            elif self.activations[i] == 'sigmoid':
                x = torch.sigmoid(x)
            # Add more activation functions if needed (如果需要，添加更多激活函数)
            else:
                raise ValueError(f"Unsupported activation function: {self.activations[i]}")

        # Apply output layer (应用输出层)
        x = self.layers[-1](x)

        return x # Return logits; Softmax will be applied in loss function or prediction

if __name__ == '__main__':
    # Example Usage (示例用法)
    print("Testing MLPClassifierPyTorch...")
    print("正在测试MLPClassifierPyTorch...")
    
    # Define model parameters (定义模型参数)
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    activations = ['relu'] * len(hidden_sizes)

    # Create a dummy input tensor (创建虚拟输入张量)
    batch_size = 64
    dummy_input = torch.randn(batch_size, 1, 28, 28) # MNIST image size
    
    # Initialize the model (初始化模型)
    model = MLPClassifierPyTorch(layer_sizes, activations)
    print(f"Model architecture: {model}")
    print(f"模型架构: {model}")

    # Perform a forward pass (执行前向传播)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"输出形状: {output.shape}")
    
    # Check if the output is on CPU by default (检查输出默认是否在CPU上)
    print(f"Output device: {output.device}")
    print(f"输出设备: {output.device}")

    print("MLPClassifierPyTorch test completed.")
    print("MLPClassifierPyTorch测试完成。") 