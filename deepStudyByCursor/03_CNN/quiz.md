#!/usr/bin/env python3
"""
Quiz for Chapter 3: Convolutional Neural Networks
第三章测验：卷积神经网络

This quiz covers:
- CNN basics and convolution operations
- Pooling layers and feature maps
- CNN architectures (LeNet, AlexNet, VGG, ResNet)
- Practical CNN implementation

本测验涵盖：
- CNN基础和卷积操作
- 池化层和特征图
- CNN架构（LeNet、AlexNet、VGG、ResNet）
- CNN的实际实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def multiple_choice_questions():
    """Multiple choice questions about CNNs"""
    print("=== Multiple Choice Questions ===")
    print("=== 选择题 ===\n")
    
    questions = [
        {
            "question": "What is the main advantage of using convolutional layers over fully connected layers for image processing?",
            "question_zh": "对于图像处理，使用卷积层相比全连接层的主要优势是什么？",
            "options": [
                "A) Faster computation",
                "B) Parameter sharing and translation invariance", 
                "C) Better gradient flow",
                "D) Higher accuracy"
            ],
            "answer": "B",
            "explanation": "Convolutional layers use parameter sharing (same filter across the image) and provide translation invariance, making them ideal for image processing.",
            "explanation_zh": "卷积层使用参数共享（相同滤波器在图像上使用）并提供平移不变性，使其非常适合图像处理。"
        },
        {
            "question": "For a 5x5 input with a 3x3 kernel and stride=1, what is the output size?",
            "question_zh": "对于5x5输入，使用3x3卷积核，步长=1，输出大小是多少？",
            "options": [
                "A) 3x3",
                "B) 4x4", 
                "C) 5x5",
                "D) 7x7"
            ],
            "answer": "A",
            "explanation": "Output size = (Input - Kernel + 2*Padding) / Stride + 1 = (5-3+0)/1 + 1 = 3",
            "explanation_zh": "输出大小 = (输入-卷积核+2*填充)/步长 + 1 = (5-3+0)/1 + 1 = 3"
        },
        {
            "question": "What is the purpose of pooling layers in CNNs?",
            "question_zh": "CNN中池化层的目的是什么？",
            "options": [
                "A) Increase the number of parameters",
                "B) Reduce spatial dimensions and provide translation invariance",
                "C) Add non-linearity",
                "D) Normalize the input"
            ],
            "answer": "B",
            "explanation": "Pooling layers reduce spatial dimensions (downsampling) and provide some translation invariance.",
            "explanation_zh": "池化层减少空间维度（下采样）并提供一定的平移不变性。"
        },
        {
            "question": "In ResNet, what problem do residual connections solve?",
            "question_zh": "在ResNet中，残差连接解决了什么问题？",
            "options": [
                "A) Overfitting",
                "B) Vanishing gradient problem",
                "C) Computational complexity",
                "D) Memory usage"
            ],
            "answer": "B",
            "explanation": "Residual connections help gradients flow better through deep networks, solving the vanishing gradient problem.",
            "explanation_zh": "残差连接帮助梯度在深度网络中更好地流动，解决了梯度消失问题。"
        },
        {
            "question": "Which activation function is most commonly used in modern CNNs?",
            "question_zh": "现代CNN中最常用的激活函数是什么？",
            "options": [
                "A) Sigmoid",
                "B) Tanh",
                "C) ReLU",
                "D) Linear"
            ],
            "answer": "C",
            "explanation": "ReLU is most commonly used because it's computationally efficient and helps with gradient flow.",
            "explanation_zh": "ReLU最常用，因为它计算效率高且有助于梯度流动。"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"问题{i}：{q['question_zh']}")
        for option in q['options']:
            print(f"  {option}")
        
        user_answer = input("Your answer (A/B/C/D): ").upper().strip()
        
        if user_answer == q['answer']:
            print("✓ Correct! 正确！")
            print(f"Explanation: {q['explanation']}")
            print(f"解释：{q['explanation_zh']}")
            score += 1
        else:
            print(f"✗ Incorrect. The correct answer is {q['answer']}")
            print(f"✗ 错误。正确答案是{q['answer']}")
            print(f"Explanation: {q['explanation']}")
            print(f"解释：{q['explanation_zh']}")
        print("-" * 50)
    
    print(f"Score: {score}/{len(questions)}")
    print(f"得分：{score}/{len(questions)}")

def fill_in_blanks():
    """Fill in the blank questions"""
    print("\n=== Fill in the Blanks ===")
    print("=== 填空题 ===\n")
    
    questions = [
        {
            "question": "The mathematical operation in convolution is: output[i,j] = Σ Σ input[i+m, j+n] * ______[m,n]",
            "question_zh": "卷积中的数学操作是：output[i,j] = Σ Σ input[i+m, j+n] * ______[m,n]",
            "answer": "kernel",
            "explanation": "The convolution operation multiplies input values with kernel/filter weights.",
            "explanation_zh": "卷积操作将输入值与卷积核/滤波器权重相乘。"
        },
        {
            "question": "In a CNN, ______ layers extract features while ______ layers reduce spatial dimensions.",
            "question_zh": "在CNN中，______层提取特征，而______层减少空间维度。",
            "answer": "convolutional, pooling",
            "explanation": "Convolutional layers extract features, pooling layers downsample.",
            "explanation_zh": "卷积层提取特征，池化层进行下采样。"
        },
        {
            "question": "The number of parameters in a convolutional layer is: (kernel_height × kernel_width × input_channels + 1) × ______",
            "question_zh": "卷积层的参数数量是：(卷积核高度 × 卷积核宽度 × 输入通道数 + 1) × ______",
            "answer": "output_channels",
            "explanation": "Each output channel has its own set of weights plus bias.",
            "explanation_zh": "每个输出通道都有自己的权重集合加上偏置。"
        },
        {
            "question": "In image classification, the final layer is usually a ______ layer that outputs class probabilities.",
            "question_zh": "在图像分类中，最后一层通常是______层，输出类别概率。",
            "answer": "softmax",
            "explanation": "Softmax converts raw scores to probabilities that sum to 1.",
            "explanation_zh": "Softmax将原始分数转换为总和为1的概率。"
        },
        {
            "question": "Data augmentation techniques like rotation and flipping help prevent ______ and improve generalization.",
            "question_zh": "旋转和翻转等数据增强技术有助于防止______并改善泛化。",
            "answer": "overfitting",
            "explanation": "Data augmentation increases training data diversity, reducing overfitting.",
            "explanation_zh": "数据增强增加了训练数据的多样性，减少了过拟合。"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"问题{i}：{q['question_zh']}")
        
        user_answer = input("Your answer: ").strip().lower()
        correct_answers = [ans.strip().lower() for ans in q['answer'].split(',')]
        
        if any(ans in user_answer for ans in correct_answers):
            print("✓ Correct! 正确！")
            score += 1
        else:
            print(f"✗ Incorrect. The correct answer is: {q['answer']}")
            print(f"✗ 错误。正确答案是：{q['answer']}")
        
        print(f"Explanation: {q['explanation']}")
        print(f"解释：{q['explanation_zh']}")
        print("-" * 50)
    
    print(f"Score: {score}/{len(questions)}")
    print(f"得分：{score}/{len(questions)}")

def short_answer_questions():
    """Short answer questions"""
    print("\n=== Short Answer Questions ===")
    print("=== 简答题 ===\n")
    
    questions = [
        {
            "question": "Explain the difference between valid and same padding in convolution.",
            "question_zh": "解释卷积中valid填充和same填充的区别。",
            "sample_answer": "Valid padding means no padding is added, so output size is smaller than input. Same padding adds enough padding so that output size equals input size when stride=1.",
            "sample_answer_zh": "Valid填充意味着不添加填充，所以输出大小小于输入。Same填充添加足够的填充，使得当步长=1时输出大小等于输入大小。"
        },
        {
            "question": "Why do we use batch normalization in CNNs? What are its benefits?",
            "question_zh": "为什么在CNN中使用批归一化？它的好处是什么？",
            "sample_answer": "Batch normalization normalizes inputs to each layer, which stabilizes training, allows higher learning rates, reduces dependence on initialization, and acts as regularization.",
            "sample_answer_zh": "批归一化对每层的输入进行归一化，这稳定了训练，允许更高的学习率，减少对初始化的依赖，并起到正则化作用。"
        },
        {
            "question": "Compare max pooling and average pooling. When would you use each?",
            "question_zh": "比较最大池化和平均池化。什么时候使用每种？",
            "sample_answer": "Max pooling keeps the strongest features and is more common. Average pooling preserves more information but may dilute important features. Max pooling is preferred for feature detection, average pooling for smoother downsampling.",
            "sample_answer_zh": "最大池化保留最强的特征，更常用。平均池化保留更多信息但可能稀释重要特征。最大池化适用于特征检测，平均池化适用于更平滑的下采样。"
        },
        {
            "question": "What is transfer learning and why is it useful in CNN training?",
            "question_zh": "什么是迁移学习，为什么它在CNN训练中有用？",
            "sample_answer": "Transfer learning uses pre-trained models on large datasets as starting points. It's useful because lower layers learn general features that transfer well, reducing training time and data requirements.",
            "sample_answer_zh": "迁移学习使用在大数据集上预训练的模型作为起点。它有用是因为较低层学习的一般特征可以很好地迁移，减少了训练时间和数据需求。"
        }
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"问题{i}：{q['question_zh']}")
        print("\nPlease provide your answer:")
        print("请提供您的答案：")
        
        user_answer = input().strip()
        
        print(f"\nSample Answer: {q['sample_answer']}")
        print(f"参考答案：{q['sample_answer_zh']}")
        print("-" * 50)

def programming_exercises():
    """Programming exercises"""
    print("\n=== Programming Exercises ===")
    print("=== 编程练习 ===\n")
    
    print("Exercise 1: Implement a simple CNN for MNIST classification")
    print("练习1：实现一个简单的MNIST分类CNN")
    print()
    
    # Exercise 1: Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=10):
            super(SimpleCNN, self).__init__()
            # TODO: Define layers
            # 第一个卷积层：输入3通道，输出32通道，卷积核大小3x3
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            # 第二个卷积层：输入32通道，输出64通道，卷积核大小3x3
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            # 池化层
            self.pool = nn.MaxPool2d(2, 2)
            # 全连接层（需要根据输入图像大小计算）
            # 假设输入图像为32x32，经过两次池化后为8x8
            self.fc1 = nn.Linear(64 * 8 * 8, 512)
            self.fc2 = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            # TODO: Implement forward pass
            # Apply conv1 -> ReLU -> MaxPool
            x = self.pool(F.relu(self.conv1(x)))
            # Apply conv2 -> ReLU -> MaxPool
            x = self.pool(F.relu(self.conv2(x)))
            # Flatten and apply FC layers
            x = x.view(-1, 64 * 8 * 8)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    print("Complete the SimpleCNN class above.")
    print("完成上面的SimpleCNN类。")
    print()
    
    # Solution
    print("Solution / 解答:")
    
    # 测试模型
    model = SimpleCNN(num_classes=10)
    print(model)

    # 测试前向传播
    test_input = torch.randn(1, 3, 32, 32)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print()
    
    print("Exercise 2: Calculate convolution output size")
    print("练习2：计算卷积输出大小")
    print()
    
    def calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0):
        """
        计算卷积操作后的输出大小
        Calculate the output size after convolution operation
        
        Args:
            input_size: 输入大小 | Input size
            kernel_size: 卷积核大小 | Kernel size
            stride: 步长 | Stride
            padding: 填充 | Padding
        
        Returns:
            output_size: 输出大小 | Output size
        """
        output_size = (input_size - kernel_size + 2 * padding) // stride + 1
        return output_size
    
    # 测试示例 | Test examples
    print("卷积输出大小计算示例 | Convolution Output Size Examples:")
    print(f"Input: 224x224, Kernel: 7x7, Stride: 2, Padding: 3")
    print(f"Output: {calculate_conv_output_size(224, 7, 2, 3)}x{calculate_conv_output_size(224, 7, 2, 3)}")

    print(f"Input: 32x32, Kernel: 3x3, Stride: 1, Padding: 1")
    print(f"Output: {calculate_conv_output_size(32, 3, 1, 1)}x{calculate_conv_output_size(32, 3, 1, 1)}")

    print(f"Input: 28x28, Kernel: 5x5, Stride: 1, Padding: 0")
    print(f"Output: {calculate_conv_output_size(28, 5, 1, 0)}x{calculate_conv_output_size(28, 5, 1, 0)}")
    print()
    
    print("Exercise 3: Implement ResNet block")
    print("练习3：实现ResNet块")
    print()
    
    class ResNetBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResNetBlock, self).__init__()
            
            # 第一个卷积层
            self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                  kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            
            # 第二个卷积层
            self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                  kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            # 残差连接的调整层（如果输入输出维度不同）
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 
                             kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        
        def forward(self, x):
            residual = x
            
            # 主路径
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            
            # 残差连接
            residual = self.shortcut(residual)
            out += residual
            
            out = F.relu(out)
            return out
    
    print("The ResNet block above implements:")
    print("- Two 3x3 convolutions with batch normalization")
    print("- Shortcut connection that adds input to output")
    print("- Handles dimension mismatch with 1x1 convolution")
    print()
    print("上面的ResNet块实现了：")
    print("- 两个3x3卷积与批归一化")
    print("- 将输入添加到输出的快捷连接")
    print("- 用1x1卷积处理维度不匹配")
    print()
    
    # 测试ResNet块
    block = ResNetBlock(64, 128, stride=2)
    test_input = torch.randn(1, 64, 32, 32)
    output = block(test_input)
    print(f"ResNet Block - Input shape: {test_input.shape}")
    print(f"ResNet Block - Output shape: {output.shape}")
    print()
    
    print("Exercise 4: CNN Feature Visualization")
    print("练习4：CNN特征可视化")
    print()
    
    import matplotlib.pyplot as plt

    def visualize_conv_filters(model, layer_name, num_filters=8):
        """
        可视化卷积层的滤波器
        Visualize convolutional layer filters
        """
        # 获取指定层的权重
        layer = dict(model.named_modules())[layer_name]
        weights = layer.weight.data
        
        # 创建子图
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(num_filters, weights.shape[0])):
            # 取第一个输入通道的滤波器
            filter_img = weights[i, 0, :, :].cpu().numpy()
            
            axes[i].imshow(filter_img, cmap='gray')
            axes[i].set_title(f'Filter {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    def visualize_feature_maps(model, input_tensor, layer_name):
        """
        可视化特征图
        Visualize feature maps
        """
        # 注册钩子函数获取中间层输出
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 注册钩子
        layer = dict(model.named_modules())[layer_name]
        handle = layer.register_forward_hook(get_activation(layer_name))
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 获取特征图
        feature_maps = activation[layer_name][0]  # 取第一个样本
        
        # 可视化前8个特征图
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(min(8, feature_maps.shape[0])):
            feature_map = feature_maps[i].cpu().numpy()
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Feature Map {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 移除钩子
        handle.remove()

    # 使用示例
    model = SimpleCNN()
    test_input = torch.randn(1, 3, 32, 32)

    # 可视化第一层卷积滤波器
    # visualize_conv_filters(model, 'conv1')

    # 可视化第一层特征图
    # visualize_feature_maps(model, test_input, 'conv1')
    print()

def main():
    """Main function to run all quiz sections"""
    print("Welcome to Chapter 3 Quiz: Convolutional Neural Networks")
    print("欢迎来到第三章测验：卷积神经网络")
    print("=" * 60)
    
    while True:
        print("\nChoose a section:")
        print("选择一个部分：")
        print("1. Multiple Choice Questions (选择题)")
        print("2. Fill in the Blanks (填空题)")
        print("3. Short Answer Questions (简答题)")
        print("4. Programming Exercises (编程练习)")
        print("5. Exit (退出)")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            multiple_choice_questions()
        elif choice == '2':
            fill_in_blanks()
        elif choice == '3':
            short_answer_questions()
        elif choice == '4':
            programming_exercises()
        elif choice == '5':
            print("Thank you for taking the quiz! 谢谢参与测验！")
            break
        else:
            print("Invalid choice. Please try again. 无效选择，请重试。")

if __name__ == "__main__":
    main() 