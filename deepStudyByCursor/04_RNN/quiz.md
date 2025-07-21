#!/usr/bin/env python3
"""
Quiz for Chapter 4: Recurrent Neural Networks
第四章测验：循环神经网络

This quiz covers:
- RNN basics and sequential data processing
- Vanishing gradient problem
- LSTM and GRU architectures
- Sequence-to-sequence models

本测验涵盖：
- RNN基础和序列数据处理
- 梯度消失问题
- LSTM和GRU架构
- 序列到序列模型
"""

import torch
import torch.nn as nn
import numpy as np

# Quiz: Modern Convolutional Neural Networks
# 测验：现代卷积神经网络

## Multiple Choice Questions 选择题

### 1. AlexNet and Deep CNNs

**Q1.1** What was the main breakthrough that AlexNet achieved?
AlexNet取得的主要突破是什么？

A) First use of convolutional layers
B) Winning ImageNet competition and proving deep learning effectiveness  
C) Introduction of batch normalization
D) Using residual connections

**Answer: B**
**解析**: AlexNet won the 2012 ImageNet competition with a significant margin, proving that deep learning could achieve state-of-the-art results in computer vision.
AlexNet以显著优势赢得了2012年ImageNet竞赛，证明了深度学习能在计算机视觉中取得最先进的结果。

**Q1.2** Which activation function did AlexNet popularize in deep learning?
AlexNet在深度学习中推广了哪种激活函数？

A) Sigmoid
B) Tanh  
C) ReLU
D) Leaky ReLU

**Answer: C**
**解析**: AlexNet popularized ReLU activation, which helps with vanishing gradient problems and is computationally efficient.
AlexNet推广了ReLU激活函数，它有助于解决梯度消失问题且计算效率高。

### 2. VGG Networks

**Q2.1** What is the key architectural principle of VGG networks?
VGG网络的关键架构原则是什么？

A) Using very large filters (11×11)
B) Using only small 3×3 filters throughout the network
C) Using 1×1 convolutions for dimensionality reduction
D) Using skip connections

**Answer: B**
**解析**: VGG networks consistently use small 3×3 filters, showing that smaller filters can achieve the same receptive field as larger ones with fewer parameters.
VGG网络始终使用小的3×3滤波器，表明小滤波器可以用更少的参数实现与大滤波器相同的感受野。

**Q2.2** How many 3×3 convolutions are needed to get the same receptive field as one 5×5 convolution?
需要多少个3×3卷积才能获得与一个5×5卷积相同的感受野？

A) 1
B) 2
C) 3
D) 4

**Answer: B**
**解析**: Two 3×3 convolutions have the same receptive field as one 5×5 convolution but with fewer parameters and more non-linearity.
两个3×3卷积与一个5×5卷积具有相同的感受野，但参数更少，非线性更强。

### 3. Network in Network (NiN)

**Q3.1** What is the main innovation of Network in Network (NiN)?
网络中的网络(NiN)的主要创新是什么？

A) Using deeper networks
B) Using 1×1 convolutions as "mlpconv" layers
C) Using batch normalization
D) Using residual connections

**Answer: B**
**解析**: NiN introduced 1×1 convolutions that act like multilayer perceptrons applied to each pixel, adding non-linearity and feature mixing.
NiN引入了1×1卷积，它们像应用于每个像素的多层感知器，增加了非线性和特征混合。

**Q3.2** What does NiN use instead of fully connected layers at the end?
NiN在最后使用什么来替代全连接层？

A) Max pooling
B) Average pooling
C) Global average pooling
D) Global max pooling

**Answer: C**
**解析**: NiN uses global average pooling to replace fully connected layers, dramatically reducing the number of parameters.
NiN使用全局平均池化来替代全连接层，大大减少了参数数量。

### 4. GoogLeNet and Inception

**Q4.1** What is the core idea behind Inception blocks?
Inception块背后的核心思想是什么？

A) Going deeper with more layers
B) Using wider networks with more channels
C) Applying multiple filter sizes in parallel
D) Using only 1×1 convolutions

**Answer: C**
**解析**: Inception blocks apply different filter sizes (1×1, 3×3, 5×5) and pooling in parallel, then concatenate the results.
Inception块并行应用不同的滤波器尺寸（1×1、3×3、5×5）和池化，然后拼接结果。

**Q4.2** Why does GoogLeNet use auxiliary classifiers during training?
为什么GoogLeNet在训练期间使用辅助分类器？

A) To increase model capacity
B) To help gradient flow in very deep networks
C) To reduce overfitting
D) To speed up training

**Answer: B**
**解析**: Auxiliary classifiers provide additional gradient signals to help train the very deep network by combating vanishing gradients.
辅助分类器提供额外的梯度信号，通过对抗梯度消失来帮助训练非常深的网络。

### 5. Batch Normalization

**Q5.1** What problem does batch normalization primarily solve?
批量归一化主要解决什么问题？

A) Overfitting
B) Internal covariate shift
C) Computational efficiency
D) Memory usage

**Answer: B**
**解析**: Batch normalization addresses internal covariate shift - the change in distribution of layer inputs during training.
批量归一化解决内部协变量偏移——训练过程中层输入分布的变化。

**Q5.2** In the batch normalization equation y = γx̂ + β, what are γ and β?
在批量归一化方程y = γx̂ + β中，γ和β是什么？

A) Fixed constants
B) Learnable parameters
C) Input statistics
D) Activation functions

**Answer: B**
**解析**: γ (scale) and β (shift) are learnable parameters that allow the network to recover the original distribution if needed.
γ（缩放）和β（偏移）是可学习参数，允许网络在需要时恢复原始分布。

### 6. ResNet

**Q6.1** What is the key innovation of ResNet?
ResNet的关键创新是什么？

A) Deeper networks
B) Skip connections/residual learning
C) Batch normalization
D) 1×1 convolutions

**Answer: B**
**解析**: ResNet introduced skip connections that allow gradients to flow directly to earlier layers, enabling training of much deeper networks.
ResNet引入了跳跃连接，允许梯度直接流向早期层，使训练更深网络成为可能。

**Q6.2** In ResNet, what does the residual function F(x) learn?
在ResNet中，残差函数F(x)学习什么？

A) The identity mapping x
B) The output H(x) directly
C) The residual H(x) - x
D) The gradient information

**Answer: C**
**解析**: The residual function F(x) learns the residual H(x) - x, where H(x) is the desired output. This makes learning identity mappings easier.
残差函数F(x)学习残差H(x) - x，其中H(x)是期望输出。这使得学习恒等映射更容易。

### 7. ResNeXt

**Q7.1** What concept does ResNeXt introduce to improve upon ResNet?
ResNeXt引入了什么概念来改进ResNet？

A) Depth
B) Width
C) Cardinality (number of paths)
D) Resolution

**Answer: C**
**解析**: ResNeXt introduces cardinality - the number of parallel paths in a block, using grouped convolutions to implement multiple branches efficiently.
ResNeXt引入了基数——块中并行路径的数量，使用分组卷积有效地实现多个分支。

**Q7.2** How does ResNeXt implement multiple branches efficiently?
ResNeXt如何高效地实现多个分支？

A) Using separate convolution layers
B) Using grouped convolutions
C) Using 1×1 convolutions only
D) Using depth-wise convolutions

**Answer: B**
**解析**: ResNeXt uses grouped convolutions to implement multiple paths efficiently, reducing computational cost while maintaining model capacity.
ResNeXt使用分组卷积高效地实现多条路径，在保持模型容量的同时降低计算成本。

### 8. DenseNet

**Q8.1** How are layers connected in DenseNet?
DenseNet中的层是如何连接的？

A) Each layer connects only to the next layer
B) Each layer connects to the previous layer via addition
C) Each layer connects to all subsequent layers via concatenation
D) Layers are connected randomly

**Answer: C**
**解析**: In DenseNet, each layer connects to all subsequent layers through feature map concatenation, maximizing information flow.
在DenseNet中，每层通过特征图拼接连接到所有后续层，最大化信息流。

**Q8.2** What is the purpose of transition layers in DenseNet?
DenseNet中过渡层的目的是什么？

A) Add non-linearity
B) Reduce spatial dimensions and control feature growth
C) Increase network depth
D) Improve gradient flow

**Answer: B**
**解析**: Transition layers use 1×1 convolutions and pooling to reduce spatial dimensions and control the growth of feature maps between dense blocks.
过渡层使用1×1卷积和池化来减少空间维度并控制稠密块之间特征图的增长。

## Short Answer Questions 简答题

### Q9. Architecture Comparison 架构比较

Compare the connection patterns in ResNet and DenseNet. What are the advantages and disadvantages of each approach?
比较ResNet和DenseNet的连接模式。每种方法的优缺点是什么？

**Answer:**
**ResNet (Additive Skip Connections):**
- Advantages: Memory efficient, faster training, easier optimization
- Disadvantages: Information may be lost through addition

**DenseNet (Concatenative Connections):**  
- Advantages: Maximum information preservation, feature reuse, parameter efficiency
- Disadvantages: High memory usage, slower training due to concatenations

**解析:**
**ResNet（加法跳跃连接）：**
- 优势：内存效率高，训练更快，优化更容易
- 劣势：通过加法可能丢失信息

**DenseNet（拼接连接）：**
- 优势：最大化信息保存，特征重用，参数效率高
- 劣势：内存使用量大，由于拼接导致训练较慢

### Q10. Design Principles 设计原则

Explain the evolution of CNN design principles from AlexNet to modern architectures. What key insights drove each major advancement?
解释从AlexNet到现代架构的CNN设计原则演变。什么关键洞察推动了每次重大进步？

**Answer:**
**Evolution of Design Principles:**

1. **AlexNet → VGG**: Smaller filters are more efficient
   - Insight: Multiple small filters > single large filter
   
2. **VGG → NiN**: 1×1 convolutions for feature mixing
   - Insight: Network-in-network concept, global average pooling
   
3. **NiN → GoogLeNet**: Multi-scale feature extraction
   - Insight: Parallel paths with different filter sizes
   
4. **GoogLeNet → ResNet**: Skip connections for deep networks
   - Insight: Residual learning enables very deep networks
   
5. **ResNet → DenseNet**: Maximum information flow
   - Insight: Dense connections maximize feature reuse

**解析:**
**设计原则演变：**

1. **AlexNet → VGG**: 小滤波器更高效
   - 洞察：多个小滤波器 > 单个大滤波器
   
2. **VGG → NiN**: 1×1卷积用于特征混合
   - 洞察：网络中的网络概念，全局平均池化
   
3. **NiN → GoogLeNet**: 多尺度特征提取
   - 洞察：不同滤波器尺寸的并行路径
   
4. **GoogLeNet → ResNet**: 跳跃连接用于深度网络
   - 洞察：残差学习使得非常深的网络成为可能
   
5. **ResNet → DenseNet**: 最大化信息流
   - 洞察：密集连接最大化特征重用

## Programming Questions 编程题

### Q11. Implementation Challenge 实现挑战

Implement a simplified Inception block that takes an input tensor and applies 1×1, 3×3, and 5×5 convolutions in parallel, then concatenates the results.
实现一个简化的Inception块，它接受输入张量并并行应用1×1、3×3和5×5卷积，然后拼接结果。

**Answer:**
```python
import torch
import torch.nn as nn

class SimpleInceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5):
        super().__init__()
        
        # 1×1 convolution path
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3×3 convolution path
        self.branch3x3 = nn.Conv2d(in_channels, out_3x3, kernel_size=3, padding=1)
        
        # 5×5 convolution path  
        self.branch5x5 = nn.Conv2d(in_channels, out_5x5, kernel_size=5, padding=2)
        
    def forward(self, x):
        branch1x1 = torch.relu(self.branch1x1(x))
        branch3x3 = torch.relu(self.branch3x3(x))
        branch5x5 = torch.relu(self.branch5x5(x))
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1x1, branch3x3, branch5x5], dim=1)
        return outputs

# Test the implementation
model = SimpleInceptionBlock(64, 16, 32, 16)
x = torch.randn(1, 64, 32, 32)
output = model(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # Should be [1, 64, 32, 32]
```

### Q12. Architecture Analysis 架构分析

Write a function that computes the number of parameters in a basic ResNet block vs a basic DenseNet layer, given the input channels and growth rate.
编写一个函数，计算基本ResNet块与基本DenseNet层的参数数量，给定输入通道和增长率。

**Answer:**
```python
def compare_parameters(in_channels, growth_rate=32):
    """
    Compare parameters between ResNet block and DenseNet layer
    比较ResNet块和DenseNet层的参数数量
    """
    
    # ResNet Basic Block parameters
    # Two 3×3 convolutions: in_channels → in_channels → in_channels
    resnet_params = (
        in_channels * in_channels * 3 * 3 +  # First conv
        in_channels * in_channels * 3 * 3    # Second conv
    )
    
    # DenseNet Layer parameters (with bottleneck)
    # 1×1 conv: in_channels → 4*growth_rate
    # 3×3 conv: 4*growth_rate → growth_rate
    bottleneck_channels = 4 * growth_rate
    densenet_params = (
        in_channels * bottleneck_channels * 1 * 1 +  # 1×1 conv
        bottleneck_channels * growth_rate * 3 * 3    # 3×3 conv
    )
    
    print(f"Input channels: {in_channels}")
    print(f"Growth rate: {growth_rate}")
    print(f"ResNet block parameters: {resnet_params:,}")
    print(f"DenseNet layer parameters: {densenet_params:,}")
    print(f"Ratio (ResNet/DenseNet): {resnet_params/densenet_params:.2f}")
    
    return resnet_params, densenet_params

# Test with different configurations
compare_parameters(256, 32)
compare_parameters(512, 32)
```

## True/False Questions 判断题

### Q13-Q20: Mark True (T) or False (F) 标记正确（T）或错误（F）

**Q13.** Batch normalization is applied before the activation function in most modern networks.
在大多数现代网络中，批量归一化在激活函数之前应用。

**Answer: T**
**解析**: The standard order is Conv → BatchNorm → ReLU, though pre-activation variants exist.
标准顺序是卷积 → 批量归一化 → ReLU，尽管存在预激活变体。

**Q14.** DenseNet typically requires more memory during training than ResNet.
DenseNet在训练期间通常比ResNet需要更多内存。

**Answer: T**
**解析**: DenseNet's concatenative connections require storing all intermediate feature maps, leading to higher memory usage.
DenseNet的拼接连接需要存储所有中间特征图，导致更高的内存使用。

**Q15.** The growth rate in DenseNet refers to how fast the network depth increases.
DenseNet中的增长率指的是网络深度增加的速度。

**Answer: F**
**解析**: Growth rate refers to the number of output channels each layer adds, not the depth increase.
增长率指的是每层添加的输出通道数，而不是深度增加。

**Q16.** Skip connections in ResNet only help with gradient flow and don't affect the model's representational capacity.
ResNet中的跳跃连接只有助于梯度流，不影响模型的表示能力。

**Answer: F**
**解析**: Skip connections both help with gradient flow and enable the network to learn more complex functions by allowing identity mappings.
跳跃连接既有助于梯度流，也通过允许恒等映射使网络能够学习更复杂的函数。

**Q17.** Global average pooling in NiN reduces the risk of overfitting compared to fully connected layers.
NiN中的全局平均池化与全连接层相比降低了过拟合风险。

**Answer: T**
**解析**: Global average pooling has no parameters to overfit, unlike fully connected layers which have many parameters.
全局平均池化没有可过拟合的参数，不像具有许多参数的全连接层。

**Q18.** Inception blocks process all input channels with each filter size.
Inception块用每种滤波器尺寸处理所有输入通道。

**Answer: F**
**解析**: Inception blocks often use 1×1 convolutions for dimensionality reduction before 3×3 and 5×5 convolutions.
Inception块通常在3×3和5×5卷积之前使用1×1卷积进行降维。

**Q19.** Grouped convolutions in ResNeXt reduce computational cost while maintaining model capacity.
ResNeXt中的分组卷积在保持模型容量的同时降低计算成本。

**Answer: T**
**解析**: Grouped convolutions reduce parameters and computation while the multiple groups maintain representational capacity.
分组卷积减少参数和计算，而多个组保持表示能力。

**Q20.** Transition layers in DenseNet only perform spatial downsampling.
DenseNet中的过渡层只执行空间下采样。

**Answer: F**
**解析**: Transition layers perform both channel compression (via 1×1 conv) and spatial downsampling (via pooling).
过渡层既执行通道压缩（通过1×1卷积）又执行空间下采样（通过池化）。

---

## Answer Key 答案

**Multiple Choice:** 1.1-B, 1.2-C, 2.1-B, 2.2-B, 3.1-B, 3.2-C, 4.1-C, 4.2-B, 5.1-B, 5.2-B, 6.1-B, 6.2-C, 7.1-C, 7.2-B, 8.1-C, 8.2-B

**True/False:** 13-T, 14-T, 15-F, 16-F, 17-T, 18-F, 19-T, 20-F

**Total Score: ___/28** 