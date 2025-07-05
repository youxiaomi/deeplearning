# 第二章：多层感知机与反向传播 - 测试题

## 选择题

### 1. 单层感知机无法解决XOR问题的根本原因是什么？
A. 参数数量不够  
B. 学习率设置不当  
C. XOR问题是非线性可分的  
D. 激活函数选择错误  

### 2. 在多层感知机中，隐藏层的主要作用是什么？
A. 减少计算量  
B. 提供非线性变换能力  
C. 防止过拟合  
D. 加快训练速度  

### 3. 反向传播算法的核心数学原理是什么？
A. 梯度下降  
B. 链式法则  
C. 泰勒展开  
D. 拉格朗日乘数法  

### 4. 对于Sigmoid激活函数，当输入z=0时，其导数值是多少？
A. 0  
B. 0.25  
C. 0.5  
D. 1  

### 5. 在小批量梯度下降中，批量大小的选择会影响什么？
A. 模型的表达能力  
B. 训练的稳定性和速度  
C. 激活函数的形状  
D. 网络的层数  

## 填空题

### 1. 对于一个具有n个输入、h个隐藏神经元、m个输出的三层MLP，从输入层到隐藏层的权重矩阵W^(1)的维度是 _______ × _______

### 2. 在反向传播过程中，误差信号从 _______ 层开始，逐层向 _______ 传播。

### 3. 交叉熵损失函数结合Softmax激活函数时，输出层的梯度具有简洁形式：∂L/∂z_i = _______

### 4. 梯度下降的更新公式是：θ_new = θ_old - η × _______

### 5. 学习率如果设置过大，可能导致训练过程出现 _______ 现象；如果设置过小，会导致训练 _______ 过慢。

## 简答题

### 1. 用数学方法证明单层感知机无法解决XOR问题。

### 2. 解释反向传播算法中链式法则的应用，并给出一个具体的梯度计算例子。

### 3. 比较批量梯度下降、随机梯度下降和小批量梯度下降的优缺点。

### 4. 说明为什么需要激活函数，以及如果没有激活函数会发生什么？

## 编程题

### 1. 实现一个简单的多层感知机类，包含前向传播和反向传播功能：

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid函数的导数"""
    s = sigmoid(x)
    return s * (1 - s)

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        初始化多层感知机
        
        参数:
        - input_size: 输入层神经元数量
        - hidden_size: 隐藏层神经元数量  
        - output_size: 输出层神经元数量
        - learning_rate: 学习率
        """
        # 初始化权重和偏置（小随机数）
        self.W1 = np.random.randn(hidden_size, input_size) * 0.1
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.1
        self.b2 = np.zeros((output_size, 1))
        
        self.learning_rate = learning_rate
        
        # 存储中间变量（用于反向传播）
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
    def forward(self, X):
        """
        前向传播
        
        参数:
        - X: 输入数据，形状为 (input_size, batch_size)
        
        返回:
        - a2: 输出层的激活值
        """
        # 隐藏层计算
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # 输出层计算  
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        反向传播
        
        参数:
        - X: 输入数据
        - y: 真实标签
        - output: 前向传播的输出
        """
        m = X.shape[1]  # 样本数量
        
        # 输出层误差
        dz2 = output - y
        dW2 = (1/m) * np.dot(dz2, self.a1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        
        # 隐藏层误差
        dz1 = np.dot(self.W2.T, dz2) * sigmoid_derivative(self.z1)
        dW1 = (1/m) * np.dot(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        
        # 更新参数
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        """
        训练模型
        
        参数:
        - X: 训练数据
        - y: 训练标签
        - epochs: 训练轮数
        
        返回:
        - losses: 每轮的损失值列表
        """
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失（均方误差）
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # 反向传播
            self.backward(X, y, output)
            
            # 每100轮打印一次损失
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
```

### 2. 使用上面的MLP类解决XOR问题：

```python
def solve_xor_problem():
    """使用MLP解决XOR问题"""
    # XOR数据集
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])  # 输入: (2, 4)
    y = np.array([[0, 1, 1, 0]])   # 输出: (1, 4)
    
    print("XOR问题数据集：")
    print("输入 X:")
    print(X)
    print("期望输出 y:")
    print(y)
    
    # 创建MLP模型
    mlp = SimpleMLP(input_size=2, hidden_size=4, output_size=1, learning_rate=1.0)
    
    print("\n开始训练...")
    losses = mlp.train(X, y, epochs=1000)
    
    # 测试训练结果
    print("\n训练完成！测试结果：")
    final_output = mlp.forward(X)
    
    for i in range(4):
        input_val = X[:, i]
        expected = y[0, i]
        predicted = final_output[0, i]
        print(f"输入: ({input_val[0]}, {input_val[1]}) -> 期望: {expected}, 预测: {predicted:.4f}")
    
    return mlp, losses

# 运行示例
mlp_model, training_losses = solve_xor_problem()

# 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(training_losses)
plt.title('XOR问题训练损失曲线')
plt.xlabel('训练轮次 (Epochs)')
plt.ylabel('均方误差损失')
plt.grid(True)
plt.show()
```

### 3. 手动计算反向传播的一个步骤：

```python
def manual_backprop_example():
    """手动计算反向传播的例子"""
    print("给定一个简单的网络：")
    print("输入: x = [1, 1]")
    print("隐藏层权重: W1 = [[0.5, 0.5], [-0.5, -0.5]]")
    print("隐藏层偏置: b1 = [-0.2, 0.7]")
    print("输出层权重: W2 = [1.0, 1.0]")  
    print("输出层偏置: b2 = -0.5")
    print("目标输出: t = 0")
    
    # 定义参数
    x = np.array([1, 1])
    W1 = np.array([[0.5, 0.5], [-0.5, -0.5]])
    b1 = np.array([-0.2, 0.7])
    W2 = np.array([1.0, 1.0])
    b2 = -0.5
    t = 0
    
    print("\n前向传播计算：")
    
    # 隐藏层计算
    z1 = np.dot(W1, x) + b1
    h = sigmoid(z1)
    print(f"隐藏层加权和 z1 = {z1}")
    print(f"隐藏层输出 h = {h}")
    
    # 输出层计算
    z2 = np.dot(W2, h) + b2
    y = sigmoid(z2)
    print(f"输出层加权和 z2 = {z2:.3f}")
    print(f"最终输出 y = {y:.3f}")
    
    # 损失计算
    loss = 0.5 * (t - y) ** 2
    print(f"损失 L = {loss:.3f}")
    
    print("\n反向传播计算：")
    
    # 输出层误差
    delta2 = (y - t) * sigmoid_derivative(z2)
    print(f"输出层误差 δ2 = {delta2:.3f}")
    
    # 隐藏层误差
    delta1 = np.dot(W2, delta2) * sigmoid_derivative(z1)
    print(f"隐藏层误差 δ1 = {delta1}")
    
    # 梯度计算
    dW2 = delta2 * h
    db2 = delta2
    dW1 = np.outer(delta1, x)
    db1 = delta1
    
    print(f"\n梯度：")
    print(f"∂L/∂W2 = {dW2}")
    print(f"∂L/∂b2 = {db2:.3f}")
    print(f"∂L/∂W1 = {dW1}")
    print(f"∂L/∂b1 = {db1}")
    
    # 运行手动计算示例
    manual_backprop_example()
```

---

## 答案解析

### 选择题答案
1. **C** - XOR问题是非线性可分的
2. **B** - 提供非线性变换能力
3. **B** - 链式法则
4. **B** - 0.25 (因为σ'(0) = σ(0)(1-σ(0)) = 0.5×0.5 = 0.25)
5. **B** - 训练的稳定性和速度

### 填空题答案
1. **h × n**
2. **输出**，**前**
3. **y_i - t_i**
4. **∇L(θ)**
5. **震荡**，**收敛**

### 简答题答案要点

#### 1. XOR问题的线性不可分性证明：
通过设立不等式组，证明不存在线性分界面能同时满足所有XOR约束条件。

设感知机的决策函数为 f(x) = w₁x₁ + w₂x₂ + b

对于XOR问题：
- (0,0) → 0: w₁×0 + w₂×0 + b < 0 ⟹ b < 0
- (0,1) → 1: w₁×0 + w₂×1 + b > 0 ⟹ w₂ + b > 0
- (1,0) → 1: w₁×1 + w₂×0 + b > 0 ⟹ w₁ + b > 0  
- (1,1) → 0: w₁×1 + w₂×1 + b < 0 ⟹ w₁ + w₂ + b < 0

从前三个不等式得到：w₁ > -b 且 w₂ > -b 且 b < 0
因此：w₁ + w₂ > -2b > 0
但第四个不等式要求：w₁ + w₂ < -b < 0

这产生了矛盾，证明了单层感知机无法解决XOR问题。

#### 2. 链式法则应用：
反向传播中的链式法则：∂L/∂w = (∂L/∂y)(∂y/∂z)(∂z/∂w)

每一层的误差都是后一层误差与连接权重的乘积，这样误差信号可以从输出层逐层传播到输入层。

#### 3. 梯度下降方法比较：
- **批量GD**：使用全部训练数据，稳定但计算慢
- **随机GD**：使用单个样本，快速但不稳定，容易震荡
- **小批量GD**：平衡两者优势，是实际应用中的主流选择

#### 4. 激活函数的必要性：
- 引入非线性，使网络能够学习复杂的非线性模式
- 没有激活函数的多层网络等价于单层线性模型
- 激活函数提供了网络的表达能力和学习复杂函数的基础 

## 02_MLP_Backpropagation: 多层感知机与反向传播 - 随堂测验

### 02.1 PyTorch与GPU加速

**问题 1 (Question 1):**

你的原始 `train.py` 脚本使用NumPy从零实现MLP。请问，为什么这个脚本不能直接利用GPU进行加速？

**Why can't your original `train.py` script, which implements MLP from scratch using NumPy, directly utilize the GPU for acceleration?

**答案 (Answer):**

原始的NumPy实现是在CPU上执行数学运算的，NumPy本身不具备直接与GPU硬件交互的能力。要利用GPU，需要使用像PyTorch或TensorFlow这样的深度学习框架，它们提供了将计算任务分配到GPU的接口和优化。

The original NumPy implementation performs mathematical operations on the CPU, and NumPy itself does not have the capability to directly interact with GPU hardware. To utilize the GPU, deep learning frameworks like PyTorch or TensorFlow are required, as they provide interfaces and optimizations to offload computational tasks to the GPU.

---

**问题 2 (Question 2):**

在PyTorch中，如果你想把模型和数据从CPU移动到GPU上进行计算，你会使用哪个核心方法？请举例说明。

**In PyTorch, if you want to move a model and data from the CPU to the GPU for computation, which core method would you use? Please provide an example.

**答案 (Answer):**

你会使用 `.to(device)` 方法。

You would use the `.to(device)` method.

**例子 (Example):**
```python
import torch
import torch.nn as nn

# Check if GPU is available (检查GPU是否可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a dummy model (创建一个虚拟模型)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
model.to(device) # Move model to GPU (将模型移动到GPU)
print(f"Model is on: {next(model.parameters()).device}")

# Create a dummy tensor (创建一个虚拟张量)
data = torch.randn(5, 10)
data = data.to(device) # Move data to GPU (将数据移动到GPU)
print(f"Data is on: {data.device}")
```

---

**问题 3 (Question 3):**

在PyTorch的训练循环中，`optimizer.zero_grad()`、`loss.backward()` 和 `optimizer.step()` 这三个步骤分别有什么作用？

**In a PyTorch training loop, what are the roles of `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()` respectively?

**答案 (Answer):**

*   `optimizer.zero_grad()`:
    *   **作用 (Role)**: 清除模型中所有可训练参数的梯度。PyTorch默认会累积梯度，所以每次反向传播之前都需要清零，以避免梯度累积导致错误更新。
    *   **Role**: Zeros the gradients of all optimized `torch.Tensor`s. PyTorch accumulates gradients by default, so it's necessary to zero them before each backward pass to prevent incorrect updates due to accumulated gradients.

*   `loss.backward()`:
    *   **作用 (Role)**: 执行反向传播。它根据损失函数计算模型中所有可训练参数的梯度。
    *   **Role**: Performs backpropagation. It computes the gradients of the loss with respect to all trainable parameters in the model.

*   `optimizer.step()`:
    *   **作用 (Role)**: 根据计算出的梯度更新模型的参数。优化器（如Adam或SGD）会使用这些梯度和学习率来调整模型的权重和偏置。
    *   **Role**: Updates the model's parameters based on the computed gradients. The optimizer (e.g., Adam or SGD) uses these gradients and the learning rate to adjust the model's weights and biases.

---

**问题 4 (Question 4):**

`torchvision.datasets.MNIST` 和 `torch.utils.data.DataLoader` 在PyTorch的数据加载流程中分别扮演什么角色？

**What roles do `torchvision.datasets.MNIST` and `torch.utils.data.DataLoader` play in PyTorch's data loading pipeline?

**答案 (Answer):**

*   `torchvision.datasets.MNIST` (或任何 `torch.utils.data.Dataset` 的子类):
    *   **角色 (Role)**: 它代表了数据集本身。它负责加载单个数据样本及其对应的标签。对于MNIST，它处理数据集的下载、读取和基本的预处理（通过 `transform` 参数）。
    *   **Role**: It represents the dataset itself. It is responsible for loading individual data samples and their corresponding labels. For MNIST, it handles downloading, reading, and basic preprocessing (via the `transform` argument).

*   `torch.utils.data.DataLoader`:
    *   **角色 (Role)**: 它是一个迭代器，包裹着 `Dataset`，并提供了一种从数据集中高效加载批次数据的方法。它负责批量处理 (batching)、数据混洗 (shuffling) 和并行数据加载 (multi-process data loading using `num_workers`)。`DataLoader` 使训练过程能够以小批量的方式进行，这对于深度学习的优化至关重要。
    *   **Role**: It is an iterator that wraps a `Dataset` and provides an efficient way to load batches of data from the dataset. It is responsible for batching, shuffling, and parallel data loading (using `num_workers`). The `DataLoader` enables the training process to work with mini-batches, which is crucial for deep learning optimization. 