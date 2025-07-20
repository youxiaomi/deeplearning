# 第十五章：高斯过程 - 测试题 (Quiz)

## 15.1 Introduction to Gaussian Processes (高斯过程简介)

### 选择题 (Multiple Choice)

**Q1.1**: What is the key difference between Gaussian processes and traditional neural networks?
**Q1.1**: 高斯过程和传统神经网络的关键区别是什么？

A) GPs are faster to train (GP训练更快)
B) GPs define distributions over functions, while NNs learn specific parameter values (GP定义函数上的分布，而NN学习特定参数值)
C) GPs only work for regression problems (GP只适用于回归问题)
D) GPs require more data (GP需要更多数据)

**答案**: B
**解析**: Gaussian processes define distributions over entire functions rather than learning specific parameter values like neural networks. This allows them to provide uncertainty quantification naturally.
高斯过程定义整个函数上的分布，而不是像神经网络那样学习特定参数值。这使它们能够自然地提供不确定性量化。

**Q1.2**: Why are Gaussian processes called "non-parametric" models?
**Q1.2**: 为什么高斯过程被称为"非参数"模型？

A) They have no parameters (它们没有参数)
B) The number of parameters is fixed (参数数量是固定的)
C) The model complexity grows with the amount of data (模型复杂度随数据量增长)
D) They don't use probability distributions (它们不使用概率分布)

**答案**: C
**解析**: GPs are non-parametric because their effective complexity scales with the number of data points, unlike parametric models with a fixed number of parameters.
GP是非参数的，因为其有效复杂度随数据点数量缩放，不像具有固定参数数量的参数模型。

### 填空题 (Fill in the Blanks)

**Q1.3**: A Gaussian process is formally defined as f(x) ~ GP(_____, _____), where the first term represents the _____ function and the second term represents the _____ function.
**Q1.3**: 高斯过程正式定义为 f(x) ~ GP(_____, _____)，其中第一项表示_____函数，第二项表示_____函数。

**答案**: m(x), k(x,x'), 均值(mean), 协方差(covariance/kernel)

### 简答题 (Short Answer)

**Q1.4**: Give three real-world scenarios where uncertainty quantification would be more important than just getting a point prediction.
**Q1.4**: 给出三个现实世界场景，其中不确定性量化比仅获得点预测更重要。

**答案**:
1. **Medical diagnosis**: Knowing the confidence level of a diagnosis is crucial for treatment decisions
   **医疗诊断**: 了解诊断的置信水平对治疗决策至关重要

2. **Financial trading**: Understanding prediction uncertainty helps in risk management and position sizing
   **金融交易**: 理解预测不确定性有助于风险管理和仓位控制

3. **Autonomous driving**: Uncertainty in obstacle detection and path planning is critical for safety
   **自动驾驶**: 障碍物检测和路径规划中的不确定性对安全至关重要

---

## 15.2 Gaussian Process Priors (高斯过程先验)

### 选择题 (Multiple Choice)

**Q2.1**: In the RBF kernel k(x,x') = σ²f exp(-||x-x'||²/(2l²)), what happens when the length scale l becomes very small?
**Q2.1**: 在RBF核 k(x,x') = σ²f exp(-||x-x'||²/(2l²)) 中，当长度尺度l变得非常小时会发生什么？

A) The function becomes very smooth (函数变得非常平滑)
B) The function becomes very wiggly/rapidly varying (函数变得非常波动/快速变化)
C) The signal variance increases (信号方差增加)
D) The noise level decreases (噪声水平降低)

**答案**: B
**解析**: Small length scales mean that function values decorrelate quickly with distance, leading to rapidly varying, "wiggly" functions.
小的长度尺度意味着函数值随距离快速去相关，导致快速变化的"波动"函数。

**Q2.2**: What is the relationship between infinite-width neural networks and Gaussian processes?
**Q2.2**: 无限宽神经网络和高斯过程之间的关系是什么？

A) They are completely unrelated (它们完全无关)
B) As neural network width approaches infinity, it converges to a GP with the neural network kernel (当神经网络宽度趋于无穷时，它收敛到具有神经网络核的GP)
C) GPs are always better than neural networks (GP总是比神经网络更好)
D) Neural networks are a special case of GPs (神经网络是GP的特例)

**答案**: B
**解析**: This is a fundamental theoretical result that connects neural networks and Gaussian processes through the neural network kernel.
这是通过神经网络核连接神经网络和高斯过程的基本理论结果。

### 数学题 (Mathematical Problems)

**Q2.3**: Given three points x₁ = 0, x₂ = 1, x₃ = 2, compute the 3×3 covariance matrix using the RBF kernel with σ²f = 1 and l = 1.
**Q2.3**: 给定三个点 x₁ = 0, x₂ = 1, x₃ = 2，使用σ²f = 1和l = 1的RBF核计算3×3协方差矩阵。

**答案**:
$$K = \begin{pmatrix}
1 & e^{-1/2} & e^{-2} \\
e^{-1/2} & 1 & e^{-1/2} \\
e^{-2} & e^{-1/2} & 1
\end{pmatrix} \approx \begin{pmatrix}
1.000 & 0.607 & 0.135 \\
0.607 & 1.000 & 0.607 \\
0.135 & 0.607 & 1.000
\end{pmatrix}$$

### 编程题 (Programming)

**Q2.4**: Implement a function to generate samples from a GP prior using the RBF kernel.
**Q2.4**: 实现一个函数，使用RBF核从GP先验生成样本。

**答案**:
```python
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(x1, x2, signal_var=1.0, length_scale=1.0):
    """计算RBF核"""
    if np.isscalar(x1):
        x1 = np.array([x1])
    if np.isscalar(x2):
        x2 = np.array([x2])
    
    dist_sq = np.sum((x1[:, None] - x2[None, :])**2, axis=0)
    return signal_var * np.exp(-dist_sq / (2 * length_scale**2))

def sample_gp_prior(x_test, n_samples=3, signal_var=1.0, length_scale=1.0, noise_var=1e-6):
    """从GP先验生成样本"""
    n_test = len(x_test)
    
    # 计算协方差矩阵
    K = np.zeros((n_test, n_test))
    for i in range(n_test):
        for j in range(n_test):
            K[i, j] = rbf_kernel(x_test[i], x_test[j], signal_var, length_scale)
    
    # 添加少量噪声以确保数值稳定性
    K += noise_var * np.eye(n_test)
    
    # 生成样本
    samples = np.random.multivariate_normal(np.zeros(n_test), K, n_samples)
    
    return samples

# 使用示例
x_test = np.linspace(-3, 3, 50)
samples = sample_gp_prior(x_test, n_samples=5)

plt.figure(figsize=(10, 6))
for i in range(samples.shape[0]):
    plt.plot(x_test, samples[i], alpha=0.7)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Samples from GP Prior')
plt.grid(True)
plt.show()
```

---

## 15.3 Gaussian Process Inference (高斯过程推断)

### 选择题 (Multiple Choice)

**Q3.1**: In GP regression, what does the predictive variance represent?
**Q3.1**: 在GP回归中，预测方差表示什么？

A) The noise in the training data (训练数据中的噪声)
B) The uncertainty in our prediction at a test point (在测试点预测的不确定性)
C) The signal variance of the kernel (核的信号方差)
D) The error in the hyperparameters (超参数中的误差)

**答案**: B
**解析**: The predictive variance quantifies how uncertain we are about our prediction at each test point, incorporating both prior uncertainty and information gained from observations.
预测方差量化了我们对每个测试点预测的不确定性，结合了先验不确定性和从观测中获得的信息。

**Q3.2**: What is the computational complexity of exact GP inference for n training points?
**Q3.2**: 对于n个训练点，精确GP推断的计算复杂度是什么？

A) O(n) (线性)
B) O(n²) (二次)
C) O(n³) (三次)
D) O(n log n) (对数线性)

**答案**: C
**解析**: The bottleneck is matrix inversion of the n×n covariance matrix, which requires O(n³) operations.
瓶颈是n×n协方差矩阵的求逆，需要O(n³)次操作。

### 数学推导题 (Mathematical Derivation)

**Q3.3**: Derive the predictive mean equation for GP regression starting from the joint Gaussian distribution.
**Q3.3**: 从联合高斯分布开始推导GP回归的预测均值方程。

**答案**:
Given the joint distribution:
$$\begin{pmatrix} y \\ f_* \end{pmatrix} \sim \mathcal{N}\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} K + \sigma_n^2 I & K_* \\ K_*^T & K_{**} \end{pmatrix}\right)$$

Using the conditional distribution formula for multivariate Gaussians:
$$p(f_*|y) = \mathcal{N}(f_*; \mu_*, \Sigma_*)$$

Where:
$$\mu_* = K_*^T(K + \sigma_n^2 I)^{-1}y$$
$$\Sigma_* = K_{**} - K_*^T(K + \sigma_n^2 I)^{-1}K_*$$

### 编程题 (Programming)

**Q3.4**: Implement GP regression from scratch and compare with GPyTorch on a toy dataset.
**Q3.4**: 从头实现GP回归，并在玩具数据集上与GPyTorch比较。

**答案**:
```python
import numpy as np
import torch
import gpytorch
from scipy.linalg import solve, cholesky
import matplotlib.pyplot as plt

# 从头实现GP回归
class SimpleGP:
    def __init__(self, signal_var=1.0, length_scale=1.0, noise_var=0.1):
        self.signal_var = signal_var
        self.length_scale = length_scale
        self.noise_var = noise_var
    
    def rbf_kernel(self, x1, x2):
        """RBF核函数"""
        if np.isscalar(x1):
            x1 = np.array([x1])
        if np.isscalar(x2):
            x2 = np.array([x2])
        
        dist_sq = np.sum((x1[:, None] - x2[None, :])**2, axis=0)
        return self.signal_var * np.exp(-dist_sq / (2 * self.length_scale**2))
    
    def fit(self, X_train, y_train):
        """训练GP模型"""
        self.X_train = X_train
        self.y_train = y_train
        
        # 计算核矩阵
        n = len(X_train)
        self.K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.K[i, j] = self.rbf_kernel(X_train[i], X_train[j])
        
        # 添加噪声
        self.K_noise = self.K + self.noise_var * np.eye(n)
        
        # 预计算 K^(-1) y
        self.alpha = solve(self.K_noise, y_train)
    
    def predict(self, X_test):
        """预测"""
        n_test = len(X_test)
        mu = np.zeros(n_test)
        var = np.zeros(n_test)
        
        for i, x in enumerate(X_test):
            # 计算核向量
            k_star = np.array([self.rbf_kernel(x, x_train) for x_train in self.X_train])
            
            # 预测均值
            mu[i] = k_star.dot(self.alpha)
            
            # 预测方差
            v = solve(self.K_noise, k_star)
            var[i] = self.rbf_kernel(x, x) - k_star.dot(v)
        
        return mu, var

# GPyTorch实现
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 创建玩具数据集
np.random.seed(42)
torch.manual_seed(42)

X_train = np.array([0., 1., 2., 3., 4.])
y_train = np.sin(X_train) + 0.1 * np.random.randn(len(X_train))
X_test = np.linspace(-1, 5, 50)

# 从头实现的GP
gp_scratch = SimpleGP(signal_var=1.0, length_scale=1.0, noise_var=0.1)
gp_scratch.fit(X_train, y_train)
mu_scratch, var_scratch = gp_scratch.predict(X_test)

# GPyTorch实现
train_x = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.float32)
test_x = torch.tensor(X_test, dtype=torch.float32)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# 设置超参数以匹配我们的实现
model.covar_module.outputscale = torch.tensor(1.0)
model.covar_module.base_kernel.lengthscale = torch.tensor(1.0)
likelihood.noise = torch.tensor(0.1)

model.eval()
likelihood.eval()

with torch.no_grad():
    pred = likelihood(model(test_x))
    mu_gpytorch = pred.mean.numpy()
    var_gpytorch = pred.variance.numpy()

# 比较结果
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(X_test, mu_scratch, 'b-', label='Scratch Implementation')
plt.fill_between(X_test, mu_scratch - 2*np.sqrt(var_scratch), 
                 mu_scratch + 2*np.sqrt(var_scratch), alpha=0.3)
plt.scatter(X_train, y_train, c='red', marker='x', s=50, label='Training Data')
plt.title('From Scratch Implementation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(X_test, mu_gpytorch, 'g-', label='GPyTorch')
plt.fill_between(X_test, mu_gpytorch - 2*np.sqrt(var_gpytorch), 
                 mu_gpytorch + 2*np.sqrt(var_gpytorch), alpha=0.3)
plt.scatter(X_train, y_train, c='red', marker='x', s=50, label='Training Data')
plt.title('GPyTorch Implementation')
plt.legend()

plt.tight_layout()
plt.show()

# 检查数值差异
print(f"Mean difference: {np.mean(np.abs(mu_scratch - mu_gpytorch)):.6f}")
print(f"Variance difference: {np.mean(np.abs(var_scratch - var_gpytorch)):.6f}")
```

### 概念题 (Conceptual Questions)

**Q3.5**: Explain why the predictive variance decreases near training points and increases far from them.
**Q3.5**: 解释为什么预测方差在训练点附近减少，远离训练点时增加。

**答案**:
The predictive variance σ²(x*) = k(x*,x*) - k*ᵀ(K + σ²ₙI)⁻¹k* consists of two terms:

1. **Prior variance** k(x*,x*): Our uncertainty before seeing any data
2. **Information reduction** k*ᵀ(K + σ²ₙI)⁻¹k*: How much uncertainty is reduced by observations

Near training points:
- High correlation with observed data (large elements in k*)
- Large information reduction term
- Low predictive variance (high confidence)

Far from training points:
- Low correlation with observed data (small elements in k*)
- Small information reduction term
- High predictive variance (low confidence, approaching prior uncertainty)

预测方差 σ²(x*) = k(x*,x*) - k*ᵀ(K + σ²ₙI)⁻¹k* 包含两项：

1. **先验方差** k(x*,x*)：看到任何数据前的不确定性
2. **信息减少** k*ᵀ(K + σ²ₙI)⁻¹k*：观测减少了多少不确定性

在训练点附近：
- 与观测数据高度相关（k*中的大元素）
- 大的信息减少项
- 低预测方差（高置信度）

远离训练点时：
- 与观测数据低相关（k*中的小元素）
- 小的信息减少项
- 高预测方差（低置信度，接近先验不确定性）

### 应用题 (Application Problem)

**Q3.6**: Design a GP model for predicting daily temperature using historical weather data. What kernel would you choose and why?
**Q3.6**: 设计一个GP模型，使用历史天气数据预测日温度。你会选择什么核函数，为什么？

**答案**:
For daily temperature prediction, I would use a **composite kernel** combining multiple components:

1. **RBF kernel** for smooth local variations:
   k₁(t,t') = σ₁² exp(-(t-t')²/(2l₁²))

2. **Periodic kernel** for yearly seasonal patterns:
   k₂(t,t') = σ₂² exp(-2sin²(π|t-t'|/365)/(l₂²))

3. **Linear kernel** for long-term climate trends:
   k₃(t,t') = σ₃²(t-c)(t'-c)

**Combined kernel**: k(t,t') = k₁(t,t') + k₂(t,t') + k₃(t,t')

**Rationale**:
- RBF captures day-to-day smooth variations
- Periodic kernel captures seasonal cycles (summer/winter)
- Linear kernel captures climate change trends
- Each component models different aspects of temperature variation

对于日温度预测，我会使用**复合核函数**结合多个组件：

1. **RBF核**用于平滑局部变化：
2. **周期核**用于年度季节模式：
3. **线性核**用于长期气候趋势：

**组合核函数**：k(t,t') = k₁(t,t') + k₂(t,t') + k₃(t,t')

**理由**：
- RBF捕捉日常平滑变化
- 周期核捕捉季节循环（夏季/冬季）
- 线性核捕捉气候变化趋势
- 每个组件建模温度变化的不同方面 