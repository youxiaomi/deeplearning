# Mathematics for Deep Learning - Quiz
## 深度学习数学基础 - 测试题

### Quiz 1: Linear Algebra Fundamentals 线性代数基础

**Question 1.1:**
Given vectors a = [3, 4] and b = [1, 2], calculate the dot product a · b.
给定向量 a = [3, 4] 和 b = [1, 2]，计算点积 a · b。

**Answer:**
a · b = 3×1 + 4×2 = 3 + 8 = 11

**Question 1.2:**
What is the length (magnitude) of vector v = [6, 8]?
向量 v = [6, 8] 的长度（模）是多少？

**Answer:**
|v| = √(6² + 8²) = √(36 + 64) = √100 = 10

**Question 1.3:**
Given matrix A = [[2, 1], [3, 4]] and B = [[1, 0], [2, 1]], calculate A × B.
给定矩阵 A = [[2, 1], [3, 4]] 和 B = [[1, 0], [2, 1]]，计算 A × B。

**Answer:**
A × B = [[2×1 + 1×2, 2×0 + 1×1], [3×1 + 4×2, 3×0 + 4×1]]
      = [[2 + 2, 0 + 1], [3 + 8, 0 + 4]]
      = [[4, 1], [11, 4]]

**Question 1.4:**
If matrix A has eigenvalue λ = 3 with eigenvector v = [1, 2], verify that Av = λv.
如果矩阵 A 有特征值 λ = 3 和特征向量 v = [1, 2]，验证 Av = λv。

**Answer:**
This question requires knowing matrix A. The eigenvalue equation states that Av = λv = 3v = [3, 6].
这个问题需要知道矩阵 A。特征值方程表明 Av = λv = 3v = [3, 6]。

**Question 1.5:**
Explain the geometric interpretation of matrix multiplication.
解释矩阵乘法的几何意义。

**Answer:**
Matrix multiplication represents geometric transformations such as rotation, scaling, reflection, and shearing applied to vectors in space.
矩阵乘法表示几何变换，如旋转、缩放、反射和剪切，应用于空间中的向量。

### Quiz 2: Calculus 微积分

**Question 2.1:**
Find the derivative of f(x) = x³ + 2x² - 5x + 1.
求 f(x) = x³ + 2x² - 5x + 1 的导数。

**Answer:**
f'(x) = 3x² + 4x - 5

**Question 2.2:**
If f(x) = sin(x²), find f'(x) using the chain rule.
如果 f(x) = sin(x²)，使用链式法则求 f'(x)。

**Answer:**
Let u = x², then f(x) = sin(u)
f'(x) = cos(u) × du/dx = cos(x²) × 2x = 2x cos(x²)

**Question 2.3:**
For function f(x,y) = x²y + 3xy², find ∂f/∂x and ∂f/∂y.
对于函数 f(x,y) = x²y + 3xy²，求 ∂f/∂x 和 ∂f/∂y。

**Answer:**
∂f/∂x = 2xy + 3y²
∂f/∂y = x² + 6xy

**Question 2.4:**
What is the gradient of f(x,y) = x² + y² at point (2, 3)?
函数 f(x,y) = x² + y² 在点 (2, 3) 处的梯度是什么？

**Answer:**
∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]
At (2, 3): ∇f = [2×2, 2×3] = [4, 6]

**Question 2.5:**
Explain how gradient descent uses derivatives to minimize a function.
解释梯度下降如何使用导数来最小化函数。

**Answer:**
Gradient descent uses the gradient (vector of partial derivatives) to find the direction of steepest increase, then moves in the opposite direction to minimize the function. The magnitude of the gradient indicates how steep the slope is.
梯度下降使用梯度（偏导数向量）找到最陡峭增长的方向，然后朝相反方向移动以最小化函数。梯度的大小表示坡度有多陡。

### Quiz 3: Probability and Statistics 概率与统计

**Question 3.1:**
A fair six-sided die is rolled. What is the probability of getting an even number?
掷一个公平的六面骰子。得到偶数的概率是多少？

**Answer:**
Even numbers: {2, 4, 6}
P(even) = 3/6 = 1/2 = 0.5

**Question 3.2:**
If X ~ N(μ=100, σ²=225), what is the probability that X is within one standard deviation of the mean?
如果 X ~ N(μ=100, σ²=225)，X 在均值一个标准差范围内的概率是多少？

**Answer:**
σ = √225 = 15
P(85 ≤ X ≤ 115) ≈ 0.68 (68% rule for normal distribution)

**Question 3.3:**
Given sample data: [12, 15, 18, 14, 16], calculate the sample mean and sample standard deviation.
给定样本数据：[12, 15, 18, 14, 16]，计算样本均值和样本标准差。

**Answer:**
Sample mean: x̄ = (12 + 15 + 18 + 14 + 16)/5 = 75/5 = 15
Sample variance: s² = Σ(xi - x̄)²/(n-1) = [(12-15)² + (15-15)² + (18-15)² + (14-15)² + (16-15)²]/4
                   = [9 + 0 + 9 + 1 + 1]/4 = 20/4 = 5
Sample standard deviation: s = √5 ≈ 2.24

**Question 3.4:**
What is the difference between population and sample statistics?
总体统计量和样本统计量的区别是什么？

**Answer:**
Population statistics describe the entire population (e.g., population mean μ), while sample statistics describe a subset of the population (e.g., sample mean x̄). Sample statistics are estimates of population parameters.
总体统计量描述整个总体（如总体均值 μ），而样本统计量描述总体的一个子集（如样本均值 x̄）。样本统计量是总体参数的估计。

**Question 3.5:**
Explain the Central Limit Theorem in simple terms.
用简单的话解释中心极限定理。

**Answer:**
The Central Limit Theorem states that when you take many samples from any population and calculate their means, those sample means will form a normal distribution, regardless of the original population's distribution, as long as the sample size is large enough.
中心极限定理指出，当你从任何总体中取许多样本并计算它们的均值时，只要样本大小足够大，这些样本均值将形成正态分布，无论原始总体的分布如何。

### Quiz 4: Information Theory 信息论

**Question 4.1:**
Calculate the entropy of a fair coin (P(H) = 0.5, P(T) = 0.5).
计算公平硬币的熵（P(H) = 0.5, P(T) = 0.5）。

**Answer:**
H = -Σ P(x) log₂ P(x)
H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
H = -(0.5 × (-1) + 0.5 × (-1)) = -(-0.5 - 0.5) = 1 bit

**Question 4.2:**
Which has higher entropy: a biased coin with P(H) = 0.9, P(T) = 0.1, or a fair coin?
哪个有更高的熵：偏向硬币 P(H) = 0.9, P(T) = 0.1，还是公平硬币？

**Answer:**
Fair coin entropy = 1 bit (from previous question)
Biased coin entropy = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1))
                    = -(0.9 × (-0.152) + 0.1 × (-3.322))
                    = -(-0.137 - 0.332) = 0.469 bits
Fair coin has higher entropy.

**Question 4.3:**
What is cross-entropy loss and why is it used in classification?
什么是交叉熵损失，为什么在分类中使用它？

**Answer:**
Cross-entropy loss measures the difference between predicted probabilities and true labels. It's used in classification because it penalizes confident wrong predictions more heavily and provides smooth gradients for optimization.
交叉熵损失测量预测概率和真实标签之间的差异。它在分类中使用，因为它对自信的错误预测施加更重的惩罚，并为优化提供平滑的梯度。

**Question 4.4:**
If the true distribution is P = [0.7, 0.2, 0.1] and predicted distribution is Q = [0.6, 0.3, 0.1], calculate the KL divergence KL(P||Q).
如果真实分布是 P = [0.7, 0.2, 0.1]，预测分布是 Q = [0.6, 0.3, 0.1]，计算 KL 散度 KL(P||Q)。

**Answer:**
KL(P||Q) = Σ P(i) × log(P(i)/Q(i))
         = 0.7 × log(0.7/0.6) + 0.2 × log(0.2/0.3) + 0.1 × log(0.1/0.1)
         = 0.7 × log(1.167) + 0.2 × log(0.667) + 0.1 × 0
         = 0.7 × 0.154 + 0.2 × (-0.405) + 0.1 × 0
         = 0.108 - 0.081 + 0 = 0.027

**Question 4.5:**
Explain mutual information and give a real-world example.
解释互信息并给出一个现实世界的例子。

**Answer:**
Mutual information measures how much knowing one variable tells us about another. Example: the mutual information between "cloudy weather" and "rain" is high because knowing it's cloudy gives us significant information about the likelihood of rain.
互信息测量知道一个变量对另一个变量提供多少信息。例子：「多云天气」和「下雨」之间的互信息很高，因为知道多云能为我们提供关于下雨可能性的重要信息。

### Quiz 5: Maximum Likelihood Estimation 最大似然估计

**Question 5.1:**
You flip a coin 10 times and get 7 heads. What is the maximum likelihood estimate for the probability of heads?
你掷硬币10次得到7次正面。正面朝上概率的最大似然估计是多少？

**Answer:**
MLE for probability = Number of successes / Total trials = 7/10 = 0.7

**Question 5.2:**
Given samples from a normal distribution: [98, 102, 100, 96, 104], find the MLE for the mean.
给定来自正态分布的样本：[98, 102, 100, 96, 104]，求均值的最大似然估计。

**Answer:**
MLE for mean = Sample mean = (98 + 102 + 100 + 96 + 104)/5 = 500/5 = 100

**Question 5.3:**
Why is the sample mean the MLE for the population mean in a normal distribution?
为什么样本均值是正态分布中总体均值的最大似然估计？

**Answer:**
For a normal distribution, the likelihood function is maximized when the derivative with respect to μ equals zero. This occurs when μ equals the sample mean, making the sample mean the MLE.
对于正态分布，当关于 μ 的导数等于零时，似然函数达到最大值。这发生在 μ 等于样本均值时，使样本均值成为最大似然估计。

**Question 5.4:**
What is the relationship between MLE and minimizing negative log-likelihood?
最大似然估计和最小化负对数似然之间的关系是什么？

**Answer:**
Maximizing likelihood is equivalent to minimizing negative log-likelihood because the logarithm is a monotonic function. Taking the negative makes it a minimization problem, which is more convenient for optimization algorithms.
最大化似然等价于最小化负对数似然，因为对数是单调函数。取负数使其成为最小化问题，这对优化算法更方便。

**Question 5.5:**
In what situations might MLE not be the best estimation method?
在什么情况下最大似然估计可能不是最佳估计方法？

**Answer:**
MLE might not be best when: (1) sample size is very small, (2) there's prior knowledge that should be incorporated (use MAP instead), (3) robustness to outliers is needed, or (4) when overfitting is a concern.
在以下情况下最大似然估计可能不是最佳的：(1) 样本大小很小，(2) 有应该纳入的先验知识（使用MAP代替），(3) 需要对异常值的鲁棒性，或 (4) 当过拟合是一个问题时。

### Quiz 6: Distributions 分布

**Question 6.1:**
What are the parameters of a normal distribution and what do they represent?
正态分布的参数是什么，它们代表什么？

**Answer:**
Normal distribution has two parameters: μ (mean) represents the center/location of the distribution, and σ² (variance) or σ (standard deviation) represents the spread/width of the distribution.
正态分布有两个参数：μ（均值）表示分布的中心/位置，σ²（方差）或 σ（标准差）表示分布的展开/宽度。

**Question 6.2:**
When would you use a Poisson distribution? Give an example.
什么时候使用泊松分布？给出一个例子。

**Answer:**
Poisson distribution models the number of events occurring in a fixed interval of time or space when events occur independently at a constant rate. Example: number of customers arriving at a store per hour, number of typos per page.
泊松分布建模在固定时间或空间间隔内发生的事件数量，当事件以恒定速率独立发生时。例子：每小时到达商店的客户数量，每页的错字数量。

**Question 6.3:**
What is the difference between a probability mass function (PMF) and a probability density function (PDF)?
概率质量函数（PMF）和概率密度函数（PDF）的区别是什么？

**Answer:**
PMF is used for discrete random variables and gives the probability that the variable equals a specific value. PDF is used for continuous random variables and gives the probability density, not probability itself.
PMF 用于离散随机变量，给出变量等于特定值的概率。PDF 用于连续随机变量，给出概率密度，而不是概率本身。

**Question 6.4:**
If X ~ Uniform(0, 10), what is P(X ≤ 3)?
如果 X ~ Uniform(0, 10)，P(X ≤ 3) 是多少？

**Answer:**
For uniform distribution on [0, 10], P(X ≤ 3) = (3 - 0)/(10 - 0) = 3/10 = 0.3

**Question 6.5:**
What is the relationship between the exponential and Poisson distributions?
指数分布和泊松分布之间的关系是什么？

**Answer:**
If events follow a Poisson process (Poisson distribution for counts), then the time between events follows an exponential distribution. They are complementary: Poisson counts events, exponential models waiting times.
如果事件遵循泊松过程（计数的泊松分布），那么事件之间的时间遵循指数分布。它们是互补的：泊松计数事件，指数建模等待时间。

### Quiz 7: Eigendecomposition 特征分解

**Question 7.1:**
What is an eigenvector and eigenvalue?
什么是特征向量和特征值？

**Answer:**
An eigenvector is a non-zero vector that, when a linear transformation is applied, only changes by a scalar factor. The eigenvalue is that scalar factor. Mathematically: Av = λv, where v is the eigenvector and λ is the eigenvalue.
特征向量是一个非零向量，当应用线性变换时，只按标量因子改变。特征值就是那个标量因子。数学上：Av = λv，其中 v 是特征向量，λ 是特征值。

**Question 7.2:**
Given matrix A = [[4, 1], [0, 3]], find the eigenvalues.
给定矩阵 A = [[4, 1], [0, 3]]，求特征值。

**Answer:**
For eigenvalues, solve det(A - λI) = 0
det([[4-λ, 1], [0, 3-λ]]) = (4-λ)(3-λ) - 0 = 0
(4-λ)(3-λ) = 0
λ₁ = 4, λ₂ = 3

**Question 7.3:**
What does it mean geometrically when a matrix has eigenvalue λ = 0?
当矩阵的特征值 λ = 0 时，几何上意味着什么？

**Answer:**
An eigenvalue of 0 means the matrix is singular (non-invertible) and maps some non-zero vectors to the zero vector. Geometrically, it collapses space along the direction of the corresponding eigenvector.
特征值为 0 意味着矩阵是奇异的（不可逆的），并将一些非零向量映射到零向量。几何上，它沿着相应特征向量的方向压缩空间。

**Question 7.4:**
How are eigendecomposition and Principal Component Analysis (PCA) related?
特征分解和主成分分析（PCA）如何相关？

**Answer:**
PCA uses eigendecomposition of the covariance matrix. The eigenvectors become the principal components (directions of maximum variance), and eigenvalues represent the amount of variance explained by each component.
PCA 使用协方差矩阵的特征分解。特征向量成为主成分（最大方差的方向），特征值表示每个成分解释的方差量。

**Question 7.5:**
Why are eigenvalues and eigenvectors important in deep learning?
为什么特征值和特征向量在深度学习中很重要？

**Answer:**
They help understand optimization landscapes, analyze network stability, perform dimensionality reduction (PCA), initialize weights, and understand how gradients flow through networks. They're also crucial in techniques like spectral normalization.
它们帮助理解优化景观、分析网络稳定性、执行降维（PCA）、初始化权重，以及理解梯度如何在网络中流动。它们在谱归一化等技术中也很关键。

### Quiz 8: Naive Bayes 朴素贝叶斯

**Question 8.1:**
State Bayes' theorem and explain each component.
陈述贝叶斯定理并解释每个组成部分。

**Answer:**
P(A|B) = P(B|A) × P(A) / P(B)
Where:
- P(A|B): posterior probability of A given B
- P(B|A): likelihood of B given A  
- P(A): prior probability of A
- P(B): marginal probability of B

**Question 8.2:**
Why is Naive Bayes called "naive"?
为什么朴素贝叶斯被称为"朴素"？

**Answer:**
It's called "naive" because it assumes that all features are conditionally independent given the class label. This assumption is often unrealistic in practice but simplifies computation significantly.
它被称为"朴素"是因为它假设在给定类标签的情况下所有特征都是条件独立的。这个假设在实践中通常不现实，但显著简化了计算。

**Question 8.3:**
Given P(Spam) = 0.3, P(contains "free"|Spam) = 0.8, P(contains "free"|Ham) = 0.1, and P(Ham) = 0.7, what is P(Spam|contains "free")?
给定 P(垃圾邮件) = 0.3，P(包含"free"|垃圾邮件) = 0.8，P(包含"free"|正常邮件) = 0.1，P(正常邮件) = 0.7，求 P(垃圾邮件|包含"free")。

**Answer:**
P(contains "free") = P(contains "free"|Spam) × P(Spam) + P(contains "free"|Ham) × P(Ham)
                   = 0.8 × 0.3 + 0.1 × 0.7 = 0.24 + 0.07 = 0.31

P(Spam|contains "free") = P(contains "free"|Spam) × P(Spam) / P(contains "free")
                        = (0.8 × 0.3) / 0.31 = 0.24 / 0.31 ≈ 0.77

**Question 8.4:**
What are the advantages and disadvantages of Naive Bayes classifiers?
朴素贝叶斯分类器的优点和缺点是什么？

**Answer:**
Advantages: Simple, fast, works well with small datasets, handles multiple classes naturally, good baseline.
Disadvantages: Strong independence assumption, can be outperformed by more sophisticated methods, sensitive to skewed data.
优点：简单、快速、在小数据集上效果好、自然处理多个类别、良好的基线。
缺点：强独立性假设、可能被更复杂的方法超越、对倾斜数据敏感。

**Question 8.5:**
In what scenarios is Naive Bayes particularly effective?
在什么场景下朴素贝叶斯特别有效？

**Answer:**
Text classification (spam detection, sentiment analysis), medical diagnosis with independent symptoms, real-time prediction where speed is crucial, when training data is limited, and as a baseline for comparison.
文本分类（垃圾邮件检测、情感分析）、具有独立症状的医学诊断、需要速度的实时预测、训练数据有限时，以及作为比较的基线。

### Quiz 9: Integral Calculus 积分学

**Question 9.1:**
Find ∫(3x² + 2x - 1)dx.
求 ∫(3x² + 2x - 1)dx。

**Answer:**
∫(3x² + 2x - 1)dx = x³ + x² - x + C

**Question 9.2:**
Evaluate the definite integral ∫₀² x² dx.
计算定积分 ∫₀² x² dx。

**Answer:**
∫₀² x² dx = [x³/3]₀² = 8/3 - 0 = 8/3

**Question 9.3:**
How is integration used in probability theory?
积分如何在概率论中使用？

**Answer:**
Integration is used to calculate probabilities for continuous distributions (area under PDF curves), find cumulative distribution functions, calculate expected values, and determine moments of distributions.
积分用于计算连续分布的概率（PDF曲线下的面积）、找到累积分布函数、计算期望值和确定分布的矩。

**Question 9.4:**
If f(x) = 2x on [0,3], verify that this is a valid probability density function.
如果 f(x) = 2x 在 [0,3] 上，验证这是一个有效的概率密度函数。

**Answer:**
For a valid PDF: (1) f(x) ≥ 0 for all x, and (2) ∫f(x)dx = 1
(1) f(x) = 2x ≥ 0 for x ∈ [0,3] ✗ (This is not valid as written)
Need f(x) = cx where ∫₀³ cx dx = 1
∫₀³ cx dx = c[x²/2]₀³ = c(9/2) = 1, so c = 2/9
Valid PDF: f(x) = (2/9)x on [0,3]

**Question 9.5:**
Explain the Fundamental Theorem of Calculus in your own words.
用你自己的话解释微积分基本定理。

**Answer:**
The Fundamental Theorem of Calculus connects differentiation and integration. It states that integration and differentiation are inverse operations. If you integrate a function and then differentiate the result, you get back the original function.
微积分基本定理连接了微分和积分。它表明积分和微分是逆运算。如果你对一个函数积分然后对结果微分，你会得到原来的函数。

### Quiz 10: Complex Mixed Problems 复杂综合问题

**Question 10.1:**
A neural network uses sigmoid activation σ(z) = 1/(1+e^(-z)). Find dσ/dz and explain why this derivative is important in backpropagation.
神经网络使用 sigmoid 激活函数 σ(z) = 1/(1+e^(-z))。求 dσ/dz 并解释为什么这个导数在反向传播中很重要。

**Answer:**
dσ/dz = σ(z)(1-σ(z))

This derivative is crucial in backpropagation because:
1. It's used to compute gradients through the chain rule
2. It has a maximum value of 0.25 (when σ(z) = 0.5), which can cause vanishing gradients
3. It becomes very small when σ(z) approaches 0 or 1, making learning slow

**Question 10.2:**
Explain how the chain rule enables backpropagation in neural networks.
解释链式法则如何在神经网络中实现反向传播。

**Answer:**
The chain rule allows us to compute gradients of composite functions by multiplying partial derivatives along the computational path. In neural networks, we compute ∂Loss/∂weight by multiplying gradients backward through layers: ∂Loss/∂weight = (∂Loss/∂output) × (∂output/∂hidden) × (∂hidden/∂weight).
链式法则允许我们通过沿计算路径乘以偏导数来计算复合函数的梯度。在神经网络中，我们通过向后传播通过层来计算 ∂Loss/∂weight：∂Loss/∂weight = (∂Loss/∂output) × (∂output/∂hidden) × (∂hidden/∂weight)。

**Question 10.3:**
Why is the normal distribution so important in machine learning and statistics?
为什么正态分布在机器学习和统计学中如此重要？

**Answer:**
1. Central Limit Theorem: Sample means approach normal distribution
2. Many natural phenomena follow normal distributions
3. Mathematical tractability: Easy to work with analytically
4. Maximum entropy distribution for given mean and variance
5. Foundation for many statistical tests and confidence intervals
6. Gaussian processes and Bayesian methods rely on it

**Question 10.4:**
How do eigenvalues relate to the condition number of a matrix, and why does this matter for optimization?
特征值如何与矩阵的条件数相关，为什么这对优化很重要？

**Answer:**
Condition number = λ_max/λ_min (ratio of largest to smallest eigenvalue). High condition numbers indicate:
1. Ill-conditioned optimization problems
2. Different curvatures in different directions
3. Slow convergence in gradient descent
4. Numerical instability
This matters because it affects convergence speed and stability of optimization algorithms.

**Question 10.5:**
Design a probability problem that demonstrates the difference between joint, marginal, and conditional probability.
设计一个概率问题来展示联合概率、边际概率和条件概率之间的区别。

**Answer:**
Weather and Traffic Example:
- Let R = "Rain", T = "Traffic Jam"
- Joint: P(R and T) = 0.15 (probability of both rain and traffic jam)
- Marginal: P(R) = 0.3 (probability of rain), P(T) = 0.4 (probability of traffic jam)  
- Conditional: P(T|R) = 0.5 (probability of traffic jam given rain)

Relationship: P(R and T) = P(T|R) × P(R) = 0.5 × 0.3 = 0.15 ✓

This problem demonstrates how these three types of probability relate through the multiplication rule and how conditional probability changes our assessment when we have additional information. 