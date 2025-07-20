# 推荐系统测试题 Recommender Systems Quiz

## 第18.1章：推荐系统概述 Overview of Recommender Systems

### 题目1：协同过滤基础概念 Collaborative Filtering Basics

**问题 Question:**
Explain the difference between user-based and item-based collaborative filtering. Give a real-world example for each approach.

解释基于用户和基于物品的协同过滤之间的区别。为每种方法给出一个现实世界的例子。

**答案 Answer:**

**User-based Collaborative Filtering 基于用户的协同过滤:**
- **Concept 概念**: Finds users with similar preferences and recommends items that similar users have liked.
  找到具有相似偏好的用户，推荐相似用户喜欢的物品。
- **Example 例子**: Netflix finds that you and another user both rated "Inception" and "The Matrix" highly. If that user also loved "Interstellar", Netflix recommends "Interstellar" to you.
  Netflix发现你和另一个用户都对《盗梦空间》和《黑客帝国》给出高分。如果那个用户也很喜欢《星际穿越》，Netflix就会向你推荐《星际穿越》。

**Item-based Collaborative Filtering 基于物品的协同过滤:**
- **Concept 概念**: Recommends items similar to those the user has previously liked.
  推荐与用户之前喜欢的物品相似的物品。
- **Example 例子**: Amazon notices you bought "Harry Potter" books. Since people who buy Harry Potter also tend to buy "Lord of the Rings", Amazon recommends LOTR books to you.
  亚马逊注意到你购买了《哈利·波特》书籍。由于购买哈利·波特的人也倾向于购买《指环王》，亚马逊向你推荐《指环王》书籍。

### 题目2：显式反馈vs隐式反馈 Explicit vs Implicit Feedback

**问题 Question:**
A music streaming platform wants to improve its recommendation system. They have the following data:
- User ratings (1-5 stars) for songs
- Play count for each song per user  
- Skip rate (percentage of song played before skipping)
- User's saved playlists

Classify each data type as explicit or implicit feedback and explain how each could be used in recommendations.

音乐流媒体平台想要改进其推荐系统。他们有以下数据：
- 用户对歌曲的评分（1-5星）
- 每个用户每首歌的播放次数
- 跳过率（跳过前播放歌曲的百分比）
- 用户保存的播放列表

将每种数据类型分类为显式或隐式反馈，并解释每种如何用于推荐。

**答案 Answer:**

1. **User ratings (1-5 stars) 用户评分（1-5星）**
   - **Type 类型**: Explicit Feedback 显式反馈
   - **Usage 用法**: Direct preference indication. Songs rated 4-5 stars indicate strong positive preference.
     直接偏好指示。评分4-5星的歌曲表示强烈的正向偏好。

2. **Play count 播放次数**
   - **Type 类型**: Implicit Feedback 隐式反馈  
   - **Usage 用法**: Higher play counts suggest preference. Could weight by recency (recent plays more important).
     较高的播放次数表明偏好。可以按近期加权（最近播放更重要）。

3. **Skip rate 跳过率**
   - **Type 类型**: Implicit Feedback 隐式反馈
   - **Usage 用法**: Low skip rate (high completion rate) indicates positive preference. High skip rate suggests negative preference.
     低跳过率（高完成率）表示正向偏好。高跳过率表明负向偏好。

4. **Saved playlists 保存的播放列表**
   - **Type 类型**: Implicit Feedback 隐式反馈
   - **Usage 用法**: Songs added to playlists indicate strong positive preference. Can analyze playlist themes for genre preferences.
     添加到播放列表的歌曲表示强烈的正向偏好。可以分析播放列表主题以了解流派偏好。

## 第18.2章：评分预测 Rating Prediction

### 题目3：计算用户相似度 Computing User Similarity

**问题 Question:**
Given the following user-item rating matrix, calculate the cosine similarity between User A and User B. Show your work step by step.

给定以下用户-物品评分矩阵，计算用户A和用户B之间的余弦相似度。逐步展示你的计算过程。

```
      Item1  Item2  Item3  Item4
UserA   5      3      0      1
UserB   4      0      0      1  
UserC   1      1      0      5
```

**答案 Answer:**

**Step 1: Extract rating vectors 提取评分向量**
- User A: [5, 3, 0, 1]
- User B: [4, 0, 0, 1]

**Step 2: Calculate dot product 计算点积**
Dot product = (5×4) + (3×0) + (0×0) + (1×1) = 20 + 0 + 0 + 1 = 21

**Step 3: Calculate magnitudes 计算向量长度**
- |User A| = √(5² + 3² + 0² + 1²) = √(25 + 9 + 0 + 1) = √35 ≈ 5.916
- |User B| = √(4² + 0² + 0² + 1²) = √(16 + 0 + 0 + 1) = √17 ≈ 4.123

**Step 4: Calculate cosine similarity 计算余弦相似度**
Cosine similarity = Dot product / (|User A| × |User B|)
= 21 / (5.916 × 4.123) = 21 / 24.382 ≈ 0.861

**Result 结果**: The cosine similarity between User A and User B is approximately 0.861, indicating high similarity.
用户A和用户B之间的余弦相似度约为0.861，表示高度相似。

### 题目4：评估指标 Evaluation Metrics

**问题 Question:**
A recommendation system made the following predictions vs actual ratings:

推荐系统做出以下预测与实际评分：

```
Predicted: [4.2, 3.8, 2.1, 4.5, 1.9]
Actual:    [4.0, 4.0, 2.0, 5.0, 2.0]
```

Calculate the MAE (Mean Absolute Error) and RMSE (Root Mean Square Error). Which metric is more sensitive to outliers and why?

计算MAE（平均绝对误差）和RMSE（均方根误差）。哪个指标对异常值更敏感，为什么？

**答案 Answer:**

**Calculating MAE 计算MAE:**
MAE = (1/n) × Σ|predicted - actual|
= (1/5) × [|4.2-4.0| + |3.8-4.0| + |2.1-2.0| + |4.5-5.0| + |1.9-2.0|]
= (1/5) × [0.2 + 0.2 + 0.1 + 0.5 + 0.1]
= (1/5) × 1.1 = 0.22

**Calculating RMSE 计算RMSE:**
MSE = (1/n) × Σ(predicted - actual)²
= (1/5) × [(4.2-4.0)² + (3.8-4.0)² + (2.1-2.0)² + (4.5-5.0)² + (1.9-2.0)²]
= (1/5) × [0.04 + 0.04 + 0.01 + 0.25 + 0.01]
= (1/5) × 0.35 = 0.07

RMSE = √MSE = √0.07 ≈ 0.264

**Sensitivity to outliers 对异常值的敏感性:**
RMSE is more sensitive to outliers because it squares the errors before averaging. Large errors get disproportionately more weight. In this example, the error of 0.5 for item 4 contributes 0.25 to MSE (much larger impact) while only contributing 0.5 to MAE.

RMSE对异常值更敏感，因为它在平均之前对误差进行平方。大误差获得不成比例的更大权重。在这个例子中，第4个物品的0.5误差对MSE贡献0.25（影响更大），而对MAE只贡献0.5。

## 第18.3章：矩阵分解 Matrix Factorization

### 题目5：矩阵分解概念 Matrix Factorization Concepts

**问题 Question:**
Explain the mathematical intuition behind matrix factorization for collaborative filtering. If we have a user-item rating matrix R of size m×n, and we decompose it into P (m×k) and Q (n×k), what do the k dimensions represent? How does the number of factors k affect model performance?

解释协同过滤中矩阵分解的数学直觉。如果我们有一个大小为m×n的用户-物品评分矩阵R，我们将其分解为P（m×k）和Q（n×k），k维代表什么？因子数量k如何影响模型性能？

**答案 Answer:**

**Mathematical Intuition 数学直觉:**
Matrix factorization assumes that user preferences and item characteristics can be explained by a small number of latent factors. Instead of storing the full m×n rating matrix, we represent:
- Each user as a k-dimensional vector in P (user preferences for each factor)
- Each item as a k-dimensional vector in Q (item characteristics for each factor)

矩阵分解假设用户偏好和物品特征可以用少量潜在因子来解释。我们不存储完整的m×n评分矩阵，而是表示：
- P中的每个用户作为k维向量（用户对每个因子的偏好）
- Q中的每个物品作为k维向量（物品在每个因子上的特征）

**What k dimensions represent k维代表什么:**
The k dimensions represent latent factors that capture hidden patterns in user-item interactions. For movies, these might be:
- Factor 1: Action vs Drama preference
- Factor 2: Mainstream vs Indie preference  
- Factor 3: Comedy vs Serious tone preference

k维代表捕获用户-物品交互中隐藏模式的潜在因子。对于电影，这些可能是：
- 因子1：动作片vs剧情片偏好
- 因子2：主流vs独立电影偏好
- 因子3：喜剧vs严肃调性偏好

**Impact of k on performance k对性能的影响:**
- **Too small k**: Underfitting - cannot capture enough complexity in user preferences
- **Too large k**: Overfitting - model memorizes training data but generalizes poorly
- **Optimal k**: Balances expressiveness and generalization, typically found through cross-validation

- **k太小**: 欠拟合 - 无法捕获用户偏好中足够的复杂性
- **k太大**: 过拟合 - 模型记住训练数据但泛化能力差
- **最优k**: 平衡表达能力和泛化能力，通常通过交叉验证找到

### 题目6：梯度下降更新 Gradient Descent Updates

**问题 Question:**
In matrix factorization with SGD, derive the gradient update rules for user factors p_u and item factors q_i when minimizing the squared error loss with L2 regularization:

在使用SGD的矩阵分解中，推导用户因子p_u和物品因子q_i的梯度更新规则，当最小化带L2正则化的平方误差损失时：

Loss = (r_ui - p_u^T * q_i)² + λ(||p_u||² + ||q_i||²)

**答案 Answer:**

**Step 1: Define variables 定义变量**
- r_ui: actual rating 实际评分
- p_u: user factor vector 用户因子向量
- q_i: item factor vector 物品因子向量
- λ: regularization parameter 正则化参数

**Step 2: Calculate prediction error 计算预测误差**
e_ui = r_ui - p_u^T * q_i

**Step 3: Calculate gradients 计算梯度**

For user factors p_u:
∂L/∂p_u = ∂/∂p_u [(r_ui - p_u^T * q_i)² + λ||p_u||²]
         = -2(r_ui - p_u^T * q_i) * q_i + 2λp_u
         = -2e_ui * q_i + 2λp_u

For item factors q_i:
∂L/∂q_i = ∂/∂q_i [(r_ui - p_u^T * q_i)² + λ||q_i||²]
         = -2(r_ui - p_u^T * q_i) * p_u + 2λq_i
         = -2e_ui * p_u + 2λq_i

**Step 4: Update rules 更新规则**

p_u ← p_u - α * (∂L/∂p_u) = p_u - α * (-2e_ui * q_i + 2λp_u)
    = p_u + α * (2e_ui * q_i - 2λp_u)
    = p_u + 2α * (e_ui * q_i - λp_u)

q_i ← q_i - α * (∂L/∂q_i) = q_i - α * (-2e_ui * p_u + 2λq_i)
    = q_i + α * (2e_ui * p_u - 2λq_i)
    = q_i + 2α * (e_ui * p_u - λq_i)

Where α is the learning rate. The factor of 2 is often absorbed into the learning rate.
其中α是学习率。因子2通常被吸收到学习率中。

## 第18.6章：神经协同过滤 Neural Collaborative Filtering

### 题目7：NeuMF架构理解 NeuMF Architecture Understanding

**问题 Question:**
Compare the GMF (Generalized Matrix Factorization) and MLP components in NeuMF. How do they differ in modeling user-item interactions? Why is the fusion of both components beneficial?

比较NeuMF中的GMF（广义矩阵分解）和MLP组件。它们在建模用户-物品交互方面有何不同？为什么融合两个组件是有益的？

**答案 Answer:**

**GMF Component GMF组件:**
- **Operation 操作**: Element-wise multiplication of user and item embeddings (Hadamard product)
  用户和物品嵌入的逐元素乘法（Hadamard积）
- **Mathematical form 数学形式**: output = W * (p_u ⊙ q_i) + b
- **Modeling capability 建模能力**: Captures linear interactions, similar to traditional matrix factorization
  捕获线性交互，类似于传统矩阵分解
- **Interpretability 可解释性**: More interpretable, each dimension can be understood as a latent factor
  更具可解释性，每个维度可以理解为一个潜在因子

**MLP Component MLP组件:**
- **Operation 操作**: Concatenation of user and item embeddings fed through multi-layer perceptron
  用户和物品嵌入的连接通过多层感知机
- **Mathematical form 数学形式**: output = MLP([p_u; q_i])
- **Modeling capability 建模能力**: Captures non-linear, complex interactions between user and item features
  捕获用户和物品特征之间的非线性、复杂交互
- **Expressiveness 表达能力**: More expressive but less interpretable
  更具表达力但可解释性较差

**Benefits of Fusion 融合的好处:**
1. **Complementary strengths 互补优势**: GMF provides stable linear modeling while MLP captures complex patterns
   GMF提供稳定的线性建模，而MLP捕获复杂模式

2. **Robustness 鲁棒性**: If one component overfits, the other can provide regularization
   如果一个组件过拟合，另一个可以提供正则化

3. **Best of both worlds 两全其美**: Combines interpretability of matrix factorization with expressiveness of neural networks
   结合矩阵分解的可解释性和神经网络的表达能力

4. **Performance 性能**: Empirically shown to outperform either component alone
   经验表明优于单独使用任一组件

### 题目8：负采样策略 Negative Sampling Strategy

**问题 Question:**
In training NeuMF for ranking, explain why negative sampling is necessary. Compare three negative sampling strategies:
1. Random negative sampling
2. Popularity-based negative sampling  
3. Hard negative sampling

For each strategy, discuss the advantages, disadvantages, and when you would use it.

在训练NeuMF进行排序时，解释为什么需要负采样。比较三种负采样策略：
1. 随机负采样
2. 基于流行度的负采样
3. 困难负采样

对于每种策略，讨论优缺点以及何时使用。

**答案 Answer:**

**Why Negative Sampling is Necessary 为什么需要负采样:**
In ranking problems, we typically only have positive feedback (items users interacted with). To train a ranking model, we need negative examples to teach the model what users DON'T prefer. This creates a contrastive learning setup where the model learns to rank positive items higher than negative items.

在排序问题中，我们通常只有正向反馈（用户交互的物品）。为了训练排序模型，我们需要负样本来教导模型用户不喜欢什么。这创建了一个对比学习设置，模型学习将正向物品排名高于负向物品。

**1. Random Negative Sampling 随机负采样:**
- **Method 方法**: Randomly select items the user hasn't interacted with
  随机选择用户未交互的物品
- **Advantages 优点**: 
  - Simple to implement 实现简单
  - Unbiased sampling 无偏采样
  - Computationally efficient 计算高效
- **Disadvantages 缺点**: 
  - May select obviously irrelevant items (too easy negatives)
  可能选择明显不相关的物品（太容易的负样本）
  - Doesn't focus learning on challenging cases
  不专注于挑战性案例的学习
- **When to use 何时使用**: Early training stages, when computational efficiency is critical
  早期训练阶段，当计算效率至关重要时

**2. Popularity-based Negative Sampling 基于流行度的负采样:**
- **Method 方法**: Sample negatives proportional to item popularity (more popular items more likely to be selected)
  按物品流行度比例采样负样本（更流行的物品更可能被选择）
- **Advantages 优点**:
  - Avoids overly obscure negatives 避免过于冷门的负样本
  - Mimics real-world recommendation scenarios 模拟真实世界推荐场景
  - Reduces selection bias 减少选择偏差
- **Disadvantages 缺点**:
  - May be too conservative 可能过于保守
  - Popular items might be harder negatives for some users
  对某些用户来说，流行物品可能是更困难的负样本
- **When to use 何时使用**: When item popularity distribution is skewed, for more realistic training
  当物品流行度分布倾斜时，用于更现实的训练

**3. Hard Negative Sampling 困难负采样:**
- **Method 方法**: Select negative items with high predicted scores (items the model currently thinks the user might like)
  选择具有高预测分数的负向物品（模型当前认为用户可能喜欢的物品）
- **Advantages 优点**:
  - Forces model to learn fine-grained distinctions 强制模型学习细粒度区别
  - Improves discriminative power 提高判别能力
  - Focuses on challenging cases 专注于挑战性案例
- **Disadvantages 缺点**:
  - Computationally expensive (requires inference for sampling)
  计算昂贵（采样需要推理）
  - May lead to overly aggressive training 可能导致过于激进的训练
  - Risk of label noise (hard negatives might actually be positives)
  标签噪声风险（困难负样本实际上可能是正样本）
- **When to use 何时使用**: Later training stages, when you want to fine-tune model performance
  训练后期阶段，当你想要微调模型性能时

## 第18.7章：序列感知推荐 Sequence-Aware Recommender Systems

### 题目9：序列建模 Sequence Modeling

**问题 Question:**
Explain how RNN-based sequence-aware recommender systems capture temporal dynamics in user behavior. What are the key differences between using RNN vs Transformer architectures for sequential recommendation? Provide specific examples of when each would be preferred.

解释基于RNN的序列感知推荐系统如何捕获用户行为中的时间动态。在顺序推荐中使用RNN与Transformer架构的关键区别是什么？提供每种架构偏好的具体例子。

**答案 Answer:**

**How RNNs Capture Temporal Dynamics RNN如何捕获时间动态:**
RNN-based systems process user interaction sequences step by step, maintaining hidden states that encode user preferences at each time step. This allows the model to:
- Track evolving user interests over time 跟踪用户兴趣随时间的演变
- Capture short-term vs long-term preferences 捕获短期vs长期偏好
- Model sequential dependencies in user actions 建模用户行动中的序列依赖

RNN基础系统逐步处理用户交互序列，维护编码每个时间步用户偏好的隐藏状态。这允许模型：

**RNN vs Transformer for Sequential Recommendation:**

| Aspect 方面 | RNN | Transformer |
|-------------|-----|-------------|
| **Processing 处理** | Sequential, step-by-step 顺序，逐步 | Parallel, all positions at once 并行，所有位置同时 |
| **Memory 记忆** | Limited by hidden state size 受隐藏状态大小限制 | Can attend to full history 可以关注完整历史 |
| **Dependencies 依赖** | Biased toward recent items 偏向最近物品 | Flexible attention to any position 对任何位置的灵活注意 |
| **Training Speed 训练速度** | Slower (sequential) 较慢（顺序） | Faster (parallel) 较快（并行） |
| **Long sequences 长序列** | Gradient vanishing issues 梯度消失问题 | Better at handling long sequences 更好处理长序列 |

**When to prefer RNN 何时偏好RNN:**
1. **Real-time recommendation 实时推荐**: When you need to update recommendations immediately after each user action
   当需要在每次用户行动后立即更新推荐时
2. **Memory constraints 内存约束**: When computational resources are limited
   当计算资源有限时
3. **Strong recency bias 强近期偏差**: When recent actions are much more important than distant ones
   当最近行动比遥远行动重要得多时

**When to prefer Transformer 何时偏好Transformer:**
1. **Long user histories 长用户历史**: When users have extensive interaction histories (e.g., years of purchase data)
   当用户有大量交互历史时（例如，数年的购买数据）
2. **Complex temporal patterns 复杂时间模式**: When users have periodic or non-monotonic preferences (e.g., seasonal shopping)
   当用户有周期性或非单调偏好时（例如，季节性购物）
3. **Offline batch training 离线批处理训练**: When training time is more important than inference time
   当训练时间比推理时间更重要时

### 题目10：会话推荐 Session-based Recommendation

**问题 Question:**
Design a session-based recommendation system for an e-commerce website. Address the following challenges:
1. How to handle variable session lengths?
2. How to incorporate item features (price, category, brand)?
3. How to deal with cold-start sessions (new users)?
4. How to balance exploration vs exploitation?

为电商网站设计基于会话的推荐系统。解决以下挑战：
1. 如何处理可变会话长度？
2. 如何融入物品特征（价格、类别、品牌）？
3. 如何处理冷启动会话（新用户）？
4. 如何平衡探索与利用？

**答案 Answer:**

**1. Handling Variable Session Lengths 处理可变会话长度:**

**Approach 方法**: Use padding and masking with attention mechanisms
使用填充和掩码以及注意力机制

```python
# Pseudo-code implementation
class SessionEncoder(nn.Module):
    def __init__(self, max_length=50):
        self.max_length = max_length
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        
    def forward(self, session_items, session_lengths):
        # Pad sequences to max_length
        padded_items = pad_sequence(session_items, max_length)
        
        # Create attention mask
        mask = create_length_mask(session_lengths, max_length)
        
        # Process with RNN
        output, hidden = self.gru(padded_items)
        
        # Apply mask to ignore padding positions
        masked_output = output * mask.unsqueeze(-1)
        
        # Pool over valid positions only
        session_repr = masked_output.sum(1) / session_lengths.unsqueeze(-1)
        return session_repr
```

**Benefits 好处**: Enables batch processing while preserving session semantics
支持批处理同时保持会话语义

**2. Incorporating Item Features 融入物品特征:**

**Multi-modal Embedding Approach 多模态嵌入方法**:

```python
class MultiModalItemEncoder(nn.Module):
    def __init__(self):
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.category_embedding = nn.Embedding(num_categories, embed_dim)
        self.price_encoder = nn.Linear(1, embed_dim)
        self.fusion_layer = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, item_ids, categories, prices):
        item_emb = self.item_embedding(item_ids)
        cat_emb = self.category_embedding(categories)
        price_emb = self.price_encoder(prices.unsqueeze(-1))
        
        # Concatenate and fuse features
        combined = torch.cat([item_emb, cat_emb, price_emb], dim=-1)
        fused_repr = self.fusion_layer(combined)
        return fused_repr
```

**Feature Engineering 特征工程**:
- Price buckets (cheap, medium, expensive) 价格区间（便宜、中等、昂贵）
- Category hierarchy encoding 类别层次编码
- Brand popularity scores 品牌流行度分数

**3. Cold-start Sessions 冷启动会话:**

**Content-based Initialization 基于内容的初始化**:
```python
def handle_cold_start(session_items):
    if len(session_items) < 3:  # Cold start threshold
        # Use content-based similarity
        item_features = get_item_features(session_items)
        similar_items = find_similar_by_content(item_features)
        
        # Blend with popular items
        popular_items = get_trending_items()
        recommendations = blend_recommendations(similar_items, popular_items, 
                                              weights=[0.6, 0.4])
    else:
        # Use full session-based model
        recommendations = session_model.predict(session_items)
    
    return recommendations
```

**Progressive Learning 渐进学习**:
- Start with category-based recommendations 从基于类别的推荐开始
- Gradually shift to personalized patterns as session progresses 随着会话进展逐渐转向个性化模式
- Use uncertainty estimation to know when to explore 使用不确定性估计来知道何时探索

**4. Exploration vs Exploitation Balance 探索与利用平衡:**

**Epsilon-Greedy Strategy 贪婪策略**:
```python
def recommend_with_exploration(session, epsilon=0.1):
    if random.random() < epsilon:
        # Exploration: recommend diverse items
        return diverse_recommendation(session)
    else:
        # Exploitation: use best predicted items
        return model.predict_top_k(session)

def diverse_recommendation(session):
    # Ensure diversity across categories, price ranges, brands
    categories_seen = get_categories(session)
    recommendations = []
    
    for category in unexplored_categories:
        recommendations.extend(get_top_in_category(category, k=2))
    
    return recommendations
```

**Contextual Bandits 上下文强盗**:
- Use Thompson Sampling with session context 使用带会话上下文的汤普森采样
- Maintain uncertainty estimates for each item 为每个物品维护不确定性估计
- Balance based on session stage (more exploration early, exploitation later)
  基于会话阶段平衡（早期更多探索，后期利用）

**Implementation Strategy 实现策略**:
1. **Session stage awareness 会话阶段感知**: Early in session → more exploration, Later → more exploitation
   会话早期 → 更多探索，后期 → 更多利用
2. **Dynamic epsilon 动态贪婪参数**: Start with ε=0.3, decay to ε=0.05
   从ε=0.3开始，衰减到ε=0.05
3. **Multi-objective optimization 多目标优化**: Optimize for both accuracy and diversity
   同时优化准确性和多样性

## 第18.9章：因子分解机 Factorization Machines

### 题目11：FM模型数学原理 FM Mathematical Principles

**问题 Question:**
Derive the computational complexity reduction of Factorization Machines. Show how the naive computation of pairwise interactions O(n²) is reduced to O(nk) where n is the number of features and k is the embedding dimension.

推导因子分解机的计算复杂度降低。展示如何将成对交互的朴素计算O(n²)降低到O(nk)，其中n是特征数量，k是嵌入维度。

**答案 Answer:**

**Factorization Machine Model 因子分解机模型:**
ŷ = w₀ + Σᵢwᵢxᵢ + Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ

Where vᵢ ∈ ℝᵏ is the embedding vector for feature i.
其中vᵢ ∈ ℝᵏ是特征i的嵌入向量。

**Naive Computation 朴素计算 (O(n²)):**
The pairwise interaction term requires computing all pairs:
成对交互项需要计算所有对：

Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ = Σᵢ<ⱼ(Σₓ₌₁ᵏ vᵢ,ₓvⱼ,ₓ)xᵢxⱼ

This requires O(n²k) operations for n features.
这需要对n个特征进行O(n²k)次操作。

**Efficient Computation 高效计算 (O(nk)):**

**Step 1: Expand the sum 展开求和**
Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ = ½[Σᵢ,ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ - Σᵢ⟨vᵢ,vᵢ⟩xᵢ²]

**Step 2: Rewrite using dot product expansion 使用点积展开重写**
Σᵢ,ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ = Σᵢ,ⱼ(Σₓ₌₁ᵏ vᵢ,ₓvⱼ,ₓ)xᵢxⱼ
                = Σₓ₌₁ᵏ Σᵢ,ⱼ vᵢ,ₓvⱼ,ₓxᵢxⱼ
                = Σₓ₌₁ᵏ (Σᵢ vᵢ,ₓxᵢ)(Σⱼ vⱼ,ₓxⱼ)
                = Σₓ₌₁ᵏ (Σᵢ vᵢ,ₓxᵢ)²

**Step 3: Final formula 最终公式**
Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ = ½[Σₓ₌₁ᵏ (Σᵢ vᵢ,ₓxᵢ)² - Σᵢ vᵢ,ₓ²xᵢ²]

**Complexity Analysis 复杂度分析:**
- Computing Σᵢ vᵢ,ₓxᵢ for each dimension: O(n) × k = O(nk)
  为每个维度计算Σᵢ vᵢ,ₓxᵢ：O(n) × k = O(nk)
- Computing squares and final sum: O(k)
  计算平方和最终求和：O(k)
- Total: O(nk)

**Code Implementation 代码实现:**
```python
def efficient_fm_interaction(x, v):
    """
    x: feature vector [n]
    v: embedding matrix [n, k]
    """
    # O(nk) computation
    sum_square = torch.sum((torch.mm(x.unsqueeze(0), v)) ** 2, dim=1)
    square_sum = torch.sum(v ** 2 * (x ** 2).unsqueeze(1), dim=0).sum()
    
    interaction = 0.5 * (sum_square - square_sum)
    return interaction
```

This reduces complexity from O(n²k) to O(nk), enabling FM to scale to high-dimensional sparse features.
这将复杂度从O(n²k)降低到O(nk)，使FM能够扩展到高维稀疏特征。

### 题目12：特征工程 Feature Engineering for FM

**问题 Question:**
You're building a movie recommendation system using Factorization Machines. Design the feature representation for the following scenario:
- User features: age, gender, occupation, location
- Movie features: genre, director, year, rating
- Context features: time of day, day of week, season
- Historical features: user's average rating, movie's popularity

Show how to encode these features and explain the benefits of this representation for FM.

你正在使用因子分解机构建电影推荐系统。为以下场景设计特征表示：
- 用户特征：年龄、性别、职业、地点
- 电影特征：类型、导演、年份、评分
- 上下文特征：一天中的时间、星期几、季节
- 历史特征：用户平均评分、电影流行度

展示如何编码这些特征并解释这种表示对FM的好处。

**答案 Answer:**

**Feature Encoding Strategy 特征编码策略:**

**1. Categorical Features (One-hot encoding) 类别特征（独热编码):**
```python
class MovieFMFeatureEncoder:
    def __init__(self):
        # User features
        self.gender_encoder = {'M': 0, 'F': 1}
        self.occupation_encoder = {'student': 0, 'engineer': 1, 'teacher': 2, ...}
        self.location_encoder = {'NYC': 0, 'LA': 1, 'Chicago': 2, ...}
        
        # Movie features  
        self.genre_encoder = {'Action': 0, 'Comedy': 1, 'Drama': 2, ...}
        self.director_encoder = {'Spielberg': 0, 'Nolan': 1, ...}
        
        # Context features
        self.time_encoder = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
        self.day_encoder = {'Mon': 0, 'Tue': 1, ..., 'Sun': 6}
        self.season_encoder = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3}
    
    def encode_features(self, user_data, movie_data, context_data):
        features = []
        feature_names = []
        
        # User features (one-hot)
        features.extend(self._one_hot(user_data['gender'], self.gender_encoder))
        feature_names.extend(['gender_M', 'gender_F'])
        
        features.extend(self._one_hot(user_data['occupation'], self.occupation_encoder))
        feature_names.extend([f'occupation_{occ}' for occ in self.occupation_encoder.keys()])
        
        # Movie features (one-hot)  
        features.extend(self._one_hot(movie_data['genre'], self.genre_encoder))
        feature_names.extend([f'genre_{genre}' for genre in self.genre_encoder.keys()])
        
        # Numerical features (normalized)
        features.append(self._normalize(user_data['age'], min_age=10, max_age=80))
        feature_names.append('age_normalized')
        
        features.append(self._normalize(movie_data['year'], min_year=1920, max_year=2023))
        feature_names.append('year_normalized')
        
        # Historical features (normalized)
        features.append(self._normalize(user_data['avg_rating'], min_val=1, max_val=5))
        feature_names.append('user_avg_rating')
        
        features.append(self._normalize(movie_data['popularity'], min_val=0, max_val=1))
        feature_names.append('movie_popularity')
        
        return torch.tensor(features), feature_names
```

**2. Complete Feature Vector Example 完整特征向量示例:**

For user: 25-year-old male engineer from NYC watching action movie directed by Nolan on Friday evening:
对于用户：25岁男性工程师来自纽约，周五晚上观看诺兰导演的动作片：

```
Feature Vector (sparse representation):
[
    # User features
    1, 0,           # gender: [M=1, F=0]  
    0, 1, 0, 0,     # occupation: [student=0, engineer=1, teacher=0, other=0]
    1, 0, 0,        # location: [NYC=1, LA=0, Chicago=0]
    0.25,           # age_normalized: (25-10)/(80-10) = 0.25
    
    # Movie features
    1, 0, 0, 0,     # genre: [Action=1, Comedy=0, Drama=0, Sci-fi=0]
    0, 1, 0,        # director: [Spielberg=0, Nolan=1, Cameron=0]
    0.85,           # year_normalized: (2020-1920)/(2023-1920) = 0.97
    
    # Context features
    0, 0, 1, 0,     # time: [morning=0, afternoon=0, evening=1, night=0]
    0, 0, 0, 0, 1, 0, 0,  # day: [Mon=0, ..., Fri=1, Sat=0, Sun=0]
    
    # Historical features
    0.75,           # user_avg_rating: normalized
    0.60,           # movie_popularity: normalized
]
```

**3. Benefits for Factorization Machines FM的好处:**

**Automatic Feature Interaction Learning 自动特征交互学习:**
```python
# FM will automatically learn interactions like:
# - User_Engineer × Genre_Action (engineers like action movies)
# - Time_Evening × Genre_Horror (people watch horror at night)  
# - Age_Young × Director_Nolan (young people like Nolan films)
# - Season_Winter × Genre_Romance (romantic movies in winter)

class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim=10):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(num_features))
        self.v = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.01)
    
    def forward(self, x):
        # Linear terms
        linear = self.w0 + torch.sum(self.w * x)
        
        # Interaction terms (using efficient O(nk) computation)
        interactions = 0.5 * torch.sum(
            torch.pow(torch.mm(x.unsqueeze(0), self.v), 2) - 
            torch.mm(torch.pow(x, 2).unsqueeze(0), torch.pow(self.v, 2)),
            dim=1
        )
        
        return linear + interactions
```

**Key Benefits 主要好处:**

1. **Sparse Feature Handling 稀疏特征处理**: FM works well with sparse one-hot vectors
   FM适用于稀疏的独热向量

2. **Automatic Interaction Discovery 自动交互发现**: No need to manually engineer feature crosses
   无需手动工程特征交叉

3. **Cold Start Robustness 冷启动鲁棒性**: Can make predictions even with unseen feature combinations
   即使遇到未见过的特征组合也能进行预测

4. **Interpretability 可解释性**: Embedding vectors reveal feature relationships
   嵌入向量揭示特征关系

5. **Scalability 可扩展性**: O(nk) complexity enables high-dimensional features
   O(nk)复杂度支持高维特征

This representation allows FM to capture complex patterns like "young engineers in NYC prefer action movies on Friday evenings" without explicitly programming these rules.

这种表示允许FM捕获复杂模式，如"纽约的年轻工程师喜欢在周五晚上看动作片"，而无需明确编程这些规则。 

## 第18.4章：自编码器评分预测 AutoRec: Rating Prediction with Autoencoders

### 题目13：AutoRec模型概念 AutoRec Model Concepts

**问题 Question:**
Explain the core idea behind AutoRec for collaborative filtering. What is the difference between I-AutoRec and U-AutoRec, and why is the masked loss function necessary for training AutoRec?

解释AutoRec用于协同过滤的核心思想。I-AutoRec和U-AutoRec有什么区别，为什么训练AutoRec需要掩码损失函数？

**答案 Answer:**

**Core Idea 核心思想:**
AutoRec (Auto-encoder for Collaborative Filtering) leverages autoencoders to learn latent representations of users or items from their rating vectors, and then uses these representations to reconstruct and predict missing ratings. The model aims to learn an identity function for observed ratings while predicting unobserved ones.

AutoRec（协同过滤自编码器）利用自编码器从用户或物品的评分向量中学习潜在表示，然后使用这些表示来重建和预测缺失的评分。模型的目的是学习一个用于已观察评分的恒等函数，同时预测未观察到的评分。

**I-AutoRec vs U-AutoRec I-AutoRec vs U-AutoRec:**
- **I-AutoRec (Item-based AutoRec)**: Takes item rating vectors (all users' ratings for one item) as input. It learns a low-dimensional representation for each item, which can then be used to predict ratings for that item across users.
  **I-AutoRec（基于物品的AutoRec）**: 以物品评分向量（所有用户对一个物品的评分）作为输入。它学习每个物品的低维表示，然后可用于预测该物品在所有用户上的评分。
- **U-AutoRec (User-based AutoRec)**: Takes user rating vectors (one user's ratings for all items) as input. It learns a low-dimensional representation for each user, used to predict ratings for that user across items.
  **U-AutoRec（基于用户的AutoRec）**: 以用户评分向量（一个用户对所有物品的评分）作为输入。它学习每个用户的低维表示，用于预测该用户在所有物品上的评分。

**Necessity of Masked Loss Function 掩码损失函数的必要性:**
The masked loss function is crucial because the input rating matrix is sparse (most ratings are missing). The autoencoder should only be penalized for errors on observed ratings. If the loss were calculated on all elements, the model would be forced to predict 0 (or the default missing value) for unobserved ratings, which is not the goal. The mask ensures that the model learns to reconstruct only the known ratings accurately.

掩码损失函数至关重要，因为输入评分矩阵是稀疏的（大多数评分缺失）。自编码器只应因已观察评分上的误差而受到惩罚。如果损失在所有元素上计算，模型将被迫对未观察的评分预测为0（或默认缺失值），这不是目标。掩码确保模型学习只准确地重建已知评分。

### 题目14：AutoRec的优缺点 Advantages and Disadvantages of AutoRec

**问题 Question:**
Discuss the main advantages of AutoRec compared to traditional Matrix Factorization. What are its limitations?

讨论AutoRec与传统矩阵分解相比的主要优势。它有哪些局限性？

**答案 Answer:**

**Advantages of AutoRec 优势:**
1.  **Non-linear Modeling 非线性建模**: AutoRec, being a neural network, can capture complex, non-linear relationships in user-item interactions, unlike traditional matrix factorization which primarily models linear relationships.
    非线性建模：AutoRec作为神经网络，能够捕获用户-物品交互中复杂的非线性关系，这与主要建模线性关系的传统矩阵分解不同。
2.  **End-to-end Learning 端到端学习**: It learns representations and prediction directly from the raw rating data, potentially discovering more nuanced patterns.
    端到端学习：它直接从原始评分数据中学习表示和预测，可能发现更细微的模式。
3.  **Flexibility 灵活性**: The autoencoder architecture can be extended with more layers or different activation functions to increase model capacity.
    灵活性：自编码器架构可以通过更多层或不同的激活函数进行扩展，以增加模型容量。

**Limitations of AutoRec 局限性:**
1.  **Cold Start Problem 冷启动问题**: Similar to matrix factorization, AutoRec struggles with new users or items that have few or no ratings, as it cannot effectively learn their representations.
    冷启动问题：与矩阵分解类似，AutoRec难以处理评分很少或没有评分的新用户或物品，因为它无法有效学习它们的表示。
2.  **Interpretability 可解释性**: The learned latent factors in deep autoencoders are often less interpretable than the explicit latent factors in traditional matrix factorization.
    可解释性：深度自编码器中学习到的潜在因子通常比传统矩阵分解中的显式潜在因子更难解释。
3.  **Training Complexity 训练复杂性**: Training deep neural networks can be computationally expensive and requires careful tuning of hyperparameters (e.g., hidden layer size, learning rate, regularization).
    训练复杂性：训练深度神经网络可能计算成本高昂，并且需要仔细调整超参数（例如，隐藏层大小、学习率、正则化）。

## 第18.5章：个性化排序推荐系统 Personalized Ranking for Recommender Systems

### 题目15：BPR损失函数 Bayesian Personalized Ranking Loss Function

**问题 Question:**
Explain the intuition behind Bayesian Personalized Ranking (BPR) loss for personalized ranking. Why is negative sampling crucial for training models with BPR loss, and how does it differ from traditional rating prediction losses (like MSE)?

解释贝叶斯个性化排序（BPR）损失用于个性化排序的直觉。为什么负采样对于使用BPR损失训练模型至关重要，它与传统评分预测损失（如MSE）有何不同？

**答案 Answer:**

**Intuition behind BPR Loss BPR损失的直觉:**
BPR loss is designed for implicit feedback scenarios where we only observe positive interactions (e.g., a user clicked on an item, watched a movie). The core intuition is to optimize for pairwise preferences: for a given user, a positively interacted item (positive sample) should be ranked higher than an unobserved item (negative sample). Instead of predicting an absolute rating, BPR focuses on the relative order. The loss encourages the model to assign a higher score to the positive item than to the negative item.

BPR损失是为隐式反馈场景设计的，我们只观察到正向交互（例如，用户点击了一个物品，观看了一部电影）。核心直觉是优化成对偏好：对于给定用户，一个正向交互的物品（正样本）应该比一个未观察到的物品（负样本）排名更高。BPR不预测绝对评分，而是关注相对顺序。损失鼓励模型为正向物品分配比负向物品更高的分数。

**Why Negative Sampling is Crucial 负采样至关重要:**
In implicit feedback, we only know what a user liked. We don't have explicit "dislikes." Negative sampling is crucial to generate "unobserved" items that the user likely did *not* prefer. Without negative samples, the model would only learn from positive interactions, which doesn't provide enough information to distinguish between preferred and non-preferred items. It allows the model to learn a contrastive ranking by comparing a positive item with randomly sampled negative items.

在隐式反馈中，我们只知道用户喜欢什么。我们没有明确的“不喜欢”。负采样对于生成用户可能不喜欢的“未观察”物品至关重要。如果没有负样本，模型将只从正向交互中学习，这不足以区分偏好和非偏好物品。它允许模型通过将正向物品与随机采样的负向物品进行比较来学习对比排序。

**Difference from Traditional Rating Prediction Losses (like MSE) 与传统评分预测损失（如MSE）的区别:**
-   **Objective 目标**: MSE aims to minimize the squared difference between predicted and actual *ratings*. BPR aims to maximize the probability that a *positive item is ranked higher than a negative item*.
    **目标**: MSE旨在最小化预测评分与实际评分之间的平方差。BPR旨在最大化正向物品排名高于负向物品的概率。
-   **Input Data 输入数据**: MSE通常需要明确的数值评分进行训练。BPR处理隐式反馈，使用（用户、正向物品、负向物品）三元组。
    **输入数据**: MSE通常需要明确的数值评分进行训练。BPR处理隐式反馈，使用（用户、正向物品、负向物品）三元组。
-   **Focus 侧重点**: MSE focuses on precise rating prediction. BPR focuses on relative ranking and ordering.
    **侧重点**: MSE侧重于精确的评分预测。BPR侧重于相对排名和排序。

### 题目16：Hinge Loss for Ranking 排序的Hinge损失

**问题 Question:**
Describe the Hinge Loss function in the context of personalized ranking. How does it enforce a margin, and what are its advantages and disadvantages compared to BPR loss?

描述个性化排序中Hinge损失函数。它如何强制执行边际，与BPR损失相比，它有哪些优缺点？

**答案 Answer:**

**Description of Hinge Loss 描述Hinge损失:**
Hinge Loss, borrowed from Support Vector Machines (SVMs), is a margin-based loss function for ranking. For a given user \(u\), a positive item \(i\), and a negative item \(j\), the Hinge Loss ensures that the score of the positive item \( \hat{x}_{ui} \) is at least a margin \( \delta \) greater than the score of the negative item \( \hat{x}_{uj} \). If this condition is not met, a penalty is incurred.

Hinge损失借鉴自支持向量机（SVM），是一种基于边际的排序损失函数。对于给定用户\(u\)，正向物品\(i\)和负向物品\(j\)，Hinge损失确保正向物品的分数\( \hat{x}_{ui} \)至少比负向物品的分数\( \hat{x}_{uj} \)大一个边际\( \delta \)。如果未满足此条件，则会产生惩罚。

**Enforcing a Margin 强制执行边际:**
The mathematical formulation is \( L_{hinge} = \sum_{(u,i,j)} \max(0, \delta - (\hat{x}_{ui} - \hat{x}_{uj})) \).
Here, \( \delta \) is the margin parameter (commonly 1.0). The term \( (\hat{x}_{ui} - \hat{x}_{uj}) \) represents the difference in predicted scores between the positive and negative items. If this difference is greater than or equal to \( \delta \), the loss for that triplet is 0. If it's less than \( \delta \), the model incurs a penalty proportional to how far it is from meeting the margin requirement. This explicitly pushes the positive item's score to be higher than the negative item's score by at least \( \delta \).

数学公式为\( L_{hinge} = \sum_{(u,i,j)} \max(0, \delta - (\hat{x}_{ui} - \hat{x}_{uj})) \)。
其中，\( \delta \)是边际参数（通常为1.0）。\( (\hat{x}_{ui} - \hat{x}_{uj}) \)表示正负物品预测分数之间的差异。如果此差异大于或等于\( \delta \)，则该三元组的损失为0。如果小于\( \delta \)，模型将根据其与满足边际要求的距离成比例地受到惩罚。这明确地推动正向物品的分数比负向物品的分数至少高\( \delta \)。

**Advantages compared to BPR Loss 与BPR损失相比的优势:**
1.  **Robustness to Outliers 对异常值的鲁棒性**: Hinge loss is less sensitive to extreme score differences because once the margin is met, the loss is zero. BPR, based on the sigmoid function, always has a non-zero gradient, meaning it continues to optimize even for well-ranked pairs, which can sometimes be less efficient or more prone to overfitting.
    对异常值的鲁棒性：Hinge损失对极端分数差异的敏感度较低，因为一旦满足边际，损失就为零。BPR基于sigmoid函数，始终具有非零梯度，这意味着它甚至对于排名良好的对也会继续优化，这有时效率较低或更容易过拟合。
2.  **Faster Convergence 更快的收敛**: In some cases, Hinge loss can lead to faster convergence due to its "hard margin" approach, as it focuses training efforts on triplets that violate the margin.
    更快的收敛：在某些情况下，Hinge损失由于其“硬边际”方法可以导致更快的收敛，因为它将训练精力集中在违反边际的三元组上。

**Disadvantages compared to BPR Loss 与BPR损失相比的劣势:**
1.  **Less Probabilistic Interpretation 概率解释性较差**: BPR has a clear probabilistic interpretation (maximizing the posterior probability of correct rankings). Hinge loss is purely a margin-based optimization without a direct probabilistic meaning.
    概率解释性较差：BPR具有清晰的概率解释（最大化正确排名的后验概率）。Hinge损失纯粹是基于边际的优化，没有直接的概率意义。
2.  **Margin Tuning 需要调整边际参数**: The margin parameter \( \delta \) needs to be carefully tuned, which can be an additional hyperparameter to optimize.
    需要调整边际参数：边际参数\( \delta \)需要仔细调整，这可能是一个额外的需要优化的超参数。 