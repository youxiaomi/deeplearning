# Chapter 18: Recommender Systems 推荐系统

## 18.1 Overview of Recommender Systems 推荐系统概述

### 18.1.1 Collaborative Filtering 协同过滤

**Collaborative Filtering** is a fundamental approach in recommender systems that predicts user preferences based on the preferences of similar users or items. Think of it as asking your friends for movie recommendations - if you and your friend have similar tastes and your friend liked a movie, you're likely to enjoy it too.

**协同过滤**是推荐系统中的基础方法，通过相似用户或物品的偏好来预测用户的喜好。可以想象成向朋友寻求电影推荐 - 如果你和朋友有相似的品味，朋友喜欢的电影你也很可能会喜欢。

#### User-based Collaborative Filtering 基于用户的协同过滤

This method finds users with similar preferences and recommends items that similar users have liked. For example, if User A and User B both rated movies "Inception" and "The Matrix" highly, and User A also loved "Interstellar", then the system would recommend "Interstellar" to User B.

这种方法找到具有相似偏好的用户，推荐相似用户喜欢的物品。例如，如果用户A和用户B都对电影《盗梦空间》和《黑客帝国》给出高分，而用户A也很喜欢《星际穿越》，那么系统会向用户B推荐《星际穿越》。

```python
# 基于用户的协同过滤示例
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 用户-物品评分矩阵
# 行：用户，列：物品
user_item_matrix = np.array([
    [5, 3, 0, 1],  # 用户1的评分
    [4, 0, 0, 1],  # 用户2的评分
    [1, 1, 0, 5],  # 用户3的评分
    [1, 0, 0, 4],  # 用户4的评分
])

# 计算用户之间的相似度
user_similarity = cosine_similarity(user_item_matrix)
print("用户相似度矩阵：")
print(user_similarity)
```

#### Item-based Collaborative Filtering 基于物品的协同过滤

This approach recommends items similar to those the user has previously liked. If a user enjoyed "The Lord of the Rings", the system might recommend "Harry Potter" because both are fantasy movies with similar characteristics.

这种方法推荐与用户之前喜欢的物品相似的物品。如果用户喜欢《指环王》，系统可能会推荐《哈利·波特》，因为两者都是具有相似特征的奇幻电影。

### 18.1.2 Explicit Feedback and Implicit Feedback 显式反馈和隐式反馈

#### Explicit Feedback 显式反馈

**Explicit feedback** refers to direct user ratings or reviews. This is like when you rate a movie 1-5 stars on a streaming platform or write a review on Amazon. The user explicitly tells the system their preference.

**显式反馈**指用户直接给出的评分或评论。就像你在流媒体平台上给电影打1-5星评分，或在亚马逊上写评论一样。用户明确告诉系统他们的偏好。

Examples 示例:
- Star ratings (1-5 stars) 星级评分（1-5星）
- Thumbs up/down 点赞/点踩
- Written reviews 书面评论
- Purchase decisions 购买决策

#### Implicit Feedback 隐式反馈

**Implicit feedback** is derived from user behavior rather than explicit ratings. This is like observing that you watched a movie to completion (positive signal) or stopped watching after 10 minutes (negative signal).

**隐式反馈**来源于用户行为而非明确评分。就像观察你是否完整观看了一部电影（正向信号）或观看10分钟后就停止了（负向信号）。

Examples 示例:
- Click-through rates 点击率
- Time spent on page 页面停留时间
- Purchase history 购买历史
- Browsing patterns 浏览模式
- Download counts 下载次数

```python
# 隐式反馈示例：从用户行为推断偏好
class ImplicitFeedback:
    def __init__(self):
        self.user_behavior = {}
    
    def record_view(self, user_id, item_id, duration):
        """记录用户观看行为"""
        if user_id not in self.user_behavior:
            self.user_behavior[user_id] = []
        
        # 根据观看时长推断偏好
        if duration > 0.8:  # 观看80%以上认为喜欢
            preference = 1
        elif duration > 0.3:  # 观看30%-80%认为中等喜欢
            preference = 0.5
        else:  # 观看少于30%认为不喜欢
            preference = 0
            
        self.user_behavior[user_id].append({
            'item_id': item_id,
            'duration': duration,
            'preference': preference
        })
```

### 18.1.3 Recommendation Tasks 推荐任务

#### Rating Prediction 评分预测

**Rating prediction** aims to predict the exact rating a user would give to an item. It's like trying to guess whether you would rate a movie 3, 4, or 5 stars based on your past preferences.

**评分预测**旨在预测用户对物品的确切评分。就像根据你过去的偏好来猜测你会给一部电影打3星、4星还是5星。

Mathematical formulation 数学表述:
```
r̂ui = f(u, i, Θ)
```
Where 其中:
- r̂ui: predicted rating for user u on item i 用户u对物品i的预测评分
- f: prediction function 预测函数
- Θ: model parameters 模型参数

#### Ranking Prediction 排序预测

**Ranking prediction** focuses on ordering items by preference rather than predicting exact ratings. It's like creating a "top 10 movies you might like" list, where the order matters more than specific scores.

**排序预测**专注于按偏好对物品排序，而不是预测确切评分。就像创建"你可能喜欢的前10部电影"列表，顺序比具体分数更重要。

#### Top-N Recommendation 前N推荐

**Top-N recommendation** provides a ranked list of N items that the user is most likely to prefer. Netflix's "Top Picks for You" is a perfect example - they show you the 10-20 movies/shows they think you'll enjoy most.

**前N推荐**提供用户最可能喜欢的N个物品的排序列表。Netflix的"为您推荐"就是完美的例子 - 他们向你展示认为你最喜欢的10-20部电影/节目。

### 18.1.4 Summary 总结

Recommender systems are essential tools in today's digital world, helping users discover relevant content from vast catalogs. The choice between collaborative filtering approaches, feedback types, and recommendation tasks depends on the specific application and available data.

推荐系统是当今数字世界的重要工具，帮助用户从庞大的目录中发现相关内容。协同过滤方法、反馈类型和推荐任务的选择取决于具体应用和可用数据。

### 18.1.5 Exercises 练习

1. **Compare and contrast** explicit vs implicit feedback in the context of an online bookstore.
   **比较和对比**在线书店中显式反馈与隐式反馈。

2. **Design** a user-based collaborative filtering system for a music streaming service.
   **设计**一个音乐流媒体服务的基于用户的协同过滤系统。

3. **Implement** a simple item-based recommendation using cosine similarity.
   **实现**一个使用余弦相似度的简单基于物品的推荐。

## 18.2 The MovieLens Dataset MovieLens数据集

### 18.2.1 Getting the Data 获取数据

The **MovieLens dataset** is one of the most popular benchmark datasets for recommender systems research. Created by the GroupLens research lab at the University of Minnesota, it contains millions of movie ratings from thousands of users.

**MovieLens数据集**是推荐系统研究中最受欢迎的基准数据集之一。由明尼苏达大学GroupLens研究实验室创建，包含来自数千用户的数百万电影评分。

Think of MovieLens as a digital record of people's movie preferences - like having access to millions of movie review cards where people rated movies from 1 to 5 stars, along with information about when they watched the movies and what genres they prefer.

可以把MovieLens想象成人们电影偏好的数字记录 - 就像拥有数百万张电影评论卡的访问权限，人们在卡片上给电影打1到5星评分，还有他们观看电影的时间和喜欢的类型信息。

```python
# 下载和加载MovieLens数据集
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile
import os

def download_movielens_data(dataset_size='100k'):
    """
    下载MovieLens数据集
    dataset_size: '100k', '1m', '10m', '20m'
    """
    base_url = "http://files.grouplens.org/datasets/movielens/"
    
    if dataset_size == '100k':
        url = base_url + "ml-100k.zip"
        filename = "ml-100k.zip"
    elif dataset_size == '1m':
        url = base_url + "ml-1m.zip"
        filename = "ml-1m.zip"
    
    if not os.path.exists(filename):
        print(f"正在下载MovieLens {dataset_size}数据集...")
        urlretrieve(url, filename)
        
        # 解压文件
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall()
        print("下载完成！")
    else:
        print("数据集已存在")
```

### 18.2.2 Statistics of the Dataset 数据集统计

The MovieLens dataset comes in different sizes, each suitable for different research purposes and computational constraints:

MovieLens数据集有不同规模，每种都适合不同的研究目的和计算约束：

#### MovieLens 100K Dataset MovieLens 100K数据集
- **100,000 ratings** from 943 users on 1,682 movies
- **100,000个评分**，来自943个用户对1,682部电影的评分
- Rating scale: 1-5 stars 评分范围：1-5星
- Time period: 1990s 时间段：1990年代
- Perfect for learning and small experiments 适合学习和小型实验

#### MovieLens 1M Dataset MovieLens 1M数据集
- **1 million ratings** from 6,000 users on 4,000 movies
- **100万个评分**，来自6,000个用户对4,000部电影的评分
- More diverse and realistic for research 更多样化，更适合研究

```python
def analyze_dataset_statistics(ratings_df):
    """分析数据集统计信息"""
    print("=== MovieLens数据集统计 ===")
    print(f"总评分数: {len(ratings_df):,}")
    print(f"用户数: {ratings_df['user_id'].nunique():,}")
    print(f"电影数: {ratings_df['movie_id'].nunique():,}")
    print(f"评分密度: {len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['movie_id'].nunique()) * 100:.2f}%")
    
    print("\n=== 评分分布 ===")
    rating_counts = ratings_df['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        percentage = count / len(ratings_df) * 100
        print(f"{rating}星: {count:,} 个评分 ({percentage:.1f}%)")
    
    print(f"\n=== 用户活跃度 ===")
    user_ratings = ratings_df['user_id'].value_counts()
    print(f"最活跃用户评分数: {user_ratings.max()}")
    print(f"平均每用户评分数: {user_ratings.mean():.1f}")
    print(f"中位数每用户评分数: {user_ratings.median():.1f}")
    
    print(f"\n=== 电影热度 ===")
    movie_ratings = ratings_df['movie_id'].value_counts()
    print(f"最热门电影评分数: {movie_ratings.max()}")
    print(f"平均每电影评分数: {movie_ratings.mean():.1f}")
    print(f"中位数每电影评分数: {movie_ratings.median():.1f}")
```

### 18.2.3 Splitting the Dataset 数据集分割

**Dataset splitting** is crucial for evaluating recommender systems. We need to simulate real-world scenarios where we predict future preferences based on past behavior. Think of it like using your movie watching history from January to November to predict what movies you'll enjoy in December.

**数据集分割**对评估推荐系统至关重要。我们需要模拟真实场景，基于过去的行为预测未来的偏好。可以想象成使用你1月到11月的电影观看历史来预测你12月会喜欢什么电影。

#### Temporal Splitting 时间分割

This method splits data based on timestamps, using earlier interactions for training and later ones for testing. This mimics real-world deployment where we predict future preferences.

这种方法基于时间戳分割数据，使用较早的交互进行训练，较晚的进行测试。这模拟了预测未来偏好的真实部署场景。

```python
def temporal_split(ratings_df, test_ratio=0.2):
    """
    基于时间的数据分割
    """
    # 按时间戳排序
    ratings_df = ratings_df.sort_values('timestamp')
    
    # 计算分割点
    split_point = int(len(ratings_df) * (1 - test_ratio))
    
    train_data = ratings_df.iloc[:split_point]
    test_data = ratings_df.iloc[split_point:]
    
    print(f"训练集大小: {len(train_data):,} 评分")
    print(f"测试集大小: {len(test_data):,} 评分")
    print(f"训练集时间范围: {train_data['timestamp'].min()} - {train_data['timestamp'].max()}")
    print(f"测试集时间范围: {test_data['timestamp'].min()} - {test_data['timestamp'].max()}")
    
    return train_data, test_data
```

#### Random Splitting 随机分割

Random splitting randomly assigns ratings to training and test sets. While this doesn't reflect temporal patterns, it's useful for general model evaluation.

随机分割将评分随机分配到训练集和测试集。虽然这不反映时间模式，但对一般模型评估很有用。

```python
from sklearn.model_selection import train_test_split

def random_split(ratings_df, test_ratio=0.2, random_state=42):
    """
    随机数据分割
    """
    train_data, test_data = train_test_split(
        ratings_df, 
        test_size=test_ratio, 
        random_state=random_state
    )
    
    print(f"训练集大小: {len(train_data):,} 评分")
    print(f"测试集大小: {len(test_data):,} 评分")
    
    return train_data, test_data
```

#### Leave-One-Out Splitting 留一法分割

For each user, we hold out one rating for testing and use the rest for training. This is particularly useful for ranking evaluation.

对每个用户，我们保留一个评分用于测试，其余用于训练。这对排序评估特别有用。

```python
def leave_one_out_split(ratings_df):
    """
    留一法数据分割 - 每个用户留出最后一个评分作为测试
    """
    # 按用户和时间戳排序
    ratings_df = ratings_df.sort_values(['user_id', 'timestamp'])
    
    # 获取每个用户的最后一个评分作为测试集
    test_data = ratings_df.groupby('user_id').tail(1)
    
    # 剩余的作为训练集
    train_data = ratings_df.drop(test_data.index)
    
    print(f"训练集大小: {len(train_data):,} 评分")
    print(f"测试集大小: {len(test_data):,} 评分")
    print(f"每个用户在测试集中有 1 个评分")
    
    return train_data, test_data
```

### 18.2.4 Loading the Data 加载数据

Let's implement a comprehensive data loader for the MovieLens dataset that handles different formats and provides easy access to the data:

让我们实现一个全面的MovieLens数据集加载器，处理不同格式并提供对数据的便捷访问：

```python
import pandas as pd
import numpy as np
from datetime import datetime

class MovieLensDataLoader:
    """MovieLens数据加载器"""
    
    def __init__(self, dataset_path, dataset_size='100k'):
        self.dataset_path = dataset_path
        self.dataset_size = dataset_size
        self.ratings = None
        self.movies = None
        self.users = None
        
    def load_ratings(self):
        """加载评分数据"""
        if self.dataset_size == '100k':
            # MovieLens 100K格式：user_id \t item_id \t rating \t timestamp
            self.ratings = pd.read_csv(
                f"{self.dataset_path}/u.data",
                sep='\t',
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                engine='python'
            )
        elif self.dataset_size == '1m':
            # MovieLens 1M格式：user_id::movie_id::rating::timestamp
            self.ratings = pd.read_csv(
                f"{self.dataset_path}/ratings.dat",
                sep='::',
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                engine='python'
            )
        
        # 转换时间戳
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], unit='s')
        
        print(f"成功加载 {len(self.ratings):,} 个评分")
        return self.ratings
    
    def load_movies(self):
        """加载电影数据"""
        if self.dataset_size == '100k':
            # 加载电影信息
            self.movies = pd.read_csv(
                f"{self.dataset_path}/u.item",
                sep='|',
                encoding='latin-1',
                names=['movie_id', 'title', 'release_date', 'video_release_date', 'url'] + 
                      [f'genre_{i}' for i in range(19)],
                engine='python'
            )
        elif self.dataset_size == '1m':
            self.movies = pd.read_csv(
                f"{self.dataset_path}/movies.dat",
                sep='::',
                names=['movie_id', 'title', 'genres'],
                engine='python',
                encoding='latin-1'
            )
        
        print(f"成功加载 {len(self.movies):,} 部电影")
        return self.movies
    
    def load_users(self):
        """加载用户数据"""
        if self.dataset_size == '100k':
            self.users = pd.read_csv(
                f"{self.dataset_path}/u.user",
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )
        elif self.dataset_size == '1m':
            self.users = pd.read_csv(
                f"{self.dataset_path}/users.dat",
                sep='::',
                names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                engine='python'
            )
        
        print(f"成功加载 {len(self.users):,} 个用户")
        return self.users
    
    def load_all(self):
        """加载所有数据"""
        self.load_ratings()
        self.load_movies()
        self.load_users()
        
        print("\n=== 数据加载完成 ===")
        print(f"评分数: {len(self.ratings):,}")
        print(f"用户数: {len(self.users):,}")
        print(f"电影数: {len(self.movies):,}")
        
        return self.ratings, self.movies, self.users
```

### 18.2.5 Summary 总结

The MovieLens dataset provides a realistic foundation for experimenting with recommender systems. Its different sizes accommodate various computational resources, while its rich metadata enables sophisticated recommendation approaches.

MovieLens数据集为推荐系统实验提供了现实的基础。其不同规模适应各种计算资源，而丰富的元数据支持复杂的推荐方法。

### 18.2.6 Exercises 练习

1. **Load** the MovieLens 100K dataset and compute basic statistics.
   **加载**MovieLens 100K数据集并计算基本统计信息。

2. **Implement** different splitting strategies and compare their characteristics.
   **实现**不同的分割策略并比较其特征。

3. **Analyze** the sparsity problem in the user-item rating matrix.
   **分析**用户-物品评分矩阵中的稀疏性问题。

## 18.3. Matrix Factorization 矩阵分解

### 18.3.1. The Matrix Factorization Model 矩阵分解模型

**Matrix Factorization** is one of the most successful approaches in collaborative filtering. Imagine you have a huge spreadsheet where rows represent users, columns represent movies, and cells contain ratings. Most cells are empty (sparse matrix). Matrix factorization discovers hidden patterns by decomposing this sparse matrix into two smaller dense matrices.

**矩阵分解**是协同过滤中最成功的方法之一。想象你有一个巨大的电子表格，行代表用户，列代表电影，单元格包含评分。大多数单元格都是空的（稀疏矩阵）。矩阵分解通过将这个稀疏矩阵分解为两个较小的密集矩阵来发现隐藏模式。

The core idea is that user preferences and item characteristics can be represented in a lower-dimensional latent space. Think of it like reducing complex movie preferences to basic dimensions like "action vs romance" and "serious vs comedy".

核心思想是用户偏好和物品特征可以在低维潜在空间中表示。可以想象成将复杂的电影偏好简化为基本维度，如"动作vs浪漫"和"严肃vs喜剧"。

#### Mathematical Foundation 数学基础

The rating matrix R ∈ R^(m×n) is approximated by:
评分矩阵 R ∈ R^(m×n) 近似为：

```
R ≈ UV^T
```

Where 其中:
- U ∈ R^(m×k): user latent factor matrix 用户潜在因子矩阵
- V ∈ R^(n×k): item latent factor matrix 物品潜在因子矩阵  
- k: number of latent factors (much smaller than m and n) 潜在因子数量（远小于m和n）

The predicted rating for user u on item i is:
用户u对物品i的预测评分为：

```
r̂_ui = u_u^T v_i = Σ(j=1 to k) u_uj * v_ij
```

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class MatrixFactorization:
    """矩阵分解推荐系统"""
    
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.01, n_epochs=100):
        self.n_factors = n_factors      # 潜在因子数量
        self.learning_rate = learning_rate    # 学习率
        self.regularization = regularization  # 正则化参数
        self.n_epochs = n_epochs        # 训练轮数
        
    def fit(self, ratings_matrix):
        """
        训练矩阵分解模型
        ratings_matrix: 用户-物品评分矩阵，缺失值用0表示
        """
        self.n_users, self.n_items = ratings_matrix.shape
        
        # 初始化用户和物品潜在因子矩阵
        # 使用小的随机值初始化
        self.user_factors = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        
        # 用户和物品偏置项
        self.user_bias = np.zeros(self.n_users)
        self.item_bias = np.zeros(self.n_items)
        self.global_bias = np.mean(ratings_matrix[ratings_matrix > 0])
        
        # 找到非零评分的位置
        self.train_indices = np.where(ratings_matrix > 0)
        self.train_ratings = ratings_matrix[self.train_indices]
        
        # 训练历史
        self.train_losses = []
        
        print(f"开始训练矩阵分解模型...")
        print(f"用户数: {self.n_users}, 物品数: {self.n_items}")
        print(f"潜在因子数: {self.n_factors}, 训练评分数: {len(self.train_ratings)}")
        
        for epoch in range(self.n_epochs):
            self._train_epoch(ratings_matrix)
            
            if epoch % 10 == 0:
                loss = self._compute_loss(ratings_matrix)
                self.train_losses.append(loss)
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def _train_epoch(self, ratings_matrix):
        """训练一个epoch"""
        # 随机打乱训练顺序
        indices = np.random.permutation(len(self.train_ratings))
        
        for idx in indices:
            u = self.train_indices[0][idx]  # 用户索引
            i = self.train_indices[1][idx]  # 物品索引
            rating = self.train_ratings[idx]  # 真实评分
            
            # 计算预测评分
            prediction = self._predict_single(u, i)
            error = rating - prediction
            
            # 保存当前参数用于更新
            user_factor = self.user_factors[u].copy()
            item_factor = self.item_factors[i].copy()
            
            # 梯度下降更新
            # 更新潜在因子
            self.user_factors[u] += self.learning_rate * (
                error * item_factor - self.regularization * user_factor
            )
            self.item_factors[i] += self.learning_rate * (
                error * user_factor - self.regularization * item_factor
            )
            
            # 更新偏置项
            self.user_bias[u] += self.learning_rate * (
                error - self.regularization * self.user_bias[u]
            )
            self.item_bias[i] += self.learning_rate * (
                error - self.regularization * self.item_bias[i]
            )
    
    def _predict_single(self, user_idx, item_idx):
        """预测单个用户对单个物品的评分"""
        prediction = self.global_bias + self.user_bias[user_idx] + self.item_bias[item_idx]
        prediction += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return prediction
    
    def _compute_loss(self, ratings_matrix):
        """计算训练损失"""
        predictions = self.predict_all()
        
        # 只计算已知评分的损失
        mse = mean_squared_error(
            self.train_ratings, 
            predictions[self.train_indices]
        )
        
        # 添加正则化项
        regularization_loss = self.regularization * (
            np.sum(self.user_factors ** 2) + 
            np.sum(self.item_factors ** 2) +
            np.sum(self.user_bias ** 2) + 
            np.sum(self.item_bias ** 2)
        )
        
        return mse + regularization_loss
    
    def predict_all(self):
        """预测所有用户对所有物品的评分"""
        predictions = np.zeros((self.n_users, self.n_items))
        
        for u in range(self.n_users):
            for i in range(self.n_items):
                predictions[u, i] = self._predict_single(u, i)
        
        return predictions
    
    def predict(self, user_idx, item_idx):
        """预测特定用户对特定物品的评分"""
        if hasattr(user_idx, '__iter__'):  # 如果是数组
            return [self._predict_single(u, i) for u, i in zip(user_idx, item_idx)]
        else:  # 如果是单个值
            return self._predict_single(user_idx, item_idx)
    
    def recommend(self, user_idx, num_recommendations=10, exclude_rated=True):
        """为用户推荐物品"""
        user_predictions = []
        
        for item_idx in range(self.n_items):
            # 如果设置排除已评分物品且用户已评分该物品，跳过
            if exclude_rated and hasattr(self, 'train_indices'):
                user_rated_items = self.train_indices[1][self.train_indices[0] == user_idx]
                if item_idx in user_rated_items:
                    continue
            
            prediction = self._predict_single(user_idx, item_idx)
            user_predictions.append((item_idx, prediction))
        
        # 按预测评分排序
        user_predictions.sort(key=lambda x: x[1], reverse=True)
        
        return user_predictions[:num_recommendations]
    
    def plot_training_loss(self):
        """绘制训练损失曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(0, len(self.train_losses) * 10, 10), self.train_losses)
        plt.title('Matrix Factorization Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
```

### 18.3.2. Model Implementation 模型实现

Let's implement a more advanced matrix factorization model with bias terms and regularization. Bias terms capture user and item specific tendencies - some users tend to rate everything highly, while some movies are generally well-received.

让我们实现一个更高级的矩阵分解模型，包含偏置项和正则化。偏置项捕获用户和物品的特定倾向 - 一些用户倾向于给所有东西高分，而一些电影普遍受好评。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class RatingDataset(Dataset):
    """评分数据集类"""
    
    def __init__(self, user_ids, item_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)  
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]

class MatrixFactorizationNet(nn.Module):
    """PyTorch矩阵分解网络"""
    
    def __init__(self, n_users, n_items, n_factors=20, dropout=0.1):
        super(MatrixFactorizationNet, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        
        # 用户和物品嵌入层
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        
        # 偏置项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        # 用正态分布初始化嵌入层
        nn.init.normal_(self.user_embedding.weight, 0, 0.1)
        nn.init.normal_(self.item_embedding.weight, 0, 0.1)
        
        # 偏置项初始化为0
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def forward(self, user_ids, item_ids):
        """前向传播"""
        # 获取嵌入向量
        user_embeds = self.user_embedding(user_ids)  # (batch_size, n_factors)
        item_embeds = self.item_embedding(item_ids)  # (batch_size, n_factors)
        
        # 应用dropout
        user_embeds = self.dropout(user_embeds)
        item_embeds = self.dropout(item_embeds)
        
        # 计算点积
        dot_product = torch.sum(user_embeds * item_embeds, dim=1)  # (batch_size,)
        
        # 添加偏置项
        user_bias = self.user_bias(user_ids).squeeze()  # (batch_size,)
        item_bias = self.item_bias(item_ids).squeeze()  # (batch_size,)
        
        # 最终预测
        predictions = self.global_bias + user_bias + item_bias + dot_product
        
        return predictions
    
    def predict(self, user_ids, item_ids):
        """预测评分"""
        self.eval()
        with torch.no_grad():
            if not isinstance(user_ids, torch.Tensor):
                user_ids = torch.LongTensor(user_ids)
                item_ids = torch.LongTensor(item_ids)
            
            predictions = self.forward(user_ids, item_ids)
            return predictions.cpu().numpy()

class PyTorchMatrixFactorization:
    """PyTorch矩阵分解训练器"""
    
    def __init__(self, n_factors=20, learning_rate=0.01, weight_decay=1e-5, 
                 dropout=0.1, batch_size=512):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        
    def fit(self, train_data, n_epochs=100, verbose=True):
        """训练模型"""
        # 准备数据
        user_ids = train_data['user_id'].values - 1  # 转换为0开始的索引
        item_ids = train_data['movie_id'].values - 1
        ratings = train_data['rating'].values
        
        self.n_users = max(user_ids) + 1
        self.n_items = max(item_ids) + 1
        
        # 创建数据集和数据加载器
        dataset = RatingDataset(user_ids, item_ids, ratings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 创建模型
        self.model = MatrixFactorizationNet(
            n_users=self.n_users,
            n_items=self.n_items, 
            n_factors=self.n_factors,
            dropout=self.dropout
        )
        
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=self.learning_rate, 
                             weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        
        if verbose:
            print(f"开始训练PyTorch矩阵分解模型...")
            print(f"用户数: {self.n_users}, 物品数: {self.n_items}")
            print(f"潜在因子数: {self.n_factors}, 批次大小: {self.batch_size}")
        
        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()
                
                # 前向传播
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.train_losses.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    def predict(self, user_ids, item_ids):
        """预测评分"""
        # 转换为0开始的索引
        user_ids = np.array(user_ids) - 1
        item_ids = np.array(item_ids) - 1
        
        return self.model.predict(user_ids, item_ids)
    
    def recommend(self, user_id, num_recommendations=10, exclude_rated=None):
        """为用户推荐物品"""
        user_idx = user_id - 1  # 转换为0开始的索引
        
        # 为该用户计算对所有物品的预测评分
        all_items = np.arange(self.n_items)
        user_ids = np.full(self.n_items, user_idx)
        
        predictions = self.model.predict(user_ids, all_items)
        
        # 创建推荐列表
        recommendations = [(i + 1, pred) for i, pred in enumerate(predictions)]
        
        # 排除已评分物品
        if exclude_rated is not None:
            rated_items = set(exclude_rated)
            recommendations = [(item_id, pred) for item_id, pred in recommendations 
                             if item_id not in rated_items]
        
        # 按预测评分排序
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:num_recommendations]
```

### 18.3.3. Evaluation Measures 评估指标

Evaluation is crucial for recommender systems. We need metrics that reflect how well our system performs in real-world scenarios. Think of it like grading a student's predictions about movie preferences - we need fair and meaningful ways to measure accuracy.

评估对推荐系统至关重要。我们需要反映系统在真实场景中表现的指标。可以想象成给学生的电影偏好预测打分 - 我们需要公平且有意义的方式来衡量准确性。

#### Rating Prediction Metrics 评分预测指标

For rating prediction tasks, we typically use regression metrics:
对于评分预测任务，我们通常使用回归指标：

```python
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class RatingEvaluator:
    """评分预测评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, verbose=True):
        """评估预测结果"""
        # 均方根误差 (RMSE)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 平均绝对误差 (MAE) 
        mae = mean_absolute_error(y_true, y_pred)
        
        # 平均绝对百分比误差 (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R²得分
        r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        
        self.metrics = {
            'RMSE': rmse,
            'MAE': mae, 
            'MAPE': mape,
            'R²': r2
        }
        
        if verbose:
            print("=== 评分预测评估结果 ===")
            print(f"RMSE (均方根误差): {rmse:.4f}")
            print(f"MAE (平均绝对误差): {mae:.4f}")
            print(f"MAPE (平均绝对百分比误差): {mape:.2f}%")
            print(f"R² (决定系数): {r2:.4f}")
            
            # 误差分布分析
            errors = y_pred - y_true
            print(f"\n=== 误差分析 ===")
            print(f"误差均值: {np.mean(errors):.4f}")
            print(f"误差标准差: {np.std(errors):.4f}")
            print(f"误差范围: [{np.min(errors):.4f}, {np.max(errors):.4f}]")
        
        return self.metrics
    
    def plot_predictions(self, y_true, y_pred, title="Prediction vs True Ratings"):
        """绘制预测值vs真实值散点图"""
        plt.figure(figsize=(10, 8))
        
        # 散点图
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True Ratings')
        plt.ylabel('Predicted Ratings')
        plt.title('Predictions vs True Values')
        plt.grid(True)
        
        # 误差直方图
        plt.subplot(2, 2, 2)
        errors = y_pred - y_true
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True)
        
        # 残差图
        plt.subplot(2, 2, 3)
        plt.scatter(y_pred, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Ratings')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
        
        # 评分分布对比
        plt.subplot(2, 2, 4)
        plt.hist(y_true, bins=20, alpha=0.5, label='True', edgecolor='black')
        plt.hist(y_pred, bins=20, alpha=0.5, label='Predicted', edgecolor='black')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title('Rating Distribution Comparison')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
```

#### Ranking Metrics 排序指标

For recommendation ranking tasks, we use different metrics that focus on the order and relevance of recommendations:
对于推荐排序任务，我们使用关注推荐顺序和相关性的不同指标：

```python
class RankingEvaluator:
    """排序推荐评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """计算Precision@K"""
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = set(recommended_k) & set(relevant_items)
        
        return len(relevant_recommended) / k
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """计算Recall@K"""
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = set(recommended_k) & set(relevant_items)
        
        return len(relevant_recommended) / len(relevant_items)
    
    def f1_at_k(self, recommended_items, relevant_items, k):
        """计算F1@K"""
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def average_precision(self, recommended_items, relevant_items):
        """计算平均精度 (AP)"""
        if len(relevant_items) == 0:
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                relevant_count += 1
                precision = relevant_count / (i + 1)
                precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(relevant_items)
    
    def ndcg_at_k(self, recommended_items, relevant_items, k, relevance_scores=None):
        """计算NDCG@K (Normalized Discounted Cumulative Gain)"""
        if relevance_scores is None:
            # 如果没有提供相关性分数，使用二进制相关性
            relevance_scores = {item: 1 for item in relevant_items}
        
        # 计算DCG@K
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevance_scores:
                relevance = relevance_scores[item]
                dcg += relevance / np.log2(i + 2)  # +2因为索引从0开始
        
        # 计算IDCG@K (Ideal DCG)
        ideal_items = sorted(relevant_items, 
                           key=lambda x: relevance_scores.get(x, 0), 
                           reverse=True)
        idcg = 0.0
        for i, item in enumerate(ideal_items[:k]):
            relevance = relevance_scores[item]
            idcg += relevance / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_ranking(self, recommendations, test_data, k_values=[5, 10, 20]):
        """评估排序推荐结果"""
        all_precisions = {k: [] for k in k_values}
        all_recalls = {k: [] for k in k_values}
        all_f1s = {k: [] for k in k_values}
        all_ndcgs = {k: [] for k in k_values}
        all_aps = []
        
        for user_id in recommendations:
            # 获取该用户的推荐列表
            recommended_items = [item_id for item_id, _ in recommendations[user_id]]
            
            # 获取该用户在测试集中的相关物品（高评分物品）
            user_test_data = test_data[test_data['user_id'] == user_id]
            relevant_items = user_test_data[user_test_data['rating'] >= 4]['movie_id'].tolist()
            
            if len(relevant_items) == 0:
                continue
            
            # 计算各种指标
            for k in k_values:
                precision = self.precision_at_k(recommended_items, relevant_items, k)
                recall = self.recall_at_k(recommended_items, relevant_items, k)
                f1 = self.f1_at_k(recommended_items, relevant_items, k)
                ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)
                
                all_precisions[k].append(precision)
                all_recalls[k].append(recall)
                all_f1s[k].append(f1)
                all_ndcgs[k].append(ndcg)
            
            # 计算平均精度
            ap = self.average_precision(recommended_items, relevant_items)
            all_aps.append(ap)
        
        # 计算平均值
        results = {}
        for k in k_values:
            results[f'Precision@{k}'] = np.mean(all_precisions[k])
            results[f'Recall@{k}'] = np.mean(all_recalls[k])
            results[f'F1@{k}'] = np.mean(all_f1s[k])
            results[f'NDCG@{k}'] = np.mean(all_ndcgs[k])
        
        results['MAP'] = np.mean(all_aps)  # Mean Average Precision
        
        return results
```

### 18.3.4. Training and Evaluating the Model 训练和评估模型

Now let's put everything together and train a complete matrix factorization model on the MovieLens dataset:

现在让我们将所有内容整合起来，在MovieLens数据集上训练一个完整的矩阵分解模型：

```python
def train_and_evaluate_mf():
    """训练和评估矩阵分解模型的完整流程"""
    
    # 1. 加载数据
    print("=== 1. 加载MovieLens数据 ===")
    loader = MovieLensDataLoader("ml-100k", "100k")
    ratings, movies, users = loader.load_all()
    
    # 2. 数据预处理
    print("\n=== 2. 数据预处理 ===")
    # 将评分数据转换为矩阵形式
    from scipy.sparse import csr_matrix
    
    # 创建用户-物品评分矩阵
    n_users = ratings['user_id'].nunique()
    n_movies = ratings['movie_id'].nunique()
    
    rating_matrix = np.zeros((n_users, n_movies))
    for row in ratings.itertuples():
        rating_matrix[row.user_id - 1, row.movie_id - 1] = row.rating
    
    print(f"评分矩阵形状: {rating_matrix.shape}")
    print(f"稀疏度: {np.count_nonzero(rating_matrix) / rating_matrix.size * 100:.2f}%")
    
    # 3. 数据分割
    print("\n=== 3. 数据分割 ===")
    train_data, test_data = temporal_split(ratings, test_ratio=0.2)
    
    # 4. 训练模型
    print("\n=== 4. 训练矩阵分解模型 ===")
    
    # 传统矩阵分解
    mf_model = MatrixFactorization(
        n_factors=50,
        learning_rate=0.01,
        regularization=0.01,
        n_epochs=100
    )
    
    # 创建训练用的评分矩阵
    train_matrix = np.zeros((n_users, n_movies))
    for row in train_data.itertuples():
        train_matrix[row.user_id - 1, row.movie_id - 1] = row.rating
    
    mf_model.fit(train_matrix)
    
    # PyTorch矩阵分解
    pytorch_mf = PyTorchMatrixFactorization(
        n_factors=50,
        learning_rate=0.001,
        batch_size=512
    )
    pytorch_mf.fit(train_data, n_epochs=50)
    
    # 5. 评估模型
    print("\n=== 5. 模型评估 ===")
    
    # 准备测试数据
    test_users = test_data['user_id'].values - 1
    test_items = test_data['movie_id'].values - 1  
    test_ratings = test_data['rating'].values
    
    # 传统矩阵分解预测
    mf_predictions = mf_model.predict(test_users, test_items)
    
    # PyTorch矩阵分解预测
    pytorch_predictions = pytorch_mf.predict(test_data['user_id'].values, 
                                           test_data['movie_id'].values)
    
    # 评估评分预测
    print("\n--- 传统矩阵分解评估 ---")
    rating_evaluator = RatingEvaluator()
    mf_metrics = rating_evaluator.evaluate(test_ratings, mf_predictions)
    
    print("\n--- PyTorch矩阵分解评估 ---")
    pytorch_metrics = rating_evaluator.evaluate(test_ratings, pytorch_predictions)
    
    # 绘制预测结果
    rating_evaluator.plot_predictions(test_ratings, pytorch_predictions, 
                                    "PyTorch Matrix Factorization Results")
    
    # 6. 生成推荐
    print("\n=== 6. 生成推荐示例 ===")
    
    # 为几个用户生成推荐
    sample_users = [1, 50, 100]
    
    for user_id in sample_users:
        print(f"\n--- 用户 {user_id} 的推荐 ---")
        
        # 获取用户已评分的电影
        user_ratings = train_data[train_data['user_id'] == user_id]
        rated_movies = user_ratings['movie_id'].tolist()
        
        print(f"用户已评分电影数: {len(rated_movies)}")
        print(f"平均评分: {user_ratings['rating'].mean():.2f}")
        
        # 生成推荐
        recommendations = pytorch_mf.recommend(user_id, 
                                             num_recommendations=10,
                                             exclude_rated=rated_movies)
        
        print("前10推荐电影:")
        for i, (movie_id, predicted_rating) in enumerate(recommendations):
            movie_title = movies[movies['movie_id'] == movie_id]['title'].iloc[0]
            print(f"{i+1}. {movie_title} (预测评分: {predicted_rating:.2f})")
    
    return mf_model, pytorch_mf, mf_metrics, pytorch_metrics

# 运行完整的训练和评估流程
if __name__ == "__main__":
    mf_model, pytorch_mf, mf_metrics, pytorch_metrics = train_and_evaluate_mf()
```

### 18.3.5. Summary 总结

Matrix Factorization is a powerful and interpretable approach for collaborative filtering. It discovers latent factors that explain user preferences and item characteristics, making it both effective and explainable. The key advantages include:

矩阵分解是协同过滤的一种强大且可解释的方法。它发现解释用户偏好和物品特征的潜在因子，使其既有效又可解释。主要优势包括：

- **Scalability**: Efficient for large datasets 可扩展性：对大数据集高效
- **Interpretability**: Latent factors can be analyzed 可解释性：可以分析潜在因子
- **Flexibility**: Easy to incorporate additional features 灵活性：易于融入额外特征
- **Effectiveness**: Strong baseline performance 有效性：强大的基准性能

However, it also has limitations such as the cold start problem for new users/items and difficulty handling non-linear relationships.

然而，它也有局限性，如新用户/物品的冷启动问题和难以处理非线性关系。

### 18.3.6. Exercises 练习

1. **Implement** different initialization strategies for matrix factorization and compare their convergence.
   **实现**矩阵分解的不同初始化策略并比较其收敛性。

2. **Experiment** with different numbers of latent factors and analyze the bias-variance tradeoff.
   **实验**不同数量的潜在因子并分析偏差-方差权衡。

3. **Add** temporal dynamics to capture changing user preferences over time.
   **添加**时间动态以捕获用户偏好随时间的变化。

4. **Compare** matrix factorization with user-based and item-based collaborative filtering.
   **比较**矩阵分解与基于用户和基于物品的协同过滤。

---

## 18.4. AutoRec: Rating Prediction with Autoencoders 自编码器评分预测

### 18.4.1. Model 模型

AutoRec (Auto-encoder for Collaborative Filtering) is a neural network-based approach that uses autoencoders to learn user or item representations for rating prediction. Unlike traditional matrix factorization methods, AutoRec can capture complex, non-linear relationships in the data.

AutoRec（协同过滤自编码器）是一种基于神经网络的方法，使用自编码器学习用户或物品表示来进行评分预测。与传统的矩阵分解方法不同，AutoRec能够捕获数据中复杂的非线性关系。

**Key Concepts 关键概念:**

1. **Item-based AutoRec (I-AutoRec)**: Takes item rating vectors as input
   基于物品的AutoRec：以物品评分向量作为输入

2. **User-based AutoRec (U-AutoRec)**: Takes user rating vectors as input
   基于用户的AutoRec：以用户评分向量作为输入

**Architecture 架构:**
```
Input Layer (Rating Vector) → Hidden Layer → Output Layer (Reconstructed Ratings)
输入层（评分向量） → 隐藏层 → 输出层（重构评分）
```

**Mathematical Formulation 数学公式:**

For I-AutoRec, given an item rating vector r^(i) ∈ R^m:
对于I-AutoRec，给定物品评分向量 r^(i) ∈ R^m：

```
h(r^(i)) = f(W · r^(i) + μ)
g(h(r^(i))) = f'(W' · h(r^(i)) + μ')
```

Where:
- f and f' are activation functions (typically sigmoid)
- W ∈ R^{d×m} and W' ∈ R^{m×d} are weight matrices
- μ and μ' are bias vectors
- d is the hidden layer dimension

其中：
- f 和 f' 是激活函数（通常是sigmoid）
- W ∈ R^{d×m} 和 W' ∈ R^{m×d} 是权重矩阵
- μ 和 μ' 是偏置向量
- d 是隐藏层维度

**Objective Function 目标函数:**

```
min_{W,W',μ,μ'} Σ_{i=1}^n ||r^(i) - g(h(r^(i)))||_O^2 + λ(||W||_F^2 + ||W'||_F^2)
```

Where ||·||_O denotes the masked norm (only considering observed ratings).
其中 ||·||_O 表示掩码范数（只考虑观察到的评分）。

### 18.4.2. Implementing the Model 模型实现

Let's implement AutoRec using PyTorch. We'll create both I-AutoRec and U-AutoRec variants.

让我们使用PyTorch实现AutoRec。我们将创建I-AutoRec和U-AutoRec两种变体。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AutoRec(nn.Module):
    """
    AutoRec model for collaborative filtering
    用于协同过滤的AutoRec模型
    """
    def __init__(self, num_hidden, num_users, num_items, dropout=0.05):
        super(AutoRec, self).__init__()
        self.num_hidden = num_hidden
        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout
        
        # 编码器：从输入到隐藏层
        # Encoder: from input to hidden layer
        self.encoder = nn.Linear(num_items, num_hidden)
        
        # 解码器：从隐藏层到输出
        # Decoder: from hidden layer to output
        self.decoder = nn.Linear(num_hidden, num_items)
        
        # Dropout层用于正则化
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout)
        
        # 权重初始化
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        """使用Xavier初始化权重"""
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.constant_(self.encoder.bias, 0.0)
        nn.init.constant_(self.decoder.bias, 0.0)
    
    def forward(self, rating_matrix):
        """
        Forward pass of AutoRec
        AutoRec的前向传播
        
        Args:
            rating_matrix: User-item rating matrix (batch_size, num_items)
            评分矩阵：用户-物品评分矩阵 (batch_size, num_items)
        
        Returns:
            reconstructed_ratings: Predicted ratings (batch_size, num_items)
            重构评分：预测评分 (batch_size, num_items)
        """
        # 编码阶段：输入 -> 隐藏表示
        # Encoding phase: input -> hidden representation
        hidden = F.sigmoid(self.encoder(rating_matrix))
        hidden = self.dropout_layer(hidden)
        
        # 解码阶段：隐藏表示 -> 重构输出
        # Decoding phase: hidden representation -> reconstructed output
        reconstructed = self.decoder(hidden)
        
        return reconstructed

# 创建损失函数类
# Create loss function class
class AutoRecLoss(nn.Module):
    """
    Masked MSE loss for AutoRec (only compute loss on observed ratings)
    AutoRec的掩码MSE损失（只在观察到的评分上计算损失）
    """
    def __init__(self, weight_decay=0.001):
        super(AutoRecLoss, self).__init__()
        self.weight_decay = weight_decay
        
    def forward(self, predictions, targets, mask, model):
        """
        Compute masked MSE loss with L2 regularization
        计算带有L2正则化的掩码MSE损失
        
        Args:
            predictions: Predicted ratings 预测评分
            targets: True ratings 真实评分
            mask: Binary mask indicating observed ratings 指示观察评分的二进制掩码
            model: AutoRec model for regularization 用于正则化的AutoRec模型
        """
        # 计算掩码MSE损失
        # Compute masked MSE loss
        mse_loss = torch.sum(mask * (predictions - targets) ** 2) / torch.sum(mask)
        
        # 添加L2正则化
        # Add L2 regularization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param, 2) ** 2
        
        total_loss = mse_loss + self.weight_decay * l2_reg
        return total_loss
```

### 18.4.3. Reimplementing the Evaluator 重新实现评估器

We need to create an evaluator that can handle the specific requirements of AutoRec, including masked evaluation and appropriate metrics.

我们需要创建一个评估器，能够处理AutoRec的特定要求，包括掩码评估和适当的指标。

```python
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class AutoRecEvaluator:
    """
    Evaluator for AutoRec model
    AutoRec模型的评估器
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
    def evaluate(self, test_loader, criterion):
        """
        Evaluate the model on test data
        在测试数据上评估模型
        
        Args:
            test_loader: DataLoader for test data 测试数据的DataLoader
            criterion: Loss function 损失函数
            
        Returns:
            Dictionary with evaluation metrics 包含评估指标的字典
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_masks = []
        
        with torch.no_grad():
            for batch_idx, (ratings, masks) in enumerate(test_loader):
                ratings = ratings.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                # Forward pass
                predictions = self.model(ratings)
                
                # 计算损失
                # Compute loss
                loss = criterion(predictions, ratings, masks, self.model)
                total_loss += loss.item()
                
                # 收集预测值和真实值（只考虑观察到的评分）
                # Collect predictions and targets (only observed ratings)
                masked_predictions = predictions * masks
                masked_targets = ratings * masks
                
                all_predictions.extend(masked_predictions[masks == 1].cpu().numpy())
                all_targets.extend(masked_targets[masks == 1].cpu().numpy())
        
        # 计算评估指标
        # Compute evaluation metrics
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        avg_loss = total_loss / len(test_loader)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'loss': avg_loss,
            'num_samples': len(all_targets)
        }
    
    def predict_ratings(self, user_ratings, mask=None):
        """
        Predict ratings for a single user
        为单个用户预测评分
        
        Args:
            user_ratings: User's rating vector 用户评分向量
            mask: Mask for observed ratings 观察评分的掩码
            
        Returns:
            Predicted ratings 预测评分
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(user_ratings, np.ndarray):
                user_ratings = torch.FloatTensor(user_ratings).unsqueeze(0)
            
            user_ratings = user_ratings.to(self.device)
            predictions = self.model(user_ratings)
            
            if mask is not None:
                # 只返回未观察到的评分预测
                # Only return predictions for unobserved ratings
                mask = torch.FloatTensor(mask).to(self.device)
                predictions = predictions * (1 - mask)  # 掩盖已知评分
            
            return predictions.squeeze().cpu().numpy()
```

### 18.4.4. Training and Evaluating the Model 训练和评估模型

Now let's implement the training pipeline and see how to use AutoRec in practice.

现在让我们实现训练流水线，看看如何在实践中使用AutoRec。

```python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

class AutoRecTrainer:
    """
    Trainer class for AutoRec model
    AutoRec模型的训练器类
    """
    def __init__(self, model, device='cpu', learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = AutoRecLoss()
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        训练一个epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (ratings, masks) in enumerate(train_loader):
            ratings = ratings.to(self.device)
            masks = masks.to(self.device)
            
            # 梯度清零
            # Zero gradients
            self.optimizer.zero_grad()
            
            # 前向传播
            # Forward pass
            predictions = self.model(ratings)
            
            # 计算损失
            # Compute loss
            loss = self.criterion(predictions, ratings, masks, self.model)
            
            # 反向传播
            # Backward pass
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            # Gradient clipping (prevent gradient explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 参数更新
            # Parameter update
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """
        Validate the model
        验证模型
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for ratings, masks in val_loader:
                ratings = ratings.to(self.device)
                masks = masks.to(self.device)
                
                predictions = self.model(ratings)
                loss = self.criterion(predictions, ratings, masks, self.model)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs, patience=10):
        """
        Full training loop with early stopping
        带有早停的完整训练循环
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练AutoRec模型...")
        print(f"Starting AutoRec model training...")
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            # Train one epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            # Validation
            val_loss = self.validate(val_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  训练损失 Train Loss: {train_loss:.4f}')
            print(f'  验证损失 Val Loss: {val_loss:.4f}')
            
            # 早停检查
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                # Save best model
                torch.save(self.model.state_dict(), 'best_autorec_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                print(f'在第{epoch+1}个epoch后触发早停')
                break
        
        # 加载最佳模型
        # Load best model
        self.model.load_state_dict(torch.load('best_autorec_model.pth'))
        
    def plot_training_curves(self):
        """
        Plot training and validation loss curves
        绘制训练和验证损失曲线
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss 训练损失', color='blue')
        plt.plot(self.val_losses, label='Validation Loss 验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('AutoRec Training Progress AutoRec训练进度')
        plt.legend()
        plt.grid(True)
        plt.show()

# 示例使用方法
# Example usage
def create_sample_data(num_users=1000, num_items=500, sparsity=0.05):
    """
    Create sample rating data for demonstration
    创建示例评分数据用于演示
    """
    # 创建稀疏评分矩阵
    # Create sparse rating matrix
    ratings = np.zeros((num_users, num_items))
    masks = np.zeros((num_users, num_items))
    
    # 随机选择用户-物品对进行评分
    # Randomly select user-item pairs for rating
    num_ratings = int(num_users * num_items * sparsity)
    user_indices = np.random.randint(0, num_users, num_ratings)
    item_indices = np.random.randint(0, num_items, num_ratings)
    
    # 生成1-5的评分
    # Generate ratings from 1-5
    rating_values = np.random.randint(1, 6, num_ratings)
    
    for i, (u, item, rating) in enumerate(zip(user_indices, item_indices, rating_values)):
        ratings[u, item] = rating
        masks[u, item] = 1
    
    return ratings, masks

# 训练示例
# Training example
if __name__ == "__main__":
    # 设置设备
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建示例数据
    # Create sample data
    ratings, masks = create_sample_data(num_users=1000, num_items=500)
    
    # 转换为张量
    # Convert to tensors
    ratings_tensor = torch.FloatTensor(ratings)
    masks_tensor = torch.FloatTensor(masks)
    
    # 创建数据集和数据加载器
    # Create dataset and data loader
    dataset = TensorDataset(ratings_tensor, masks_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 创建模型
    # Create model
    model = AutoRec(num_hidden=128, num_users=1000, num_items=500, dropout=0.1)
    
    # 创建训练器
    # Create trainer
    trainer = AutoRecTrainer(model, device=device, learning_rate=0.001)
    
    # 训练模型
    # Train model
    trainer.train(train_loader, val_loader, num_epochs=100, patience=10)
    
    # 绘制训练曲线
    # Plot training curves
    trainer.plot_training_curves()
    
    # 评估模型
    # Evaluate model
    evaluator = AutoRecEvaluator(model, device=device)
    test_metrics = evaluator.evaluate(val_loader, trainer.criterion)
    
    print("Test Results 测试结果:")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAE: {test_metrics['mae']:.4f}")
    print(f"Loss: {test_metrics['loss']:.4f}")
```

### 18.4.5. Summary 总结

AutoRec represents a significant advancement in collaborative filtering by introducing neural networks and autoencoders to the recommendation problem. The key advantages include:

AutoRec代表了协同过滤的重大进步，通过将神经网络和自编码器引入推荐问题。主要优势包括：

**Advantages 优势:**
1. **Non-linear modeling**: Can capture complex user-item interactions
   非线性建模：能够捕获复杂的用户-物品交互

2. **Scalability**: More efficient than traditional neighborhood methods
   可扩展性：比传统的邻域方法更高效

3. **Flexibility**: Easy to incorporate additional features
   灵活性：易于融入额外特征

4. **Performance**: Often outperforms traditional matrix factorization
   性能：通常优于传统矩阵分解

**Limitations 局限性:**
1. **Cold start problem**: Difficulty with new users/items
   冷启动问题：难以处理新用户/物品

2. **Interpretability**: Less interpretable than matrix factorization
   可解释性：比矩阵分解的可解释性差

3. **Training complexity**: Requires careful hyperparameter tuning
   训练复杂性：需要仔细的超参数调优

### 18.4.6. Exercises 练习

1. **Implementation Exercise 实现练习:**
   Implement a user-based AutoRec (U-AutoRec) variant and compare its performance with item-based AutoRec.
   实现基于用户的AutoRec（U-AutoRec）变体，并与基于物品的AutoRec比较性能。

2. **Architecture Modification 架构修改:**
   Add multiple hidden layers to create a deeper autoencoder. Analyze how depth affects performance.
   添加多个隐藏层以创建更深的自编码器。分析深度如何影响性能。

3. **Regularization Study 正则化研究:**
   Experiment with different regularization techniques (L1, L2, dropout) and their combinations.
   实验不同的正则化技术（L1、L2、dropout）及其组合。

4. **Activation Function Analysis 激活函数分析:**
   Compare different activation functions (ReLU, Tanh, ELU) in the hidden layer.
   比较隐藏层中不同的激活函数（ReLU、Tanh、ELU）。

## 18.5. Personalized Ranking for Recommender Systems 个性化排序推荐系统

### Overview 概述

Personalized ranking is a fundamental task in recommender systems where the goal is to learn a ranking function that orders items according to user preferences. Unlike rating prediction, ranking focuses on the relative order of items rather than absolute rating values.

个性化排序是推荐系统中的一个基本任务，目标是学习一个根据用户偏好对物品进行排序的排序函数。与评分预测不同，排序关注的是物品的相对顺序而不是绝对评分值。

Think of this like organizing your music playlist. You don't need to assign specific numerical scores to each song; you just need to know which songs you prefer over others.

可以把这想象成整理你的音乐播放列表。你不需要给每首歌分配具体的数字分数；你只需要知道哪些歌比其他歌更受欢迎。

### 18.5.1. Bayesian Personalized Ranking Loss and its Implementation 贝叶斯个性化排序损失及其实现

Bayesian Personalized Ranking (BPR) is a widely-used framework for learning personalized rankings from implicit feedback data. The core idea is to optimize pairwise preferences: for each user, items they have interacted with should be ranked higher than items they haven't.

贝叶斯个性化排序（BPR）是一个广泛使用的框架，用于从隐式反馈数据中学习个性化排序。核心思想是优化成对偏好：对于每个用户，他们交互过的物品应该比没有交互过的物品排名更高。

**Mathematical Foundation 数学基础:**

Given a user u, a positive item i (interacted), and a negative item j (not interacted), BPR assumes:
给定用户u、正样本物品i（有交互）和负样本物品j（无交互），BPR假设：

```
P(i >_u j) = σ(x_uij)
```

Where x_uij = ŷ_ui - ŷ_uj (difference in predicted preferences)
其中 x_uij = ŷ_ui - ŷ_uj（预测偏好的差异）

**BPR Loss Function BPR损失函数:**

```
L_BPR = Σ_{(u,i,j)∈D_S} -ln σ(x_uij) + λ||Θ||²
```

Where:
- D_S is the training set of (user, positive_item, negative_item) triplets
- σ is the sigmoid function
- Θ represents model parameters
- λ is the regularization coefficient

其中：
- D_S 是（用户，正物品，负物品）三元组的训练集
- σ 是sigmoid函数
- Θ 表示模型参数
- λ 是正则化系数

**Implementation 实现:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking Loss
    贝叶斯个性化排序损失
    """
    def __init__(self, weight_decay=0.01):
        super(BPRLoss, self).__init__()
        self.weight_decay = weight_decay
        
    def forward(self, pos_scores, neg_scores, model_params=None):
        """
        Compute BPR loss
        计算BPR损失
        
        Args:
            pos_scores: Scores for positive items 正物品的分数
            neg_scores: Scores for negative items 负物品的分数
            model_params: Model parameters for regularization 用于正则化的模型参数
        """
        # 计算偏好差异
        # Compute preference difference
        diff_scores = pos_scores - neg_scores
        
        # BPR损失：-log(sigmoid(x_uij))
        # BPR loss: -log(sigmoid(x_uij))
        bpr_loss = -F.logsigmoid(diff_scores).mean()
        
        # 添加L2正则化
        # Add L2 regularization
        if model_params is not None and self.weight_decay > 0:
            l2_reg = 0
            for param in model_params:
                l2_reg += torch.norm(param, 2) ** 2
            total_loss = bpr_loss + self.weight_decay * l2_reg
        else:
            total_loss = bpr_loss
            
        return total_loss

class BPRDataset(Dataset):
    """
    Dataset for BPR training with negative sampling
    用于BPR训练的数据集（带负采样）
    """
    def __init__(self, user_item_pairs, num_items, num_negatives=1):
        """
        Args:
            user_item_pairs: List of (user_id, item_id) positive interactions
                            正交互的(user_id, item_id)列表
            num_items: Total number of items 物品总数
            num_negatives: Number of negative samples per positive 每个正样本的负样本数量
        """
        self.user_item_pairs = user_item_pairs
        self.num_items = num_items
        self.num_negatives = num_negatives
        
        # 构建用户交互集合（用于负采样）
        # Build user interaction sets (for negative sampling)
        self.user_items = {}
        for user_id, item_id in user_item_pairs:
            if user_id not in self.user_items:
                self.user_items[user_id] = set()
            self.user_items[user_id].add(item_id)
    
    def __len__(self):
        return len(self.user_item_pairs) * self.num_negatives
    
    def __getitem__(self, idx):
        # 获取正样本
        # Get positive sample
        pos_idx = idx // self.num_negatives
        user_id, pos_item_id = self.user_item_pairs[pos_idx]
        
        # 负采样：选择用户没有交互过的物品
        # Negative sampling: select items user hasn't interacted with
        neg_item_id = self._negative_sampling(user_id)
        
        return torch.LongTensor([user_id]), torch.LongTensor([pos_item_id]), torch.LongTensor([neg_item_id])
    
    def _negative_sampling(self, user_id):
        """
        Sample a negative item for the given user
        为给定用户采样一个负物品
        """
        user_pos_items = self.user_items.get(user_id, set())
        
        while True:
            neg_item = np.random.randint(0, self.num_items)
            if neg_item not in user_pos_items:
                return neg_item

class MatrixFactorizationBPR(nn.Module):
    """
    Matrix Factorization model with BPR loss
    使用BPR损失的矩阵分解模型
    """
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(MatrixFactorizationBPR, self).__init__()
        
        # 用户和物品嵌入
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 偏置项
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        
        # 权重初始化
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small random values"""
        """用小的随机值初始化嵌入"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0)
        nn.init.constant_(self.item_bias.weight, 0)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass to compute user-item scores
        前向传播计算用户-物品分数
        
        Args:
            user_ids: User IDs tensor 用户ID张量
            item_ids: Item IDs tensor 物品ID张量
            
        Returns:
            scores: Predicted user-item interaction scores 预测的用户-物品交互分数
        """
        # 获取嵌入
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)  # (batch_size, embedding_dim)
        item_embeds = self.item_embedding(item_ids)  # (batch_size, embedding_dim)
        
        # 获取偏置
        # Get biases
        user_bias = self.user_bias(user_ids).squeeze()  # (batch_size,)
        item_bias = self.item_bias(item_ids).squeeze()  # (batch_size,)
        
        # 计算内积
        # Compute dot product
        dot_product = torch.sum(user_embeds * item_embeds, dim=1)  # (batch_size,)
        
        # 最终分数：内积 + 偏置
        # Final score: dot product + biases
        scores = dot_product + user_bias + item_bias + self.global_bias
        
        return scores
    
    def predict(self, user_id, item_ids):
        """
        Predict scores for a user on multiple items
        为一个用户预测多个物品的分数
        """
        self.eval()
        with torch.no_grad():
            user_ids = torch.LongTensor([user_id] * len(item_ids))
            item_ids = torch.LongTensor(item_ids)
            scores = self.forward(user_ids, item_ids)
            return scores.numpy()

# 训练器类
# Trainer class
class BPRTrainer:
    """
    Trainer for BPR-based models
    基于BPR的模型训练器
    """
    def __init__(self, model, device='cpu', learning_rate=0.01):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = BPRLoss()
        self.train_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (user_ids, pos_item_ids, neg_item_ids) in enumerate(train_loader):
            user_ids = user_ids.squeeze().to(self.device)
            pos_item_ids = pos_item_ids.squeeze().to(self.device)
            neg_item_ids = neg_item_ids.squeeze().to(self.device)
            
            # 梯度清零
            # Zero gradients
            self.optimizer.zero_grad()
            
            # 计算正负样本分数
            # Compute positive and negative scores
            pos_scores = self.model(user_ids, pos_item_ids)
            neg_scores = self.model(user_ids, neg_item_ids)
            
            # 计算BPR损失
            # Compute BPR loss
            loss = self.criterion(pos_scores, neg_scores, self.model.parameters())
            
            # 反向传播和优化
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 500 == 0:
                print(f'Batch {batch_idx}, BPR Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, num_epochs):
        """Full training loop"""
        print("开始BPR训练...")
        print("Starting BPR training...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, 平均损失 Average Loss: {train_loss:.4f}')
        
        print("BPR训练完成！")
        print("BPR training completed!")

# 示例使用
# Example usage
def demonstrate_bpr():
    """
    Demonstrate BPR training with synthetic data
    使用合成数据演示BPR训练
    """
    # 创建合成数据
    # Create synthetic data
    num_users, num_items = 1000, 500
    num_interactions = 5000
    
    # 生成随机用户-物品交互
    # Generate random user-item interactions
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    user_item_pairs = list(zip(user_ids, item_ids))
    
    # 创建数据集和数据加载器
    # Create dataset and data loader
    dataset = BPRDataset(user_item_pairs, num_items, num_negatives=1)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 创建模型
    # Create model
    model = MatrixFactorizationBPR(num_users, num_items, embedding_dim=64)
    
    # 训练模型
    # Train model
    trainer = BPRTrainer(model, learning_rate=0.01)
    trainer.train(train_loader, num_epochs=10)
    
    # 测试预测
    # Test prediction
    test_user = 0
    test_items = [0, 1, 2, 3, 4]
    scores = model.predict(test_user, test_items)
    print(f"\n用户 {test_user} 对物品 {test_items} 的预测分数:")
    print(f"User {test_user} predicted scores for items {test_items}:")
    print(scores)

if __name__ == "__main__":
    demonstrate_bpr()
```

### 18.5.2. Hinge Loss and its Implementation Hinge损失及其实现

Hinge Loss is another popular ranking loss function borrowed from Support Vector Machines. It provides a margin-based approach to ranking, which can be more robust than BPR in certain scenarios.

Hinge损失是另一个流行的排序损失函数，借鉴自支持向量机。它提供了一种基于边际的排序方法，在某些场景下比BPR更加鲁棒。

**Mathematical Foundation 数学基础:**

The hinge loss for ranking aims to ensure that positive items have scores higher than negative items by at least a margin δ:

排序的hinge损失旨在确保正物品的分数比负物品高至少一个边际δ：

```
L_hinge = Σ_{(u,i,j)} max(0, δ - (ŷ_ui - ŷ_uj))
```

Where:
- δ is the margin (typically set to 1)
- ŷ_ui and ŷ_uj are predicted scores for positive and negative items

其中：
- δ 是边际（通常设为1）
- ŷ_ui 和 ŷ_uj 是正负物品的预测分数

**Implementation 实现:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HingeLoss(nn.Module):
    """
    Hinge Loss for ranking
    用于排序的Hinge损失
    """
    def __init__(self, margin=1.0, weight_decay=0.01):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.weight_decay = weight_decay
        
    def forward(self, pos_scores, neg_scores, model_params=None):
        """
        Compute hinge loss
        计算hinge损失
        
        Args:
            pos_scores: Scores for positive items 正物品分数
            neg_scores: Scores for negative items 负物品分数
            model_params: Model parameters for regularization 正则化参数
        """
        # 计算边际损失：max(0, margin - (pos_score - neg_score))
        # Compute margin loss: max(0, margin - (pos_score - neg_score))
        diff_scores = pos_scores - neg_scores
        hinge_loss = F.relu(self.margin - diff_scores).mean()
        
        # 添加L2正则化
        # Add L2 regularization
        if model_params is not None and self.weight_decay > 0:
            l2_reg = 0
            for param in model_params:
                l2_reg += torch.norm(param, 2) ** 2
            total_loss = hinge_loss + self.weight_decay * l2_reg
        else:
            total_loss = hinge_loss
            
        return total_loss

class RankingMLP(nn.Module):
    """
    Multi-Layer Perceptron for ranking with user and item features
    用于排序的多层感知机（包含用户和物品特征）
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_dims=[128, 64]):
        super(RankingMLP, self).__init__()
        
        # 嵌入层
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP层
        # MLP layers
        input_dim = embedding_dim * 2  # 用户和物品嵌入连接
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, user_ids, item_ids):
        """
        Forward pass
        前向传播
        """
        # 获取嵌入
        # Get embeddings
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # 连接用户和物品嵌入
        # Concatenate user and item embeddings
        combined = torch.cat([user_embeds, item_embeds], dim=1)
        
        # 通过MLP
        # Pass through MLP
        scores = self.mlp(combined).squeeze()
        
        return scores

class HingeTrainer:
    """
    Trainer for Hinge Loss-based ranking models
    基于Hinge损失的排序模型训练器
    """
    def __init__(self, model, device='cpu', learning_rate=0.001, margin=1.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = HingeLoss(margin=margin)
        self.train_losses = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch with hinge loss"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (user_ids, pos_item_ids, neg_item_ids) in enumerate(train_loader):
            user_ids = user_ids.squeeze().to(self.device)
            pos_item_ids = pos_item_ids.squeeze().to(self.device)
            neg_item_ids = neg_item_ids.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # 计算正负样本分数
            # Compute positive and negative scores
            pos_scores = self.model(user_ids, pos_item_ids)
            neg_scores = self.model(user_ids, neg_item_ids)
            
            # 计算Hinge损失
            # Compute Hinge loss
            loss = self.criterion(pos_scores, neg_scores, self.model.parameters())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 500 == 0:
                print(f'Batch {batch_idx}, Hinge Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, num_epochs):
        """Full training loop"""
        print("开始Hinge损失训练...")
        print("Starting Hinge loss training...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, 平均损失 Average Loss: {train_loss:.4f}')
        
        print("Hinge损失训练完成！")
        print("Hinge loss training completed!")

# 比较BPR和Hinge损失的实验
# Experiment comparing BPR and Hinge loss
def compare_ranking_losses():
    """
    Compare BPR and Hinge loss on the same data
    在相同数据上比较BPR和Hinge损失
    """
    # 数据准备
    # Data preparation
    num_users, num_items = 1000, 500
    num_interactions = 5000
    
    user_ids = np.random.randint(0, num_users, num_interactions)
    item_ids = np.random.randint(0, num_items, num_interactions)
    user_item_pairs = list(zip(user_ids, item_ids))
    
    dataset = BPRDataset(user_item_pairs, num_items, num_negatives=1)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # BPR模型训练
    # BPR model training
    print("=" * 50)
    print("训练BPR模型")
    print("Training BPR Model")
    print("=" * 50)
    
    bpr_model = MatrixFactorizationBPR(num_users, num_items, embedding_dim=64)
    bpr_trainer = BPRTrainer(bpr_model, learning_rate=0.01)
    bpr_trainer.train(train_loader, num_epochs=5)
    
    # Hinge损失模型训练
    # Hinge loss model training
    print("\n" + "=" * 50)
    print("训练Hinge损失模型")
    print("Training Hinge Loss Model")
    print("=" * 50)
    
    hinge_model = RankingMLP(num_users, num_items, embedding_dim=64)
    hinge_trainer = HingeTrainer(hinge_model, learning_rate=0.001, margin=1.0)
    hinge_trainer.train(train_loader, num_epochs=5)
    
    # 比较结果
    # Compare results
    print("\n" + "=" * 50)
    print("模型比较结果")
    print("Model Comparison Results")
    print("=" * 50)
    
    test_user = 0
    test_items = [0, 1, 2, 3, 4]
    
    # BPR预测
    bpr_scores = bpr_model.predict(test_user, test_items)
    print(f"BPR模型预测分数 BPR Model Scores: {bpr_scores}")
    
    # Hinge预测
    hinge_model.eval()
    with torch.no_grad():
        user_tensor = torch.LongTensor([test_user] * len(test_items))
        item_tensor = torch.LongTensor(test_items)
        hinge_scores = hinge_model(user_tensor, item_tensor).numpy()
    print(f"Hinge模型预测分数 Hinge Model Scores: {hinge_scores}")
    
    # 损失曲线比较
    # Loss curve comparison
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(bpr_trainer.train_losses, label='BPR Loss', color='blue')
    plt.title('BPR Training Loss BPR训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(hinge_trainer.train_losses, label='Hinge Loss', color='red')
    plt.title('Hinge Training Loss Hinge训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_ranking_losses()
```

### 18.5.3. Summary 总结

Personalized ranking is essential for modern recommender systems, especially when dealing with implicit feedback data. Both BPR and Hinge loss provide effective ways to learn from pairwise preferences:

个性化排序对现代推荐系统至关重要，特别是在处理隐式反馈数据时。BPR和Hinge损失都提供了从成对偏好中学习的有效方法：

**BPR (Bayesian Personalized Ranking) 贝叶斯个性化排序:**
- **Advantages 优势:** Probabilistic foundation, smooth gradients, works well with sparse data
  概率基础、平滑梯度、在稀疏数据上表现良好
- **Disadvantages 劣势:** Sensitive to hyperparameters, may converge slowly
  对超参数敏感、可能收敛较慢

**Hinge Loss Hinge损失:**
- **Advantages 优势:** Margin-based approach, more robust to outliers, faster convergence
  基于边际的方法、对异常值更鲁棒、收敛更快
- **Disadvantages 劣势:** Less probabilistic interpretation, requires margin tuning
  概率解释性较差、需要调整边际参数

**Common Applications 常见应用:**
1. **E-commerce**: Product recommendations 电商产品推荐
2. **Content platforms**: Article/video ranking 内容平台的文章/视频排序
3. **Social media**: Friend/content suggestions 社交媒体的朋友/内容建议
4. **Music/Movie platforms**: Playlist generation 音乐/电影平台的播放列表生成

### 18.5.4. Exercises 练习

1. **Loss Function Comparison 损失函数比较:**
   Implement additional ranking losses (e.g., logistic loss, WARP loss) and compare their performance.
   实现额外的排序损失（如logistic损失、WARP损失）并比较它们的性能。

2. **Negative Sampling Strategies 负采样策略:**
   Experiment with different negative sampling strategies (uniform, popularity-based, hard negative mining).
   实验不同的负采样策略（均匀采样、基于流行度、困难负样本挖掘）。

3. **Margin Analysis 边际分析:**
   Study how different margin values in Hinge loss affect model performance and convergence.
   研究Hinge损失中不同边际值如何影响模型性能和收敛。

4. **Multi-Class Ranking 多类排序:**
   Extend the ranking framework to handle multiple levels of preferences (e.g., like, neutral, dislike).
   扩展排序框架以处理多级偏好（如喜欢、中立、不喜欢）。 

## 18.6. Neural Collaborative Filtering for Personalized Ranking 个性化排序的神经协同过滤

Neural Collaborative Filtering (NCF) represents a significant advancement in recommender systems by leveraging deep learning to overcome the limitations of traditional matrix factorization. While matrix factorization assumes linear relationships between users and items, NCF can capture complex, non-linear interactions through neural networks.

神经协同过滤（NCF）通过利用深度学习克服传统矩阵分解的局限性，代表了推荐系统的重大进步。虽然矩阵分解假设用户和物品之间存在线性关系，但NCF可以通过神经网络捕获复杂的非线性交互。

Think of traditional matrix factorization as assuming that user preferences can be explained by simple, independent factors (like "action movie lover" + "sci-fi enthusiast"). NCF, however, recognizes that preferences might have complex interdependencies - perhaps someone likes action movies only when they're also sci-fi, but not fantasy action movies.

传统矩阵分解假设用户偏好可以用简单、独立的因子来解释（如"动作片爱好者"+"科幻爱好者"）。然而，NCF认识到偏好可能存在复杂的相互依赖关系 - 也许有人只喜欢科幻动作片，但不喜欢奇幻动作片。

### 18.6.1. The NeuMF Model NeuMF模型

The Neural Matrix Factorization (NeuMF) model combines the strengths of traditional matrix factorization with the expressiveness of neural networks. It consists of two main components that are eventually fused together:

神经矩阵分解（NeuMF）模型结合了传统矩阵分解的优势和神经网络的表达能力。它由两个主要组件组成，最终融合在一起：

#### Generalized Matrix Factorization (GMF) 广义矩阵分解

GMF extends traditional matrix factorization by replacing the inner product with a more flexible neural network layer. Instead of simply multiplying user and item embeddings, it uses element-wise multiplication followed by a linear transformation:

GMF通过用更灵活的神经网络层替代内积来扩展传统矩阵分解。它不是简单地相乘用户和物品嵌入，而是使用逐元素乘法，然后进行线性变换：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GMF(nn.Module):
    """广义矩阵分解模型"""
    
    def __init__(self, num_users, num_items, embedding_size):
        """
        初始化GMF模型
        
        参数:
        - num_users: 用户数量
        - num_items: 物品数量  
        - embedding_size: 嵌入维度
        """
        super(GMF, self).__init__()
        
        # 用户和物品嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # 输出层，将嵌入维度映射到1（预测评分）
        self.output_layer = nn.Linear(embedding_size, 1)
        
        # 初始化嵌入层权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 使用正态分布初始化嵌入层
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, user_indices, item_indices):
        """
        前向传播
        
        参数:
        - user_indices: 用户索引 [batch_size]
        - item_indices: 物品索引 [batch_size]
        
        返回:
        - predictions: 预测评分 [batch_size, 1]
        """
        # 获取用户和物品嵌入
        user_embed = self.user_embedding(user_indices)  # [batch_size, embedding_size]
        item_embed = self.item_embedding(item_indices)  # [batch_size, embedding_size]
        
        # 逐元素相乘（Hadamard积）
        element_wise_product = user_embed * item_embed  # [batch_size, embedding_size]
        
        # 通过输出层得到预测
        predictions = self.output_layer(element_wise_product)  # [batch_size, 1]
        
        return predictions

# 使用示例
def demonstrate_gmf():
    """演示GMF模型的使用"""
    
    # 模拟数据
    num_users, num_items = 1000, 500
    embedding_size = 64
    batch_size = 32
    
    # 创建模型
    model = GMF(num_users, num_items, embedding_size)
    
    # 模拟输入数据
    user_indices = torch.randint(0, num_users, (batch_size,))
    item_indices = torch.randint(0, num_items, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        predictions = model(user_indices, item_indices)
    
    print(f"GMF模型输出形状: {predictions.shape}")
    print(f"预测值范围: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    
    return model

# 运行演示
gmf_model = demonstrate_gmf()
```

#### Multi-Layer Perceptron (MLP) 多层感知机

The MLP component captures more complex, non-linear interactions between users and items. It concatenates user and item embeddings and passes them through multiple hidden layers:

MLP组件捕获用户和物品之间更复杂的非线性交互。它连接用户和物品嵌入，并通过多个隐藏层传递：

```python
class MLP(nn.Module):
    """多层感知机组件"""
    
    def __init__(self, num_users, num_items, embedding_size, hidden_sizes, dropout_rate=0.2):
        """
        初始化MLP模型
        
        参数:
        - num_users: 用户数量
        - num_items: 物品数量
        - embedding_size: 嵌入维度
        - hidden_sizes: 隐藏层大小列表，如[128, 64, 32]
        - dropout_rate: Dropout比率
        """
        super(MLP, self).__init__()
        
        # 用户和物品嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        
        # 构建MLP层
        layers = []
        input_size = embedding_size * 2  # 连接用户和物品嵌入
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(input_size, 1))
        
        self.mlp_layers = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        
        # 初始化MLP层权重
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, user_indices, item_indices):
        """
        前向传播
        
        参数:
        - user_indices: 用户索引
        - item_indices: 物品索引
        
        返回:
        - predictions: 预测评分
        """
        # 获取嵌入
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)
        
        # 连接用户和物品嵌入
        concat_embed = torch.cat([user_embed, item_embed], dim=1)
        
        # 通过MLP层
        predictions = self.mlp_layers(concat_embed)
        
        return predictions

# 演示MLP模型
def demonstrate_mlp():
    """演示MLP模型的使用"""
    
    num_users, num_items = 1000, 500
    embedding_size = 64
    hidden_sizes = [128, 64, 32]  # 三个隐藏层
    batch_size = 32
    
    # 创建模型
    model = MLP(num_users, num_items, embedding_size, hidden_sizes)
    
    # 模拟输入
    user_indices = torch.randint(0, num_users, (batch_size,))
    item_indices = torch.randint(0, num_items, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        predictions = model(user_indices, item_indices)
    
    print(f"MLP模型输出形状: {predictions.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    return model

mlp_model = demonstrate_mlp()
```

#### NeuMF Integration NeuMF集成

The NeuMF model combines GMF and MLP by concatenating their outputs and passing through a final prediction layer. This fusion allows the model to capture both linear and non-linear interactions:

NeuMF模型通过连接GMF和MLP的输出并通过最终预测层来组合它们。这种融合允许模型捕获线性和非线性交互：

```python
class NeuMF(nn.Module):
    """神经矩阵分解模型"""
    
    def __init__(self, num_users, num_items, 
                 gmf_embedding_size=64, mlp_embedding_size=64,
                 mlp_hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        """
        初始化NeuMF模型
        
        参数:
        - num_users: 用户数量
        - num_items: 物品数量
        - gmf_embedding_size: GMF嵌入维度
        - mlp_embedding_size: MLP嵌入维度
        - mlp_hidden_sizes: MLP隐藏层大小
        - dropout_rate: Dropout比率
        """
        super(NeuMF, self).__init__()
        
        # GMF组件
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_embedding_size)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_embedding_size)
        
        # MLP组件
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embedding_size)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embedding_size)
        
        # MLP隐藏层
        mlp_layers = []
        input_size = mlp_embedding_size * 2
        
        for hidden_size in mlp_hidden_sizes:
            mlp_layers.append(nn.Linear(input_size, hidden_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        self.mlp_layers = nn.Sequential(*mlp_layers)
        
        # 融合层：连接GMF和MLP的输出
        fusion_input_size = gmf_embedding_size + mlp_hidden_sizes[-1]
        self.fusion_layer = nn.Linear(fusion_input_size, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化GMF嵌入
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        
        # 初始化MLP嵌入
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        # 初始化MLP层
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        # 初始化融合层
        nn.init.xavier_normal_(self.fusion_layer.weight)
        nn.init.constant_(self.fusion_layer.bias, 0)
    
    def forward(self, user_indices, item_indices):
        """
        前向传播
        
        参数:
        - user_indices: 用户索引
        - item_indices: 物品索引
        
        返回:
        - predictions: 预测评分
        """
        # GMF部分
        gmf_user_embed = self.gmf_user_embedding(user_indices)
        gmf_item_embed = self.gmf_item_embedding(item_indices)
        gmf_output = gmf_user_embed * gmf_item_embed  # 逐元素相乘
        
        # MLP部分
        mlp_user_embed = self.mlp_user_embedding(user_indices)
        mlp_item_embed = self.mlp_item_embedding(item_indices)
        mlp_input = torch.cat([mlp_user_embed, mlp_item_embed], dim=1)
        mlp_output = self.mlp_layers(mlp_input)
        
        # 融合GMF和MLP输出
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        predictions = self.fusion_layer(fusion_input)
        
        return predictions
    
    def get_embeddings(self, user_indices, item_indices):
        """获取用户和物品的嵌入表示（用于分析）"""
        with torch.no_grad():
            # GMF嵌入
            gmf_user_embed = self.gmf_user_embedding(user_indices)
            gmf_item_embed = self.gmf_item_embedding(item_indices)
            
            # MLP嵌入
            mlp_user_embed = self.mlp_user_embedding(user_indices)
            mlp_item_embed = self.mlp_item_embedding(item_indices)
            
            return {
                'gmf_user': gmf_user_embed,
                'gmf_item': gmf_item_embed,
                'mlp_user': mlp_user_embed,
                'mlp_item': mlp_item_embed
            }

# 演示完整的NeuMF模型
def demonstrate_neumf():
    """演示NeuMF模型的使用"""
    
    num_users, num_items = 1000, 500
    batch_size = 32
    
    # 创建模型
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        gmf_embedding_size=64,
        mlp_embedding_size=64,
        mlp_hidden_sizes=[128, 64, 32],
        dropout_rate=0.2
    )
    
    # 模拟输入
    user_indices = torch.randint(0, num_users, (batch_size,))
    item_indices = torch.randint(0, num_items, (batch_size,))
    
    # 前向传播
    with torch.no_grad():
        predictions = model(user_indices, item_indices)
        embeddings = model.get_embeddings(user_indices[:5], item_indices[:5])
    
    print(f"NeuMF模型输出形状: {predictions.shape}")
    print(f"预测值示例: {predictions[:5].flatten()}")
    print(f"模型总参数数: {sum(p.numel() for p in model.parameters())}")
    
    # 分析嵌入维度
    for key, embed in embeddings.items():
        print(f"{key} 嵌入形状: {embed.shape}")
    
    return model

neumf_model = demonstrate_neumf()
```

### 18.6.2. Model Implementation 模型实现

Let's implement a complete NeuMF model with training capabilities, including proper data handling and optimization:

让我们实现一个完整的具有训练能力的NeuMF模型，包括适当的数据处理和优化：

```python
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

class NeuMFTrainer:
    """NeuMF模型训练器"""
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        """
        初始化训练器
        
        参数:
        - model: NeuMF模型实例
        - learning_rate: 学习率
        - weight_decay: 权重衰减（正则化）
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)
        self.criterion = nn.MSELoss()  # 均方误差损失
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, device='cpu'):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (user_indices, item_indices, ratings) in enumerate(train_loader):
            # 移动数据到设备
            user_indices = user_indices.to(device)
            item_indices = item_indices.to(device) 
            ratings = ratings.float().to(device)
            
            # 前向传播
            predictions = self.model(user_indices, item_indices).squeeze()
            
            # 计算损失
            loss = self.criterion(predictions, ratings)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印进度
            if (batch_idx + 1) % 100 == 0:
                print(f'批次 {batch_idx + 1}, 损失: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader, device='cpu'):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for user_indices, item_indices, ratings in val_loader:
                # 移动数据到设备
                user_indices = user_indices.to(device)
                item_indices = item_indices.to(device)
                ratings = ratings.float().to(device)
                
                # 前向传播
                predictions = self.model(user_indices, item_indices).squeeze()
                
                # 计算损失
                loss = self.criterion(predictions, ratings)
                total_loss += loss.item()
                num_batches += 1
                
                # 收集预测和真实值
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # 计算额外指标
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        mae = mean_absolute_error(all_targets, all_predictions)
        
        return avg_loss, rmse, mae
    
    def fit(self, train_loader, val_loader, epochs=50, device='cpu', 
            early_stopping_patience=10):
        """
        训练模型
        
        参数:
        - train_loader: 训练数据加载器
        - val_loader: 验证数据加载器  
        - epochs: 训练轮数
        - device: 训练设备
        - early_stopping_patience: 早停耐心值
        """
        self.model.to(device)
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练，设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            # 训练
            train_loss = self.train_epoch(train_loader, device)
            
            # 验证
            val_loss, val_rmse, val_mae = self.validate(val_loader, device)
            
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}, RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_neumf_model.pt')
                print("保存最佳模型")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"早停触发，验证损失连续{patience_counter}轮未改善")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_neumf_model.pt'))
        print("训练完成，已加载最佳模型")
    
    def plot_training_history(self):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失', color='blue')
        plt.plot(self.val_losses, label='验证损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        # 损失差异
        plt.subplot(1, 2, 2)
        if len(self.train_losses) == len(self.val_losses):
            loss_diff = np.array(self.val_losses) - np.array(self.train_losses)
            plt.plot(loss_diff, color='green')
            plt.xlabel('Epoch')
            plt.ylabel('验证损失 - 训练损失')
            plt.title('过拟合监控')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# 推荐生成器
class NeuMFRecommender:
    """基于NeuMF的推荐生成器"""
    
    def __init__(self, model, user_encoder, item_encoder):
        """
        初始化推荐器
        
        参数:
        - model: 训练好的NeuMF模型
        - user_encoder: 用户ID编码器
        - item_encoder: 物品ID编码器
        """
        self.model = model
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        
    def recommend_for_user(self, user_id, num_recommendations=10, 
                          exclude_rated_items=None, device='cpu'):
        """
        为单个用户生成推荐
        
        参数:
        - user_id: 原始用户ID
        - num_recommendations: 推荐数量
        - exclude_rated_items: 要排除的已评分物品列表
        - device: 计算设备
        
        返回:
        - recommendations: 推荐列表 [(item_id, predicted_rating), ...]
        """
        self.model.eval()
        
        # 编码用户ID
        if user_id not in self.user_encoder:
            raise ValueError(f"用户 {user_id} 不在训练数据中")
        
        encoded_user_id = self.user_encoder[user_id]
        
        # 获取所有物品
        all_items = list(self.item_encoder.keys())
        if exclude_rated_items:
            all_items = [item for item in all_items if item not in exclude_rated_items]
        
        # 编码物品ID
        encoded_item_ids = [self.item_encoder[item] for item in all_items]
        
        # 创建输入张量
        user_tensor = torch.full((len(encoded_item_ids),), encoded_user_id, dtype=torch.long)
        item_tensor = torch.tensor(encoded_item_ids, dtype=torch.long)
        
        # 移动到设备
        user_tensor = user_tensor.to(device)
        item_tensor = item_tensor.to(device)
        
        # 预测评分
        with torch.no_grad():
            predictions = self.model(user_tensor, item_tensor).squeeze()
            predictions = predictions.cpu().numpy()
        
        # 创建推荐列表
        item_scores = list(zip(all_items, predictions))
        
        # 按预测评分排序
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return item_scores[:num_recommendations]
    
    def batch_recommend(self, user_ids, num_recommendations=10, device='cpu'):
        """
        为多个用户批量生成推荐
        
        参数:
        - user_ids: 用户ID列表
        - num_recommendations: 每个用户的推荐数量
        - device: 计算设备
        
        返回:
        - recommendations: {user_id: [(item_id, score), ...]}
        """
        recommendations = {}
        
        for user_id in user_ids:
            try:
                user_recs = self.recommend_for_user(
                    user_id, num_recommendations, device=device
                )
                recommendations[user_id] = user_recs
            except ValueError as e:
                print(f"跳过用户 {user_id}: {e}")
                continue
                
        return recommendations

# 演示完整的训练和推荐流程
def demonstrate_complete_neumf():
    """演示完整的NeuMF训练和推荐流程"""
    
    # 模拟数据生成
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成模拟评分数据
    num_users = 500
    num_items = 300
    num_ratings = 10000
    
    # 随机生成用户-物品-评分三元组
    user_ids = np.random.randint(0, num_users, num_ratings)
    item_ids = np.random.randint(0, num_items, num_ratings)
    ratings = np.random.uniform(1, 5, num_ratings)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids, 
        'rating': ratings
    })
    
    # 去重（一个用户对一个物品只能有一个评分）
    data = data.drop_duplicates(subset=['user_id', 'item_id'])
    
    print(f"数据集大小: {len(data)}")
    print(f"用户数: {data['user_id'].nunique()}")
    print(f"物品数: {data['item_id'].nunique()}")
    print(f"评分范围: [{data['rating'].min():.2f}, {data['rating'].max():.2f}]")
    
    return data

# 运行演示
demo_data = demonstrate_complete_neumf()
```

### 18.6.3. Customized Dataset with Negative Sampling 自定义数据集与负采样

For training ranking-based recommender systems, we need to create negative samples since we typically only have positive feedback (ratings or interactions). Negative sampling is like teaching the model what users DON'T like by showing examples of items they didn't interact with.

对于训练基于排序的推荐系统，我们需要创建负样本，因为我们通常只有正向反馈（评分或交互）。负采样就像通过展示用户没有交互的物品示例来教导模型用户不喜欢什么。

```python
import random
from collections import defaultdict

class RecommenderDataset(Dataset):
    """推荐系统数据集，支持负采样"""
    
    def __init__(self, interactions, num_negatives=4, user_pool=None, item_pool=None):
        """
        初始化数据集
        
        参数:
        - interactions: 交互数据，DataFrame格式，包含user_id, item_id, rating列
        - num_negatives: 每个正样本对应的负样本数量
        - user_pool: 用户池，如果为None则从interactions中获取
        - item_pool: 物品池，如果为None则从interactions中获取
        """
        self.interactions = interactions
        self.num_negatives = num_negatives
        
        # 用户和物品编码
        self.user_encoder = {user: idx for idx, user in enumerate(
            interactions['user_id'].unique())}
        self.item_encoder = {item: idx for idx, item in enumerate(
            interactions['item_id'].unique())}
        
        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item for item, idx in self.item_encoder.items()}
        
        # 构建用户-物品交互字典
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            self.user_items[user_id].add(item_id)
        
        # 所有物品集合（用于负采样）
        self.all_items = set(interactions['item_id'].unique())
        
        # 准备训练样本
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        """准备训练样本（正样本+负样本）"""
        samples = []
        
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = row['rating']
            
            # 编码用户和物品ID
            encoded_user = self.user_encoder[user_id]
            encoded_item = self.item_encoder[item_id]
            
            # 添加正样本
            samples.append((encoded_user, encoded_item, 1, rating))
            
            # 生成负样本
            user_positive_items = self.user_items[user_id]
            negative_items = list(self.all_items - user_positive_items)
            
            # 随机选择负样本
            if len(negative_items) >= self.num_negatives:
                selected_negatives = random.sample(negative_items, self.num_negatives)
            else:
                # 如果负样本不足，进行重复采样
                selected_negatives = random.choices(negative_items, k=self.num_negatives)
            
            for neg_item in selected_negatives:
                encoded_neg_item = self.item_encoder[neg_item]
                # 负样本的标签为0，评分设为1（最低分）
                samples.append((encoded_user, encoded_neg_item, 0, 1.0))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        user_idx, item_idx, label, rating = self.samples[idx]
        return torch.tensor(user_idx), torch.tensor(item_idx), torch.tensor(rating)
    
    def get_stats(self):
        """获取数据集统计信息"""
        positive_samples = sum(1 for sample in self.samples if sample[2] == 1)
        negative_samples = len(self.samples) - positive_samples
        
        return {
            'total_samples': len(self.samples),
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'negative_ratio': negative_samples / positive_samples,
            'num_users': len(self.user_encoder),
            'num_items': len(self.item_encoder)
        }

class PairwiseRankingDataset(Dataset):
    """成对排序数据集"""
    
    def __init__(self, interactions, num_negatives=1):
        """
        初始化成对排序数据集
        每个样本包含一个正物品和一个负物品的对比
        
        参数:
        - interactions: 交互数据
        - num_negatives: 每个正样本对应的负样本数量
        """
        self.interactions = interactions
        self.num_negatives = num_negatives
        
        # 编码器
        self.user_encoder = {user: idx for idx, user in enumerate(
            interactions['user_id'].unique())}
        self.item_encoder = {item: idx for idx, item in enumerate(
            interactions['item_id'].unique())}
        
        # 用户交互物品
        self.user_items = defaultdict(set)
        for _, row in interactions.iterrows():
            self.user_items[row['user_id']].add(row['item_id'])
        
        self.all_items = set(interactions['item_id'].unique())
        self.pairs = self._generate_pairs()
    
    def _generate_pairs(self):
        """生成正负物品对"""
        pairs = []
        
        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            pos_item = row['item_id']
            
            # 编码
            encoded_user = self.user_encoder[user_id]
            encoded_pos_item = self.item_encoder[pos_item]
            
            # 生成负物品
            user_positive_items = self.user_items[user_id]
            negative_items = list(self.all_items - user_positive_items)
            
            for _ in range(self.num_negatives):
                neg_item = random.choice(negative_items)
                encoded_neg_item = self.item_encoder[neg_item]
                
                pairs.append((encoded_user, encoded_pos_item, encoded_neg_item))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        user_idx, pos_item_idx, neg_item_idx = self.pairs[idx]
        return (torch.tensor(user_idx), 
                torch.tensor(pos_item_idx), 
                torch.tensor(neg_item_idx))

# 演示数据集使用
def demonstrate_negative_sampling():
    """演示负采样数据集的使用"""
    
    # 使用之前的demo_data
    print("=== 原始数据统计 ===")
    print(f"原始交互数: {len(demo_data)}")
    print(f"用户数: {demo_data['user_id'].nunique()}")
    print(f"物品数: {demo_data['item_id'].nunique()}")
    
    # 创建负采样数据集
    print("\n=== 创建负采样数据集 ===")
    dataset = RecommenderDataset(demo_data, num_negatives=4)
    stats = dataset.get_stats()
    
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 检查一个批次的数据
    print("\n=== 检查数据批次 ===")
    for batch_users, batch_items, batch_ratings in dataloader:
        print(f"批次大小: {len(batch_users)}")
        print(f"用户ID范围: [{batch_users.min()}, {batch_users.max()}]")
        print(f"物品ID范围: [{batch_items.min()}, {batch_items.max()}]")
        print(f"评分范围: [{batch_ratings.min():.2f}, {batch_ratings.max():.2f}]")
        
        # 检查正负样本比例
        unique_ratings, counts = torch.unique(batch_ratings, return_counts=True)
        for rating, count in zip(unique_ratings, counts):
            print(f"评分 {rating:.1f}: {count} 个样本")
        break
    
    # 创建成对排序数据集
    print("\n=== 创建成对排序数据集 ===")
    pairwise_dataset = PairwiseRankingDataset(demo_data, num_negatives=2)
    pairwise_loader = DataLoader(pairwise_dataset, batch_size=32, shuffle=True)
    
    # 检查成对数据
    for batch_users, batch_pos_items, batch_neg_items in pairwise_loader:
        print(f"成对数据批次大小: {len(batch_users)}")
        print(f"正物品ID范围: [{batch_pos_items.min()}, {batch_pos_items.max()}]")
        print(f"负物品ID范围: [{batch_neg_items.min()}, {batch_neg_items.max()}]")
        break
    
    return dataset, pairwise_dataset

# 运行演示
rec_dataset, pairwise_dataset = demonstrate_negative_sampling()
```

### 18.6.4. Evaluator 评估器

For ranking-based recommendation, we need specialized evaluation metrics that measure the quality of the ranked recommendation lists rather than individual rating predictions:

对于基于排序的推荐，我们需要专门的评估指标来衡量排序推荐列表的质量，而不是个别评分预测：

```python
class RankingEvaluator:
    """排序推荐评估器"""
    
    def __init__(self, k_values=[5, 10, 20]):
        """
        初始化评估器
        
        参数:
        - k_values: 要评估的K值列表（如Top-5, Top-10等）
        """
        self.k_values = k_values
    
    def hit_ratio_at_k(self, recommended_items, relevant_items, k):
        """
        计算Hit Ratio@K
        衡量推荐列表中是否包含用户真正喜欢的物品
        
        Think of this as: "在前K个推荐中，是否至少有一个是用户真正喜欢的？"
        """
        top_k_items = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        return 1.0 if len(top_k_items.intersection(relevant_set)) > 0 else 0.0
    
    def precision_at_k(self, recommended_items, relevant_items, k):
        """
        计算Precision@K
        精确率：推荐的前K个物品中有多少是相关的
        
        Think of this as: "前K个推荐中，有百分之多少是好的推荐？"
        """
        top_k_items = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        return len(top_k_items.intersection(relevant_set)) / k
    
    def recall_at_k(self, recommended_items, relevant_items, k):
        """
        计算Recall@K
        召回率：用户真正喜欢的物品中有多少被推荐了
        
        Think of this as: "用户真正喜欢的物品中，有百分之多少被推荐了？"
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_items = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        return len(top_k_items.intersection(relevant_set)) / len(relevant_set)
    
    def ndcg_at_k(self, recommended_items, relevant_items, k, rating_threshold=4.0):
        """
        计算NDCG@K (Normalized Discounted Cumulative Gain)
        考虑推荐位置的重要性，排在前面的推荐更重要
        
        Think of this as: "不仅考虑推荐的准确性，还考虑好推荐是否排在前面"
        """
        # DCG计算
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            if item in relevant_items:
                # 相关物品的增益为1，位置越靠前增益越大
                dcg += 1.0 / np.log2(i + 2)  # +2因为log2(1)=0
        
        # IDCG计算（理想情况下的DCG）
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mean_reciprocal_rank(self, recommended_items, relevant_items):
        """
        计算MRR (Mean Reciprocal Rank)
        关注第一个相关物品的排名位置
        
        Think of this as: "第一个好推荐出现在第几位？位置越靠前越好"
        """
        for i, item in enumerate(recommended_items):
            if item in relevant_items:
                return 1.0 / (i + 1)  # +1因为排名从1开始
        return 0.0
    
    def evaluate_recommendations(self, recommendations, ground_truth, 
                               rating_threshold=4.0):
        """
        评估推荐结果
        
        参数:
        - recommendations: {user_id: [item_id1, item_id2, ...]}
        - ground_truth: {user_id: [(item_id, rating), ...]}
        - rating_threshold: 相关物品的评分阈值
        
        返回:
        - metrics: 各种评估指标的字典
        """
        all_metrics = {f'HR@{k}': [] for k in self.k_values}
        all_metrics.update({f'Precision@{k}': [] for k in self.k_values})
        all_metrics.update({f'Recall@{k}': [] for k in self.k_values})
        all_metrics.update({f'NDCG@{k}': [] for k in self.k_values})
        all_metrics['MRR'] = []
        
        for user_id in recommendations:
            if user_id not in ground_truth:
                continue
                
            recommended_items = recommendations[user_id]
            
            # 获取相关物品（高评分物品）
            user_ratings = ground_truth[user_id]
            relevant_items = [item for item, rating in user_ratings 
                            if rating >= rating_threshold]
            
            if len(relevant_items) == 0:
                continue
            
            # 计算各种指标
            for k in self.k_values:
                hr = self.hit_ratio_at_k(recommended_items, relevant_items, k)
                precision = self.precision_at_k(recommended_items, relevant_items, k)
                recall = self.recall_at_k(recommended_items, relevant_items, k)
                ndcg = self.ndcg_at_k(recommended_items, relevant_items, k)
                
                all_metrics[f'HR@{k}'].append(hr)
                all_metrics[f'Precision@{k}'].append(precision)
                all_metrics[f'Recall@{k}'].append(recall)
                all_metrics[f'NDCG@{k}'].append(ndcg)
            
            # 计算MRR
            mrr = self.mean_reciprocal_rank(recommended_items, relevant_items)
            all_metrics['MRR'].append(mrr)
        
        # 计算平均值
        final_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:  # 确保列表不为空
                final_metrics[metric_name] = np.mean(values)
            else:
                final_metrics[metric_name] = 0.0
        
        return final_metrics
    
    def print_metrics(self, metrics):
        """打印评估指标"""
        print("=== 排序推荐评估结果 ===")
        
        # 按指标类型分组打印
        metric_groups = {
            'Hit Ratio': [k for k in metrics.keys() if k.startswith('HR@')],
            'Precision': [k for k in metrics.keys() if k.startswith('Precision@')],
            'Recall': [k for k in metrics.keys() if k.startswith('Recall@')],
            'NDCG': [k for k in metrics.keys() if k.startswith('NDCG@')],
            'Other': ['MRR']
        }
        
        for group_name, metric_names in metric_groups.items():
            if metric_names and any(name in metrics for name in metric_names):
                print(f"\n{group_name}:")
                for metric_name in metric_names:
                    if metric_name in metrics:
                        print(f"  {metric_name}: {metrics[metric_name]:.4f}")

# 演示评估器使用
def demonstrate_ranking_evaluation():
    """演示排序评估的使用"""
    
    # 模拟推荐结果和真实数据
    np.random.seed(42)
    
    # 模拟推荐结果
    recommendations = {}
    ground_truth = {}
    
    num_users = 100
    num_items = 500
    
    for user_id in range(num_users):
        # 每个用户推荐20个物品
        recommended_items = np.random.choice(num_items, size=20, replace=False).tolist()
        recommendations[user_id] = recommended_items
        
        # 模拟真实评分（10-30个物品）
        num_rated_items = np.random.randint(10, 31)
        rated_items = np.random.choice(num_items, size=num_rated_items, replace=False)
        ratings = np.random.uniform(1, 5, size=num_rated_items)
        
        ground_truth[user_id] = list(zip(rated_items, ratings))
    
    # 创建评估器
    evaluator = RankingEvaluator(k_values=[5, 10, 20])
    
    # 评估推荐结果
    metrics = evaluator.evaluate_recommendations(recommendations, ground_truth)
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    return evaluator, metrics

# 运行演示
ranking_evaluator, demo_metrics = demonstrate_ranking_evaluation()
```

### 18.6.5. Training and Evaluating the Model 训练和评估模型

Now let's implement the complete training and evaluation pipeline for NeuMF with ranking-based objectives:

现在让我们实现NeuMF基于排序目标的完整训练和评估管道：

```python
class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking (BPR) 损失函数"""
    
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_scores, neg_scores):
        """
        计算BPR损失
        
        参数:
        - pos_scores: 正样本预测分数 [batch_size]
        - neg_scores: 负样本预测分数 [batch_size] 
        
        返回:
        - loss: BPR损失值
        """
        # BPR损失 = -log(sigmoid(pos_score - neg_score))
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff)).mean()
        return loss

class NeuMFRankingTrainer:
    """NeuMF排序训练器"""
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        """
        初始化排序训练器
        
        参数:
        - model: NeuMF模型
        - learning_rate: 学习率
        - weight_decay: 权重衰减
        """
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), 
                                  lr=learning_rate, 
                                  weight_decay=weight_decay)
        
        # 使用BPR损失进行排序学习
        self.bpr_loss = BPRLoss()
        
        # 训练历史
        self.train_losses = []
        self.val_metrics = []
    
    def train_epoch_pairwise(self, pairwise_loader, device='cpu'):
        """使用成对数据训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_users, batch_pos_items, batch_neg_items in pairwise_loader:
            # 移动数据到设备
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            batch_neg_items = batch_neg_items.to(device)
            
            # 前向传播
            pos_scores = self.model(batch_users, batch_pos_items).squeeze()
            neg_scores = self.model(batch_users, batch_neg_items).squeeze()
            
            # 计算BPR损失
            loss = self.bpr_loss(pos_scores, neg_scores)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate_ranking(self, test_data, evaluator, num_recommendations=20, device='cpu'):
        """评估排序性能"""
        self.model.eval()
        
        # 准备推荐和真实数据
        recommendations = {}
        ground_truth = {}
        
        # 获取唯一用户
        unique_users = test_data['user_id'].unique()
        
        with torch.no_grad():
            for user_id in unique_users:
                # 获取用户的真实评分
                user_ratings = test_data[test_data['user_id'] == user_id]
                ground_truth[user_id] = [(row['item_id'], row['rating']) 
                                       for _, row in user_ratings.iterrows()]
                
                # 为该用户生成推荐
                try:
                    # 获取所有物品
                    all_items = test_data['item_id'].unique()
                    
                    # 编码用户ID
                    if user_id < len(rec_dataset.user_encoder):
                        encoded_user = user_id
                    else:
                        continue
                    
                    # 创建用户-物品对
                    user_tensor = torch.full((len(all_items),), encoded_user, dtype=torch.long)
                    item_tensor = torch.tensor(all_items, dtype=torch.long)
                    
                    # 移动到设备
                    user_tensor = user_tensor.to(device)
                    item_tensor = item_tensor.to(device)
                    
                    # 预测分数
                    scores = self.model(user_tensor, item_tensor).squeeze()
                    
                    # 排序并获取Top-K推荐
                    _, top_indices = torch.topk(scores, k=min(num_recommendations, len(all_items)))
                    recommended_items = all_items[top_indices.cpu().numpy()].tolist()
                    
                    recommendations[user_id] = recommended_items
                    
                except Exception as e:
                    print(f"用户 {user_id} 推荐生成失败: {e}")
                    continue
        
        # 使用评估器计算指标
        if recommendations and ground_truth:
            metrics = evaluator.evaluate_recommendations(recommendations, ground_truth)
            self.val_metrics.append(metrics)
            return metrics
        else:
            return {}
    
    def fit_ranking(self, train_loader, test_data, evaluator, 
                   epochs=50, device='cpu', eval_every=5):
        """
        训练排序模型
        
        参数:
        - train_loader: 成对训练数据加载器
        - test_data: 测试数据（用于评估）
        - evaluator: 排序评估器
        - epochs: 训练轮数
        - device: 训练设备
        - eval_every: 每多少轮评估一次
        """
        self.model.to(device)
        
        print(f"开始排序训练，设备: {device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        best_ndcg = 0.0
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            # 训练
            train_loss = self.train_epoch_pairwise(train_loader, device)
            print(f"训练损失: {train_loss:.4f}")
            
            # 定期评估
            if (epoch + 1) % eval_every == 0:
                print("评估中...")
                metrics = self.evaluate_ranking(test_data, evaluator, device=device)
                
                if metrics and 'NDCG@10' in metrics:
                    current_ndcg = metrics['NDCG@10']
                    print(f"当前 NDCG@10: {current_ndcg:.4f}")
                    
                    # 保存最佳模型
                    if current_ndcg > best_ndcg:
                        best_ndcg = current_ndcg
                        torch.save(self.model.state_dict(), 'best_neumf_ranking_model.pt')
                        print(f"保存最佳模型 (NDCG@10: {best_ndcg:.4f})")
                    
                    # 打印部分关键指标
                    key_metrics = ['HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'MRR']
                    for metric in key_metrics:
                        if metric in metrics:
                            print(f"{metric}: {metrics[metric]:.4f}")
        
        # 加载最佳模型
        if best_ndcg > 0:
            self.model.load_state_dict(torch.load('best_neumf_ranking_model.pt'))
            print(f"训练完成，已加载最佳模型 (NDCG@10: {best_ndcg:.4f})")
    
    def plot_training_progress(self):
        """绘制训练进度"""
        import matplotlib.pyplot as plt
        
        if not self.train_losses or not self.val_metrics:
            print("没有足够的训练历史数据来绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失
        axes[0, 0].plot(self.train_losses, color='blue')
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('BPR Loss')
        axes[0, 0].grid(True)
        
        # NDCG@10
        if self.val_metrics:
            ndcg_values = [m.get('NDCG@10', 0) for m in self.val_metrics]
            axes[0, 1].plot(ndcg_values, color='red', marker='o')
            axes[0, 1].set_title('NDCG@10')
            axes[0, 1].set_xlabel('Evaluation Step')
            axes[0, 1].set_ylabel('NDCG@10')
            axes[0, 1].grid(True)
        
        # Hit Ratio对比
        if self.val_metrics:
            hr5_values = [m.get('HR@5', 0) for m in self.val_metrics]
            hr10_values = [m.get('HR@10', 0) for m in self.val_metrics]
            
            axes[1, 0].plot(hr5_values, label='HR@5', marker='s')
            axes[1, 0].plot(hr10_values, label='HR@10', marker='^')
            axes[1, 0].set_title('Hit Ratio')
            axes[1, 0].set_xlabel('Evaluation Step')
            axes[1, 0].set_ylabel('Hit Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # MRR
        if self.val_metrics:
            mrr_values = [m.get('MRR', 0) for m in self.val_metrics]
            axes[1, 1].plot(mrr_values, color='green', marker='d')
            axes[1, 1].set_title('Mean Reciprocal Rank')
            axes[1, 1].set_xlabel('Evaluation Step')
            axes[1, 1].set_ylabel('MRR')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

# 演示完整的NeuMF排序训练流程
def demonstrate_neumf_ranking_training():
    """演示NeuMF排序训练的完整流程"""
    
    # 分割数据
    from sklearn.model_selection import train_test_split
    
    train_data, test_data = train_test_split(demo_data, test_size=0.2, random_state=42)
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 创建成对排序数据集
    pairwise_train_dataset = PairwiseRankingDataset(train_data, num_negatives=3)
    pairwise_train_loader = DataLoader(pairwise_train_dataset, batch_size=256, shuffle=True)
    
    # 创建模型
    num_users = demo_data['user_id'].nunique()
    num_items = demo_data['item_id'].nunique()
    
    model = NeuMF(
        num_users=num_users,
        num_items=num_items,
        gmf_embedding_size=32,
        mlp_embedding_size=32,
        mlp_hidden_sizes=[64, 32, 16],
        dropout_rate=0.2
    )
    
    # 创建训练器和评估器
    trainer = NeuMFRankingTrainer(model, learning_rate=0.001, weight_decay=1e-5)
    evaluator = RankingEvaluator(k_values=[5, 10, 20])
    
    # 训练模型
    print("\n开始训练...")
    trainer.fit_ranking(
        train_loader=pairwise_train_loader,
        test_data=test_data,
        evaluator=evaluator,
        epochs=20,  # 为了演示使用较少的轮数
        device='cpu',
        eval_every=5
    )
    
    # 最终评估
    print("\n=== 最终评估 ===")
    final_metrics = trainer.evaluate_ranking(test_data, evaluator)
    evaluator.print_metrics(final_metrics)
    
    # 绘制训练进度
    trainer.plot_training_progress()
    
    return trainer, model, final_metrics

# 运行演示（注释掉以避免长时间训练）
# trainer, trained_model, final_metrics = demonstrate_neumf_ranking_training()
print("演示代码已准备就绪。要运行完整训练，请取消注释最后一行。")
```

### 18.6.6. Summary 总结

Neural Collaborative Filtering (NCF) and specifically the NeuMF model represent a significant advancement in recommendation systems by combining the strengths of traditional matrix factorization with the expressiveness of deep neural networks. The key innovations and benefits include:

神经协同过滤（NCF），特别是NeuMF模型，通过结合传统矩阵分解的优势和深度神经网络的表达能力，代表了推荐系统的重大进步。主要创新和优势包括：

**Key Advantages 主要优势:**

1. **Non-linear Modeling 非线性建模**: Unlike traditional matrix factorization which assumes linear relationships, NCF can capture complex, non-linear interactions between users and items through deep neural networks.
   与假设线性关系的传统矩阵分解不同，NCF可以通过深度神经网络捕获用户和物品之间复杂的非线性交互。

2. **Flexible Architecture 灵活架构**: The fusion of GMF and MLP components allows the model to learn both linear and non-linear patterns simultaneously.
   GMF和MLP组件的融合允许模型同时学习线性和非线性模式。

3. **Ranking-based Learning 基于排序的学习**: By using pairwise ranking losses like BPR, the model directly optimizes for recommendation ranking rather than rating prediction.
   通过使用BPR等成对排序损失，模型直接优化推荐排序而不是评分预测。

4. **Scalable Training 可扩展训练**: The neural architecture allows for efficient mini-batch training and GPU acceleration.
   神经架构允许高效的小批量训练和GPU加速。

**Applications 应用场景:**
- E-commerce product recommendations 电商产品推荐
- Movie and music recommendations 电影和音乐推荐  
- Social media content curation 社交媒体内容策划
- News article recommendations 新闻文章推荐

**Limitations 局限性:**
- Requires more computational resources than traditional methods 比传统方法需要更多计算资源
- Black-box nature makes it less interpretable 黑盒性质使其可解释性较差
- Still suffers from cold-start problems for new users/items 新用户/物品仍存在冷启动问题

### 18.6.7. Exercises 练习

1. **Implement** different fusion strategies for combining GMF and MLP outputs beyond simple concatenation.
   **实现**除了简单连接之外的不同融合策略来组合GMF和MLP输出。

2. **Experiment** with different negative sampling strategies and analyze their impact on model performance.
   **实验**不同的负采样策略并分析它们对模型性能的影响。

3. **Add** regularization techniques like dropout and batch normalization to improve model generalization.
   **添加**dropout和批归一化等正则化技术来改善模型泛化。

4. **Compare** NCF with traditional collaborative filtering methods on the same dataset.
   **比较**NCF与传统协同过滤方法在同一数据集上的表现。

5. **Implement** a content-based component to address the cold-start problem.
   **实现**基于内容的组件来解决冷启动问题。

---

## 18.7. Sequence-Aware Recommender Systems 序列感知推荐系统

Traditional collaborative filtering treats user preferences as static, but in reality, user interests evolve over time. Sequence-aware recommender systems capture the temporal dynamics of user behavior by modeling the sequential patterns in user interactions.

传统协同过滤将用户偏好视为静态的，但实际上，用户兴趣会随时间演变。序列感知推荐系统通过建模用户交互中的序列模式来捕获用户行为的时间动态。

Think of it like this: if you recently watched several Marvel movies, you're more likely to be interested in the next Marvel release than someone who watched them years ago. The sequence and timing of your actions matter.

可以这样想：如果你最近看了几部漫威电影，你比几年前看过它们的人更可能对下一部漫威电影感兴趣。你行动的序列和时机很重要。

### 18.7.1. Model Architectures 模型架构

#### RNN-based Sequence Models 基于RNN的序列模型

```python
class GRU4Rec(nn.Module):
    """GRU-based Session Recommendation Model"""
    
    def __init__(self, num_items, embedding_size=100, hidden_size=100):
        super(GRU4Rec, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_size, padding_idx=0)
        self.gru = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_items)
    
    def forward(self, session_items):
        embedded = self.item_embedding(session_items)
        gru_output, hidden = self.gru(embedded)
        output = self.output_layer(gru_output)
        return output, hidden
```

#### Transformer-based Models Transformer基础模型

```python
class SASRec(nn.Module):
    """Self-Attention Sequential Recommendation"""
    
    def __init__(self, num_items, max_len=200, embedding_size=64, num_blocks=2):
        super(SASRec, self).__init__()
        self.item_embedding = nn.Embedding(num_items + 1, embedding_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embedding_size, nhead=2, batch_first=True)
            for _ in range(num_blocks)
        ])
        
        self.output_layer = nn.Linear(embedding_size, num_items)
    
    def forward(self, session_items):
        seq_len = session_items.size(1)
        positions = torch.arange(seq_len, device=session_items.device).unsqueeze(0)
        
        item_emb = self.item_embedding(session_items)
        pos_emb = self.position_embedding(positions)
        
        hidden_states = item_emb + pos_emb
        
        for transformer in self.transformer_blocks:
            hidden_states = transformer(hidden_states)
        
        output = self.output_layer(hidden_states)
        return output
```

### 18.7.2. Summary 总结

Sequence-aware recommender systems capture temporal dynamics in user behavior, enabling more accurate and timely recommendations by understanding how user preferences evolve over time.

序列感知推荐系统捕获用户行为中的时间动态，通过理解用户偏好如何随时间演变来实现更准确和及时的推荐。

### 18.7.3. Exercises 练习

1. **Implement** attention mechanisms to focus on important items in the sequence.
   **实现**注意力机制来关注序列中的重要物品。

2. **Compare** RNN vs Transformer architectures for different sequence lengths.
   **比较**RNN与Transformer架构在不同序列长度下的表现。

---

## 18.8. Feature-Rich Recommender Systems 特征丰富的推荐系统

Real-world recommendation systems often need to incorporate various types of features beyond user-item interactions, such as user demographics, item content features, and contextual information.

现实世界的推荐系统通常需要融入除用户-物品交互之外的各种类型特征，如用户人口统计学特征、物品内容特征和上下文信息。

### 18.8.1. An Online Advertising Dataset 在线广告数据集

Online advertising provides a rich example of feature-rich recommendation, where we need to predict click-through rates based on multiple features:

在线广告提供了特征丰富推荐的丰富例子，我们需要基于多个特征预测点击率：

```python
class CTRDataset:
    """Click-Through Rate Dataset for Online Advertising"""
    
    def __init__(self):
        self.user_features = ['age', 'gender', 'occupation', 'income_level']
        self.ad_features = ['category', 'brand', 'price', 'position']
        self.context_features = ['time_of_day', 'device', 'location']
    
    def create_sample_data(self, num_samples=10000):
        """创建示例广告点击数据"""
        import random
        
        data = []
        for _ in range(num_samples):
            sample = {
                # User features
                'age': random.randint(18, 65),
                'gender': random.choice(['M', 'F']),
                'occupation': random.choice(['student', 'engineer', 'teacher', 'other']),
                'income_level': random.choice(['low', 'medium', 'high']),
                
                # Ad features  
                'category': random.choice(['electronics', 'fashion', 'food', 'travel']),
                'brand': random.choice(['brand_a', 'brand_b', 'brand_c']),
                'price': random.uniform(10, 1000),
                'position': random.randint(1, 10),
                
                # Context features
                'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night']),
                'device': random.choice(['mobile', 'desktop', 'tablet']),
                'location': random.choice(['home', 'work', 'other']),
                
                # Target (click or not)
                'clicked': random.choice([0, 1])
            }
            data.append(sample)
        
        return pd.DataFrame(data)
```

### 18.8.2. Dataset Wrapper 数据集包装器

```python
class FeatureRichDataset(Dataset):
    """Multi-feature dataset for recommendation"""
    
    def __init__(self, data):
        self.data = data
        self._encode_features()
    
    def _encode_features(self):
        """编码分类特征"""
        categorical_cols = ['gender', 'occupation', 'income_level', 'category', 
                          'brand', 'time_of_day', 'device', 'location']
        
        self.encoders = {}
        for col in categorical_cols:
            self.encoders[col] = {val: idx for idx, val in enumerate(self.data[col].unique())}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Encode categorical features
        features = []
        for col in ['gender', 'occupation', 'income_level']:
            features.append(self.encoders[col][row[col]])
        
        # Add numerical features
        features.extend([row['age'], row['price'], row['position']])
        
        return torch.tensor(features, dtype=torch.float), torch.tensor(row['clicked'], dtype=torch.float)
```

### 18.8.3. Summary 总结

Feature-rich recommender systems leverage multiple types of information to make more accurate and contextually relevant recommendations.

特征丰富的推荐系统利用多种类型的信息来做出更准确和上下文相关的推荐。

### 18.8.4. Exercises 练习

1. **Design** a feature engineering pipeline for e-commerce recommendations.
   **设计**电商推荐的特征工程管道。

2. **Implement** feature importance analysis to understand which features matter most.
   **实现**特征重要性分析来理解哪些特征最重要。

---

## 18.9. Factorization Machines 因子分解机

Factorization Machines (FM) are a powerful model class that can efficiently handle high-dimensional sparse features by modeling pairwise feature interactions.

因子分解机（FM）是一类强大的模型，能够通过建模成对特征交互来高效处理高维稀疏特征。

### 18.9.1. 2-Way Factorization Machines 二阶因子分解机

The FM model equation is: FM模型方程为：

ŷ = w₀ + Σᵢwᵢxᵢ + Σᵢ<ⱼ⟨vᵢ,vⱼ⟩xᵢxⱼ

```python
class FactorizationMachine(nn.Module):
    """二阶因子分解机模型"""
    
    def __init__(self, num_features, embedding_dim=10):
        super(FactorizationMachine, self).__init__()
        
        # 全局偏置
        self.w0 = nn.Parameter(torch.zeros(1))
        
        # 一阶权重
        self.w = nn.Parameter(torch.zeros(num_features))
        
        # 二阶交互嵌入
        self.v = nn.Parameter(torch.randn(num_features, embedding_dim) * 0.01)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 特征向量 [batch_size, num_features]
        
        返回:
        - output: 预测值 [batch_size]
        """
        # 线性项
        linear = self.w0 + torch.sum(self.w * x, dim=1)
        
        # 交互项（使用高效的O(nk)计算）
        sum_square = torch.pow(torch.mm(x, self.v), 2)
        square_sum = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2))
        
        interactions = 0.5 * torch.sum(sum_square - square_sum, dim=1)
        
        return linear + interactions
```

### 18.9.2. An Efficient Optimization Criterion 高效优化准则

For binary classification, we use logistic loss: 对于二分类，我们使用逻辑损失：

```python
def fm_loss(predictions, targets):
    """FM二分类损失函数"""
    return F.binary_cross_entropy_with_logits(predictions, targets)
```

### 18.9.3. Model Implementation 模型实现

```python
class FMTrainer:
    """因子分解机训练器"""
    
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for features, targets in train_loader:
            predictions = self.model(features)
            loss = fm_loss(predictions, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
```

### 18.9.4. Summary 总结

Factorization Machines efficiently model feature interactions with linear computational complexity, making them suitable for high-dimensional sparse data.

因子分解机以线性计算复杂度高效建模特征交互，使其适用于高维稀疏数据。

### 18.9.5. Exercises 练习

1. **Implement** higher-order factorization machines (3rd order and beyond).
   **实现**高阶因子分解机（三阶及以上）。

2. **Compare** FM with logistic regression on sparse datasets.
   **比较**FM与逻辑回归在稀疏数据集上的表现。

---

## 18.10. Deep Factorization Machines 深度因子分解机

Deep Factorization Machines (DeepFM) combine the strengths of factorization machines and deep neural networks to model both low-order and high-order feature interactions.

深度因子分解机（DeepFM）结合因子分解机和深度神经网络的优势，建模低阶和高阶特征交互。

### 18.10.1. Model Architectures 模型架构

DeepFM consists of FM and DNN components that share the same input: DeepFM由共享相同输入的FM和DNN组件组成：

```python
class DeepFM(nn.Module):
    """深度因子分解机模型"""
    
    def __init__(self, num_features, embedding_dim=10, hidden_dims=[128, 64]):
        super(DeepFM, self).__init__()
        
        # FM组件
        self.fm = FactorizationMachine(num_features, embedding_dim)
        
        # DNN组件
        dnn_layers = []
        input_dim = num_features * embedding_dim
        
        for hidden_dim in hidden_dims:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        dnn_layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*dnn_layers)
        
        # 嵌入层（FM和DNN共享）
        self.embedding = nn.Embedding(num_features, embedding_dim)
    
    def forward(self, sparse_features, dense_features):
        """
        前向传播
        
        参数:
        - sparse_features: 稀疏特征索引
        - dense_features: 稠密特征值
        
        返回:
        - output: 预测值
        """
        # FM组件
        fm_input = self._create_fm_input(sparse_features, dense_features)
        fm_output = self.fm(fm_input)
        
        # DNN组件
        embedded = self.embedding(sparse_features)  # [batch_size, num_sparse, embedding_dim]
        dnn_input = embedded.view(embedded.size(0), -1)  # 展平
        dnn_input = torch.cat([dnn_input, dense_features], dim=1)  # 加入稠密特征
        dnn_output = self.dnn(dnn_input)
        
        # 组合输出
        return fm_output + dnn_output.squeeze()
    
    def _create_fm_input(self, sparse_features, dense_features):
        """为FM组件创建输入"""
        # 简化实现：将稀疏特征转换为one-hot
        batch_size = sparse_features.size(0)
        fm_input = torch.zeros(batch_size, self.fm.w.size(0))
        
        # 设置对应位置为1（one-hot编码）
        for i, features in enumerate(sparse_features):
            fm_input[i, features] = 1.0
        
        return fm_input
```

### 18.10.2. Implementation of DeepFM DeepFM实现

```python
class DeepFMTrainer:
    """DeepFM训练器"""
    
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for sparse_features, dense_features, targets in train_loader:
            predictions = self.model(sparse_features, dense_features)
            loss = self.criterion(predictions, targets.float())
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for sparse_features, dense_features, batch_targets in test_loader:
                batch_predictions = torch.sigmoid(self.model(sparse_features, dense_features))
                predictions.extend(batch_predictions.cpu().numpy())
                targets.extend(batch_targets.cpu().numpy())
        
        # 计算AUC
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(targets, predictions)
        return auc
```

### 18.10.3. Training and Evaluating the Model 训练和评估模型

```python
def train_deepfm():
    """训练DeepFM模型的完整示例"""
    
    # 创建模型
    model = DeepFM(num_features=1000, embedding_dim=16, hidden_dims=[128, 64, 32])
    trainer = DeepFMTrainer(model, learning_rate=0.001)
    
    # 训练循环
    for epoch in range(50):
        train_loss = trainer.train_epoch(train_loader)
        
        if epoch % 5 == 0:
            test_auc = trainer.evaluate(test_loader)
            print(f'Epoch {epoch}, Loss: {train_loss:.4f}, AUC: {test_auc:.4f}')
    
    return model
```

### 18.10.4. Summary 总结

DeepFM effectively combines the advantages of FM for modeling low-order interactions and deep networks for high-order interactions, making it powerful for click-through rate prediction tasks.

DeepFM有效结合了FM建模低阶交互和深度网络建模高阶交互的优势，使其在点击率预测任务中非常强大。

### 18.10.5. Exercises 练习

1. **Implement** other deep FM variants like xDeepFM or DCN.
   **实现**其他深度FM变体，如xDeepFM或DCN。

2. **Compare** DeepFM performance with other CTR models on real datasets.
   **比较**DeepFM与其他CTR模型在真实数据集上的性能。

---

## 总结 Final Summary

This chapter covered the essential concepts and techniques in recommender systems, from traditional collaborative filtering to modern neural approaches. Key takeaways include:

本章涵盖了推荐系统中的基本概念和技术，从传统协同过滤到现代神经方法。主要要点包括：

1. **Collaborative Filtering 协同过滤**: Foundation of recommendation systems
2. **Matrix Factorization 矩阵分解**: Efficient latent factor modeling  
3. **Neural Collaborative Filtering 神经协同过滤**: Non-linear interaction modeling
4. **Sequential Recommendation 序列推荐**: Temporal dynamics modeling
5. **Feature-Rich Systems 特征丰富系统**: Multi-modal information integration
6. **Factorization Machines 因子分解机**: Efficient sparse feature handling
7. **Deep Factorization Machines 深度因子分解机**: Combined shallow and deep modeling

Each approach has its strengths and is suitable for different scenarios and data characteristics.

每种方法都有其优势，适用于不同的场景和数据特征。

--- 