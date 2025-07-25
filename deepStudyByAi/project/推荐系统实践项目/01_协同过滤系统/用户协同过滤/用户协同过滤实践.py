# -*- coding: utf-8 -*-
"""
用户协同过滤推荐系统实践
User-based Collaborative Filtering Recommendation System Practice
使用PyTorch实现
Implemented with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 步骤1：生成模拟数据
# Step 1: Generate simulated data
print("Step 1: Generating simulated data")

# 创建100个用户和50个物品的评分矩阵
# Create rating matrix for 100 users and 50 items
num_users = 100
num_items = 50
num_ratings = 1000  # 1000条评分记录

# 生成随机的用户-物品评分对
# Generate random user-item rating pairs
np.random.seed(42)
user_ids = np.random.randint(0, num_users, num_ratings)
item_ids = np.random.randint(0, num_items, num_ratings)
ratings = np.random.randint(1, 6, num_ratings)  # 评分从1到5

# 创建数据框
# Create dataframe
df = pd.DataFrame({
    'user_id': user_ids,
    'item_id': item_ids,
    'rating': ratings
})

print(f"Generated {len(df)} ratings from {num_users} users on {num_items} items")
print(f"Rating distribution:\n{df['rating'].value_counts().sort_index()}")

# 步骤2：数据预处理
# Step 2: Data preprocessing
print("\nStep 2: Data preprocessing")

# 将数据集分为训练集和测试集
# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Train set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# 创建用户-物品评分矩阵
# Create user-item rating matrix
train_matrix = np.zeros((num_users, num_items))

for _, row in train_df.iterrows():
    train_matrix[int(row['user_id']), int(row['item_id'])] = row['rating']

# 步骤3：实现余弦相似度计算
# Step 3: Implement cosine similarity calculation
print("\nStep 3: Implementing cosine similarity")

def cosine_similarity(matrix):
    """
    计算用户之间的余弦相似度
    Calculate cosine similarity between users
    """
    # 计算用户平均评分
    # Calculate user average ratings
    user_means = np.mean(matrix, axis=1, keepdims=True)
    
    # 对每个用户减去其平均评分（中心化）
    # Subtract user mean from each user (centering)
    matrix_centered = matrix - user_means
    
    # 计算相似度矩阵
    # Calculate similarity matrix
    similarity = np.dot(matrix_centered, matrix_centered.T)
    
    # 计算范数
    # Calculate norms
    norms = np.sqrt(np.diag(similarity))
    
    # 避免除以零
    # Avoid division by zero
    norms[norms == 0] = 1e-8
    
    # 归一化得到余弦相似度
    # Normalize to get cosine similarity
    similarity = similarity / norms[:, np.newaxis]
    similarity = similarity / norms[np.newaxis, :]
    
    return similarity

# 计算用户相似度矩阵
# Calculate user similarity matrix
user_similarity = cosine_similarity(train_matrix)
print(f"User similarity matrix shape: {user_similarity.shape}")
print(f"Similarity range: [{user_similarity.min():.3f}, {user_similarity.max():.3f}]")

# 步骤4：实现评分预测
# Step 4: Implement rating prediction
def predict_rating(user_id, item_id, train_matrix, user_similarity, k=20):
    """
    预测用户对物品的评分
    Predict rating for user on item
    
    Args:
        user_id: 用户ID
        item_id: 物品ID
        train_matrix: 训练评分矩阵
        user_similarity: 用户相似度矩阵
        k: 考虑的最近邻数量
    
    Returns:
        预测评分
    """
    # 找到已评分该物品的用户
    # Find users who have rated this item
    item_ratings = train_matrix[:, item_id]
    users_who_rated = np.where(item_ratings > 0)[0]
    
    if len(users_who_rated) == 0:
        # 如果没有用户评分过，返回全局平均
        # If no user has rated, return global average
        return np.mean(train_matrix[train_matrix > 0])
    
    # 获取这些用户的相似度
    # Get similarity with these users
    similarities = user_similarity[user_id, users_who_rated]
    
    # 找到最相似的k个用户
    # Find k most similar users
    if len(similarities) <= k:
        top_k_idx = np.argsort(similarities)[::-1]
    else:
        top_k_idx = np.argsort(similarities)[::-1][:k]
    
    # 获取他们的ID和相似度
    # Get their IDs and similarities
    neighbor_users = users_who_rated[top_k_idx]
    neighbor_similarities = similarities[top_k_idx]
    
    # 如果所有相似度都为0，返回用户平均分
    # If all similarities are 0, return user average
    if np.sum(np.abs(neighbor_similarities)) == 0:
        user_ratings = train_matrix[user_id]
        if np.any(user_ratings > 0):
            return np.mean(user_ratings[user_ratings > 0])
        else:
            return np.mean(train_matrix[train_matrix > 0])
    
    # 计算加权平均评分
    # Calculate weighted average rating
    user_mean = np.mean(train_matrix[user_id][train_matrix[user_id] > 0]) if np.any(train_matrix[user_id] > 0) else 3.0
    
    weighted_sum = 0
    similarity_sum = 0
    
    for i, neighbor in enumerate(neighbor_users):
        neighbor_rating = train_matrix[neighbor, item_id]
        neighbor_mean = np.mean(train_matrix[neighbor][train_matrix[neighbor] > 0])
        
        # 使用中心化评分进行加权
        # Use centered ratings for weighting
        weighted_sum += neighbor_similarities[i] * (neighbor_rating - neighbor_mean)
        similarity_sum += np.abs(neighbor_similarities[i])
    
    if similarity_sum == 0:
        return user_mean
    
    prediction = user_mean + weighted_sum / similarity_sum
    
    # 将评分限制在1-5范围内
    # Constrain rating to 1-5 range
    prediction = np.clip(prediction, 1, 5)
    
    return prediction

# 步骤5：模型评估
# Step 5: Model evaluation
print("\nStep 5: Model evaluation")

# 在测试集上进行预测
# Make predictions on test set
predictions = []
true_ratings = []

for _, row in test_df.iterrows():
    user_id = int(row['user_id'])
    item_id = int(row['item_id'])
    true_rating = row['rating']
    
    predicted_rating = predict_rating(user_id, item_id, train_matrix, user_similarity, k=20)
    
    predictions.append(predicted_rating)
    true_ratings.append(true_rating)
    
    # 显示前10个预测
    # Display first 10 predictions
    if len(predictions) <= 10:
        print(f"User {user_id}, Item {item_id}: True={true_rating}, Predicted={predicted_rating:.2f}")

# 计算均方根误差(RMSE)
# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(true_ratings, predictions))
print(f"\nModel RMSE: {rmse:.4f}")

# 计算平均绝对误差(MAE)
# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(np.array(true_ratings) - np.array(predictions)))
print(f"Model MAE: {mae:.4f}")

# 步骤6：生成推荐
# Step 6: Generate recommendations
def get_top_n_recommendations(user_id, train_matrix, user_similarity, n=10, k=20):
    """
    为用户生成Top-N推荐
    Generate Top-N recommendations for user
    
    Args:
        user_id: 用户ID
        train_matrix: 训练评分矩阵
        user_similarity: 用户相似度矩阵
        n: 推荐物品数量
        k: 考虑的最近邻数量
    
    Returns:
        推荐物品列表和预测评分
    """
    # 找出用户已评分的物品
    # Find items the user has already rated
    user_ratings = train_matrix[user_id]
    rated_items = np.where(user_ratings > 0)[0]
    
    # 为所有未评分物品生成预测
    # Generate predictions for all unrated items
    predictions = []
    for item_id in range(num_items):
        if item_id not in rated_items:  # 只考虑未评分物品
            pred = predict_rating(user_id, item_id, train_matrix, user_similarity, k)
            predictions.append((item_id, pred))
    
    # 按预测评分排序
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # 返回前n个推荐
    # Return top n recommendations
    return predictions[:n]

# 为用户0生成推荐
# Generate recommendations for user 0
print("\nStep 6: Generating recommendations for user 0")
recommendations = get_top_n_recommendations(0, train_matrix, user_similarity, n=5, k=20)

print("\nTop 5 recommendations for user 0:")
for i, (item_id, rating) in enumerate(recommendations):
    print(f"{i+1}. Item {item_id}: Predicted rating = {rating:.2f}")

# 显示用户0已评分的物品
# Show items rated by user 0
user_0_ratings = train_matrix[0]
user_0_rated_items = np.where(user_0_ratings > 0)[0]

if len(user_0_rated_items) > 0:
    print(f"\nUser 0's rated items:")
    for item_id in user_0_rated_items[:10]:  # 只显示前10个
        print(f"Item {item_id}: Rating = {user_0_ratings[item_id]}")
else:
    print(f"\nUser 0 has not rated any items in the training set.")

print("\nRecommendation system completed!")