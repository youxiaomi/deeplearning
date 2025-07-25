import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import random

class UserBasedCollaborativeFiltering:
    def __init__(self, num_users, num_items, data_path=None):
        """
        基于用户的协同过滤推荐系统
        User-based collaborative filtering recommendation system
        
        Args:
            num_users: 用户数量 Number of users
            num_items: 物品数量 Number of items
            data_path: 数据文件路径 Path to data file
        """
        self.num_users = num_users
        self.num_items = num_items
        self.data_path = data_path
        self.rating_matrix = None  # 用户-物品评分矩阵 User-item rating matrix
        self.user_similarity = None  # 用户相似度矩阵 User similarity matrix
        self.user_mean = None  # 用户平均评分 User mean rating
        
    def generate_sample_data(self, num_ratings=1000, sparsity_rate=0.95):
        """
        生成模拟用户评分数据
        Generate synthetic user rating data
        
        Args:
            num_ratings: 总评分数量 Total number of ratings
            sparsity_rate: 稀疏率，0.95表示95%的单元格为空
                           Sparsity rate, 0.95 means 95% of cells are empty
        """
        print("🔄 正在生成模拟评分数据...")
        print("🔄 Generating synthetic rating data...")
        
        # 初始化评分矩阵
        # Initialize rating matrix
        self.rating_matrix = np.zeros((self.num_users, self.num_items))
        
        # 生成评分数据
        # Generate rating data
        generated = 0
        while generated < num_ratings:
            user_id = random.randint(0, self.num_users - 1)
            item_id = random.randint(0, self.num_items - 1)
            
            # 跳过已存在的评分
            # Skip existing ratings
            if self.rating_matrix[user_id, item_id] != 0:
                continue
                
            # 模拟用户评分行为：引入用户偏好和物品受欢迎程度
            # Simulate user rating behavior: introduce user preference and item popularity
            user_bias = np.random.normal(0, 0.5)  # 用户评分偏见 User rating bias
            item_popularity = np.random.normal(3.5, 1.0)  # 物品基础得分 Item base score
            
            # 评分 = 物品受欢迎程度 + 用户偏见 + 噪声
            # Rating = Item popularity + User bias + Noise
            rating = item_popularity + user_bias + np.random.normal(0, 0.3)
            rating = np.clip(rating, 1, 5)  # 限制在1-5分之间 Constrain to 1-5 range
            rating = round(rating * 2) / 2  # 四舍五入到0.5的倍数 Round to nearest 0.5
            
            self.rating_matrix[user_id, item_id] = rating
            generated += 1
            
        print(f"✅ 模拟数据生成完成！共{generated}条评分记录");
        print(f"✅ Data generation completed! {generated} rating records generated")
        print(f"🔋 矩阵密度: {(generated/(self.num_users*self.num_items)):.2%}")
        print(f"🔋 Matrix density: {(generated/(self.num_users*self.num_items)):.2%}")
        
        return self.rating_matrix
    
    def load_data(self):
        """
        从文件加载数据
        Load data from file
        """
        if self.data_path is None:
            raise ValueError("数据路径未指定！Data path not specified!")
            
        try:
            df = pd.read_csv(self.data_path)
            self.rating_matrix = np.zeros((self.num_users, self.num_items))
            
            for _, row in df.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = float(row['rating'])
                self.rating_matrix[user_id, item_id] = rating
                
            print(f"✅ 数据加载成功！成功加载{len(df)}条记录")
            print(f"✅ Data loaded successfully! {len(df)} records loaded")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print(f"❌ Data loading failed: {e}")
            # 失败时生成模拟数据
            # Generate synthetic data on failure
            self.generate_sample_data()
            
        return self.rating_matrix
    
    def preprocess_data(self):
        """
        数据预处理：计算用户平均分，处理缺失值
        Data preprocessing: calculate user mean, handle missing values
        """
        print("🧹 开始数据预处理...")
        print("🧹 Starting data preprocessing...")
        
        # 计算每个用户的平均评分（仅考虑非零评分）
        # Calculate mean rating for each user (only non-zero ratings)
        self.user_mean = np.zeros(self.num_users)
        for user_id in range(self.num_users):
            user_ratings = self.rating_matrix[user_id][self.rating_matrix[user_id] != 0]
            if len(user_ratings) > 0:
                self.user_mean[user_id] = np.mean(user_ratings)
            else:
                self.user_mean[user_id] = 3.0  # 默认平均分 Default mean rating
        
        print("✅ 数据预处理完成！计算了用户平均分")
        print("✅ Data preprocessing completed! Calculated user mean ratings")
        
        return self.rating_matrix, self.user_mean
    
    def calculate_similarity(self, method='cosine'):
        """
        计算用户相似度矩阵
        Calculate user similarity matrix
        
        Args:
            method: 相似度计算方法 'cosine' 或 'pearson'
                    Similarity method: 'cosine' or 'pearson'
        """
        print(f"🔍 使用{method}方法计算用户相似度...")
        print(f"🔍 Calculating user similarity using {method} method...")
        
        # 创建用于计算相似度的矩阵
        # Create matrix for similarity calculation
        if method == 'pearson':
            # 皮尔逊相关系数需要中心化处理
            # Pearson correlation requires centering
            similarity_matrix = np.zeros((self.num_users, self.num_users))
            
            for i in range(self.num_users):
                for j in range(i, self.num_users):  # 只计算上三角矩阵
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # 找到两个用户都评过分的物品
                        # Find items rated by both users
                        common_items = ((self.rating_matrix[i] != 0) & 
                                      (self.rating_matrix[j] != 0))
                        
                        if np.sum(common_items) < 2:  # 至少需要2个共同评分
                            similarity_matrix[i, j] = 0.0
                            similarity_matrix[j, i] = 0.0
                        else:
                            # 提取共同评分
                            # Extract common ratings
                            ratings_i = self.rating_matrix[i][common_items]
                            ratings_j = self.rating_matrix[j][common_items]
                            
                            # 中心化处理
                            # Centering
                            ratings_i_centered = ratings_i - np.mean(ratings_i)
                            ratings_j_centered = ratings_j - np.mean(ratings_j)
                            
                            # 计算皮尔逊相关系数
                            # Calculate Pearson correlation
                            numerator = np.sum(ratings_i_centered * ratings_j_centered)
                            denominator = (np.sqrt(np.sum(ratings_i_centered**2)) * 
                                         np.sqrt(np.sum(ratings_j_centered**2)))
                            
                            if denominator == 0:
                                sim = 0.0
                            else:
                                sim = numerator / denominator
                                
                            similarity_matrix[i, j] = sim
                            similarity_matrix[j, i] = sim
        
        elif method == 'cosine':
            # 余弦相似度
            # Cosine similarity
            similarity_matrix = cosine_similarity(self.rating_matrix)
            
        self.user_similarity = similarity_matrix
        print("✅ 相似度计算完成！")
        print("✅ Similarity calculation completed!")
        
        return self.user_similarity
    
    def predict_rating(self, user_id, item_id, k=20):
        """
        预测用户对物品的评分
        Predict user's rating for an item
        
        Args:
            user_id: 用户ID User ID
            item_id: 物品ID Item ID
            k: 使用多少个最相似的用户 Use how many most similar users
        """
        # 找到目标用户已经评分的物品
        # Find items already rated by target user
        user_rated_items = self.rating_matrix[user_id] != 0
        
        # 计算其他用户与目标用户的相似度
        # Calculate similarity between other users and target user
        similarities = self.user_similarity[user_id]
        
        # 找到与目标用户有共同评分物品的用户
        # Find users who have common rated items with target user
        potential_neighbors = []
        for other_user_id in range(self.num_users):
            if other_user_id == user_id:
                continue
                
            # 检查是否有共同评分的物品
            # Check if there are common rated items
            other_rated_items = self.rating_matrix[other_user_id] != 0
            if np.sum(user_rated_items & other_rated_items) > 0:  # 至少有一个共同评分
                potential_neighbors.append(other_user_id)
        
        if len(potential_neighbors) == 0:
            # 没有找到相似用户，返回用户平均分
            # No similar users found, return user's mean rating
            return self.user_mean[user_id]
        
        # 从潜在邻居中选择最相似的k个用户
        # Select top-k most similar users from potential neighbors
        neighbor_similarities = [(sim, uid) for uid, sim in enumerate(similarities) 
                               if uid in potential_neighbors]
        neighbor_similarities.sort(reverse=True)  # 从高到低排序
        top_neighbors = neighbor_similarities[:k]
        
        if len(top_neighbors) == 0 or top_neighbors[0][0] <= 0:
            # 没有正相似度的邻居，返回用户平均分
            # No neighbors with positive similarity, return user's mean rating
            return self.user_mean[user_id]
        
        # 计算加权预测评分
        # Calculate weighted predicted rating
        numerator = 0.0
        denominator = 0.0
        
        for similarity, neighbor_id in top_neighbors:
            # 检查邻居是否评价过该物品
            # Check if neighbor has rated this item
            if self.rating_matrix[neighbor_id, item_id] == 0:
                continue
                
            # 邻居的评分偏差
            # Neighbor's rating deviation
            neighbor_deviation = self.rating_matrix[neighbor_id, item_id] - self.user_mean[neighbor_id]
            
            numerator += similarity * neighbor_deviation
            denominator += abs(similarity)  # 使用绝对值避免负值抵消
                                  # Use absolute value to prevent negative cancellation

        if denominator == 0:
            return self.user_mean[user_id]
        
        # 最终预测评分 = 用户平均分 + 加权偏差
        # Final predicted rating = user mean + weighted deviation
        predicted_rating = self.user_mean[user_id] + (numerator / denominator)
        predicted_rating = np.clip(predicted_rating, 1, 5)  # 限制在1-5分之间
                                             # Constrain to 1-5 range
        
        return predicted_rating
    
    def evaluate(self, test_data, k=20):
        """
        评估模型性能
        Evaluate model performance
        
        Args:
            test_data: 测试数据 [(user_id, item_id, true_rating), ...]
                       Test data [(user_id, item_id, true_rating), ...]
            k: 使用的邻居数量 Number of neighbors used
        """
        predictions = []
        true_ratings = []
        
        print(f"📊 正在评估模型性能，使用k={k}个邻居...")
        print(f"📊 Evaluating model performance, using k={k} neighbors...")
        
        for user_id, item_id, true_rating in test_data:
            pred_rating = self.predict_rating(user_id, item_id, k)
            predictions.append(pred_rating)
            true_ratings.append(true_rating)
        
        # 计算RMSE和MAE
        # Calculate RMSE and MAE
        predictions = np.array(predictions)
        true_ratings = np.array(true_ratings)
        
        rmse = np.sqrt(np.mean((predictions - true_ratings) ** 2))
        mae = np.mean(np.abs(predictions - true_ratings))
        
        print(f"📈 评估结果：")
        print(f"📈 Evaluation results:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE:  {mae:.4f}")
        
        return rmse, mae
    
    def get_top_n_recommendations(self, user_id, n=10, k=20):
        """
        为用户生成Top-N推荐
        Generate Top-N recommendations for user
        
        Args:
            user_id: 用户ID User ID
            n: 推荐数量 Number of recommendations
            k: 使用的邻居数量 Number of neighbors used
        """
        print(f"🎯 为用户{user_id}生成Top-{n}推荐...")
        print(f"🎯 Generating Top-{n} recommendations for user {user_id}...")
        
        # 找出用户未评分的物品
        # Find items not rated by user
        user_rated = self.rating_matrix[user_id] != 0
        unrated_items = np.where(~user_rated)[0]
        
        if len(unrated_items) == 0:
            print("❌ 用户已经评过所有物品，无法推荐")
            print("❌ User has rated all items, cannot recommend")
            return []
        
        # 预测所有未评分物品的评分
        # Predict ratings for all unrated items
        item_predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict_rating(user_id, item_id, k)
            item_predictions.append((item_id, pred_rating))
        
        # 按预测评分排序
        # Sort by predicted rating
        item_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # 返回Top-N
        # Return Top-N
        top_n = item_predictions[:n]
        
        print(f"✅ Top-{n}推荐生成完成！")
        print("✅ Top-N recommendations generated!")
        
        return top_n

# ===================== 使用示例和测试 =====================
# ===================== Usage Example and Testing =====================

def main():
    """
    主函数：演示用户协同过滤系统的工作流程
    Main function: demonstrate the workflow of user-based collaborative filtering system
    """
    print("🚀 开始演示基于用户的协同过滤推荐系统")
    print("🚀 Starting demonstration of user-based collaborative filtering recommendation system")
    print("=" * 60)
    
    # 1. 初始化系统
    # 1. Initialize system
    NUM_USERS = 100
    NUM_ITEMS = 50
    recommender = UserBasedCollaborativeFiltering(NUM_USERS, NUM_ITEMS)
    
    # 2. 生成模拟数据
    # 2. Generate synthetic data
    data = recommender.generate_sample_data(num_ratings=800)
    
    # 3. 数据预处理
    # 3. Data preprocessing
    rating_matrix, user_mean = recommender.preprocess_data()
    
    # 4. 计算用户相似度
    # 4. Calculate user similarity
    similarity_matrix = recommender.calculate_similarity(method='pearson')
    
    # 5. 模型评估
    # 5. Model evaluation
    # 创建测试集：随机选择一些已有的评分作为测试
    # Create test set: randomly select some existing ratings as test
    test_data = []
    for _ in range(100):
        user_id = random.randint(0, NUM_USERS - 1)
        item_id = random.randint(0, NUM_ITEMS - 1)
        if rating_matrix[user_id, item_id] != 0:  # 确保该评分存在
            test_data.append((user_id, item_id, rating_matrix[user_id, item_id]))
    
    rmse, mae = recommender.evaluate(test_data, k=20)
    
    # 6