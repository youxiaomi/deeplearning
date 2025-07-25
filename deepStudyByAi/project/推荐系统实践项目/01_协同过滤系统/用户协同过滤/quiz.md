# User-based Collaborative Filtering Quiz | 用户协同过滤测试题

## 📝 Basic Knowledge Test | 基础知识测试

1. **What is the basic assumption of user-based collaborative filtering?** | **用户协同过滤的基本假设是什么？**
   
   **Answer:** The basic assumption of user-based collaborative filtering is "similar users like similar items". If user A and user B have similar preferences in the past, then items that user A likes but user B hasn't encountered yet are likely to be liked by user B as well.
   **答案：** 用户协同过滤的基本假设是"相似用户喜欢相似物品"。如果用户A和用户B在过去的偏好上很相似，那么用户A喜欢而用户B还没接触过的物品，很可能也会被用户B喜欢。

2. **What is cosine similarity? What is its role in recommendation systems?** | **什么是余弦相似度？它在推荐系统中有什么作用？**
   
   **Answer:** Cosine similarity is a method of measuring the similarity between two vectors by calculating the cosine of the angle between them. In recommendation systems, it is used to calculate the similarity between users - the smaller the angle (the closer the cosine value is to 1), the more similar the two users' tastes are.
   **答案：** 余弦相似度是通过计算两个向量夹角的余弦值来衡量它们相似程度的方法。在推荐系统中，它用于计算用户之间的相似度，夹角越小（余弦值越接近1），表示两个用户的品味越相似。

3. **Why do we subtract the user's average rating in the rating prediction formula?** | **为什么在评分预测公式中要减去用户的平均评分？**
   
   **Answer:** Subtracting the user's average rating is to eliminate the user's rating bias. Some users have the habit of giving high scores (e.g., always giving 4-5 stars), while others are very strict (often giving 1-2 stars). By subtracting the average score, we focus on the user's preference relative to their own average rating, rather than the absolute rating value.
   **答案：** 减去用户的平均评分是为了消除用户的评分偏见。有些用户习惯性给高分（比如总是打4-5分），有些用户则很严格（经常打1-2分）。通过减去平均分，我们关注的是用户相对于自己平均评分的偏好，而不是绝对评分值。

## 🧮 Calculation Problems | 计算题

4. **Calculate the cosine similarity between user A[5,3,4] and user B[4,2,5].** | **计算用户A[5,3,4]和用户B[4,2,5]的余弦相似度。**
   
   **Answer:**
   [ \vec{A} \cdot \vec{B} = 5 \times 4 + 3 \times 2 + 4 \times 5 = 20 + 6 + 20 = 46 ]
   [ \|\vec{A}\| = \sqrt{5^2 + 3^2 + 4^2} = \sqrt{25 + 9 + 16} = \sqrt{50} \approx 7.07 ]
   [ \|\vec{B}\| = \sqrt{4^2 + 2^2 + 5^2} = \sqrt{16 + 4 + 25} = \sqrt{45} \approx 6.71 ]
   [ \text{sim}(A,B) = \frac{46}{7.07 \times 6.71} \approx \frac{46}{47.46} \approx 0.97 ]

   **答案：**
   相似度约为0.97。
   The similarity is approximately 0.97.

5. **User A's average rating is 3.5, user B's average rating is 3.0. User A's rating for the movie 'Interstellar' is 5. If their similarity is 0.9, predict user B's rating for 'Interstellar'.** | **用户A的平均评分是3.5，用户B的平均评分是3.0。用户A对电影《星际穿越》的评分是5。如果他们之间的相似度是0.9，预测用户B对《星际穿越》的评分。**
   
   **Answer:**
   Using the rating prediction formula:
   [ \hat{r}_{B,\text{Interstellar}} = \bar{r}_B + \text{sim}(A,B) \times (r_{A,\text{Interstellar}} - \bar{r}_A) ]
   [ = 3.0 + 0.9 \times (5 - 3.5) ]
   [ = 3.0 + 0.9 \times 1.5 ]
   [ = 3.0 + 1.35 ]
   [ = 4.35 ]

   **答案：**
   预测评分为4.35分。
   The predicted rating is 4.35.