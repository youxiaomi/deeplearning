import matplotlib.pyplot as plt

X = [1, 2, 3]
y = [2, 3, 4]

# 初始化参数
W, b = 1.0, 0.5
lr = 0.1  # 学习率
epochs = 100  # 迭代次数

# 用于存储每个epoch的y_pred值
all_y_preds = []

for epoch in range(epochs):
    # 计算预测值和误差
    y_pred = [W * x_val + b for x_val in X]
    all_y_preds.append(y_pred) # 存储当前epoch的y_pred
    print(f"y_pred: {y_pred}")
    error = [y_true - y_p for y_true, y_p in zip(y, y_pred)]
    
    mse = sum([e ** 2 for e in error]) / len(X)
    
    # 计算梯度
    dW_sum = 0
    db_sum = 0
    for i in range(len(X)):
        dW_sum += error[i] * X[i]
        db_sum += error[i]
    
    dW = -2/len(X) * dW_sum
    db = -2/len(X) * db_sum
    
    # 更新参数
    W -= lr * dW
    b -= lr * db
    
    print(f"Epoch {epoch}: W={W:.2f}, b={b:.2f}, MSE={mse:.4f}")

# 绘制图形
epochs_list = list(range(epochs))

# 将all_y_preds转换为适合绘图的格式
# 例如，y_pred_0_values, y_pred_1_values, y_pred_2_values
y_pred_values_by_index = [[all_y_preds[epoch][i] for epoch in range(epochs)] for i in range(len(X))]

plt.figure(figsize=(10, 6))
for i, y_values in enumerate(y_pred_values_by_index):
    plt.plot(epochs_list, y_values, label=f'y_pred[{i}] (for X[{i}]={X[i]})')

plt.xlabel('Epoch')
plt.ylabel('Predicted Y Value')
plt.title('Predicted Y Values vs. Epochs')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression_plot.png')
# plt.show()