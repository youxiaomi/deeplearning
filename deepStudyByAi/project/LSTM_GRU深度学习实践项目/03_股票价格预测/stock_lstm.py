"""
使用LSTM进行股票价格预测 - 时间序列预测的实际应用
Stock Price Prediction using LSTM - Real-world Application of Time Series Forecasting

这个项目演示如何使用LSTM预测股票价格，虽然实际投资需要考虑更多因素，
但这是学习时间序列预测的好例子。

This project demonstrates how to use LSTM for stock price prediction. 
Although actual investment requires considering more factors, 
this is a good example for learning time series forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# 添加上级目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import set_random_seed, create_sequences, normalize_data


class StockLSTM(nn.Module):
    """
    股票价格预测LSTM模型
    Stock Price Prediction LSTM Model
    
    这个模型就像一个经验丰富的分析师，通过观察历史价格模式来预测未来走势。
    This model is like an experienced analyst who predicts future trends by observing historical price patterns.
    """
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        初始化股票预测LSTM模型
        Initialize stock prediction LSTM model
        
        Args:
            input_size: 输入特征数（如开盘价、收盘价、成交量等）| Number of input features
            hidden_size: 隐藏层大小 | Hidden layer size
            num_layers: LSTM层数 | Number of LSTM layers
            output_size: 输出大小（通常为1，预测收盘价）| Output size (usually 1, predicting close price)
            dropout: Dropout比率 | Dropout ratio
        """
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层 - 核心的时间序列处理层
        # LSTM layer - core time series processing layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 注意力机制（简化版）- 让模型关注重要的时间点
        # Attention mechanism (simplified) - let model focus on important time points
        self.attention = nn.Linear(hidden_size, 1)
        
        # 全连接层 - 将LSTM输出转换为价格预测
        # Fully connected layer - convert LSTM output to price prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        """
        前向传播
        Forward propagation
        
        Args:
            x: 输入序列 [batch_size, seq_length, features] | Input sequence
            
        Returns:
            预测的股价 | Predicted stock price
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        # LSTM forward propagation
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 应用注意力机制
        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended_output = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 最终预测
        # Final prediction
        output = self.fc(attended_output)
        
        return output


def download_stock_data(symbol='AAPL', period='2y'):
    """
    下载股票数据
    Download stock data
    
    Args:
        symbol: 股票代码 | Stock symbol
        period: 时间周期 | Time period
        
    Returns:
        DataFrame: 股票数据 | Stock data
    """
    print(f"📈 下载股票数据: {symbol} | Downloading stock data: {symbol}")
    
    try:
        # 使用yfinance下载数据
        # Download data using yfinance
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            print("⚠️ 无法获取股票数据，使用模拟数据")
            print("⚠️ Unable to fetch stock data, using simulated data")
            return create_simulated_stock_data()
        
        print(f"✅ 成功下载 {len(data)} 天的数据")
        print(f"✅ Successfully downloaded {len(data)} days of data")
        
        return data
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print(f"❌ Download failed: {str(e)}")
        print("使用模拟数据代替 | Using simulated data instead")
        return create_simulated_stock_data()


def create_simulated_stock_data():
    """
    创建模拟股票数据
    Create simulated stock data
    """
    print("🎭 创建模拟股票数据 | Creating simulated stock data")
    
    # 生成500天的模拟数据
    # Generate 500 days of simulated data
    np.random.seed(42)
    days = 500
    
    # 基础价格趋势
    # Base price trend
    base_price = 100
    trend = np.cumsum(np.random.randn(days) * 0.01)
    
    # 季节性模式
    # Seasonal pattern
    seasonal = 5 * np.sin(np.arange(days) * 2 * np.pi / 252)  # 年度季节性
    
    # 价格计算
    # Price calculation
    prices = base_price + trend + seasonal + np.random.randn(days) * 2
    
    # 确保价格为正
    # Ensure positive prices
    prices = np.maximum(prices, 10)
    
    # 创建DataFrame
    # Create DataFrame
    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    
    data = pd.DataFrame({
        'Open': prices + np.random.randn(days) * 0.5,
        'High': prices + np.abs(np.random.randn(days)) * 2,
        'Low': prices - np.abs(np.random.randn(days)) * 2,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    # 确保High >= Low
    # Ensure High >= Low
    data['High'] = np.maximum(data['High'], data['Low'] + 0.01)
    
    return data


def prepare_stock_data(data, sequence_length=60, features=['Close', 'Volume', 'High', 'Low']):
    """
    准备股票数据用于LSTM训练
    Prepare stock data for LSTM training
    
    Args:
        data: 股票数据 | Stock data
        sequence_length: 序列长度 | Sequence length
        features: 使用的特征 | Features to use
        
    Returns:
        训练和测试数据 | Training and test data
    """
    print(f"🔧 准备股票数据 | Preparing stock data")
    print(f"使用特征: {features} | Using features: {features}")
    
    # 计算技术指标
    # Calculate technical indicators
    data = data.copy()
    
    # 移动平均线
    # Moving averages
    data['MA_5'] = data['Close'].rolling(window=5).mean()
    data['MA_20'] = data['Close'].rolling(window=20).mean()
    
    # 相对强弱指数 (RSI)
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林带
    # Bollinger Bands
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
    
    # 价格变化率
    # Price change rate
    data['Price_Change'] = data['Close'].pct_change()
    
    # 成交量变化率
    # Volume change rate
    data['Volume_Change'] = data['Volume'].pct_change()
    
    # 选择特征
    # Select features
    extended_features = features + ['MA_5', 'MA_20', 'RSI', 'Price_Change', 'Volume_Change']
    
    # 移除NaN值
    # Remove NaN values
    data = data.dropna()
    
    # 提取特征数据
    # Extract feature data
    feature_data = data[extended_features].values
    target_data = data['Close'].values
    
    print(f"数据形状: {feature_data.shape} | Data shape: {feature_data.shape}")
    print(f"有效数据点: {len(feature_data)} | Valid data points: {len(feature_data)}")
    
    # 数据标准化
    # Data normalization
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    scaled_features = feature_scaler.fit_transform(feature_data)
    scaled_targets = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
    
    # 创建序列数据
    # Create sequence data
    X, y = [], []
    
    for i in range(sequence_length, len(scaled_features)):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_targets[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"序列数据形状: X={X.shape}, y={y.shape}")
    print(f"Sequence data shape: X={X.shape}, y={y.shape}")
    
    # 划分训练集和测试集
    # Split training and test sets
    train_size = int(len(X) * 0.8)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (X_train, X_test, y_train, y_test, 
            feature_scaler, target_scaler, data.index[sequence_length:])


def train_stock_model(X_train, y_train, X_test, y_test, input_size):
    """
    训练股票预测模型
    Train stock prediction model
    """
    print("\n🚀 开始训练股票预测模型 | Starting stock prediction model training")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} | Using device: {device}")
    
    # 转换为PyTorch张量
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # 创建模型
    # Create model
    model = StockLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        output_size=1,
        dropout=0.2
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                   factor=0.5, patience=10, verbose=True)
    
    # 训练参数
    # Training parameters
    num_epochs = 100
    batch_size = 32
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    train_losses = []
    test_losses = []
    
    print("开始训练... | Starting training...")
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # 批量训练
        # Batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # 梯度裁剪
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_train_loss += loss.item()
        
        # 验证
        # Validation
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test).squeeze()
            test_loss = criterion(test_predictions, y_test).item()
        
        avg_train_loss = total_train_loss / (len(X_train) // batch_size + 1)
        
        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        
        # 学习率调度
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # 早停机制
        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            # 保存最佳模型
            # Save best model
            torch.save(model.state_dict(), 'best_stock_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'轮次 {epoch+1}/{num_epochs}:')
            print(f'  训练损失: {avg_train_loss:.6f}')
            print(f'  测试损失: {test_loss:.6f}')
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.8f}')
        
        if patience_counter >= patience:
            print(f"早停：{patience}轮次无改善")
            break
    
    # 加载最佳模型
    # Load best model
    model.load_state_dict(torch.load('best_stock_model.pth'))
    
    return model, train_losses, test_losses


def evaluate_model(model, X_test, y_test, target_scaler, dates):
    """
    评估模型性能
    Evaluate model performance
    """
    print("\n📊 评估模型性能 | Evaluating model performance")
    print("=" * 50)
    
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).squeeze().cpu().numpy()
        actual = y_test.cpu().numpy()
    
    # 反标准化
    # Denormalize
    predictions_denorm = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actual_denorm = target_scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    
    # 计算评估指标
    # Calculate evaluation metrics
    mse = mean_squared_error(actual_denorm, predictions_denorm)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_denorm, predictions_denorm)
    mape = np.mean(np.abs((actual_denorm - predictions_denorm) / actual_denorm)) * 100
    
    print(f"均方误差 (MSE): {mse:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
    
    # 方向准确率
    # Direction accuracy
    actual_direction = np.diff(actual_denorm) > 0
    pred_direction = np.diff(predictions_denorm) > 0
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print(f"方向预测准确率: {direction_accuracy:.2f}%")
    
    return predictions_denorm, actual_denorm, {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }


def visualize_stock_predictions(predictions, actual, dates, train_losses, test_losses, metrics):
    """
    可视化股票预测结果
    Visualize stock prediction results
    """
    plt.figure(figsize=(20, 15))
    
    # 预测vs实际价格
    # Predictions vs actual prices
    plt.subplot(3, 3, 1)
    test_dates = dates[-len(predictions):]
    plt.plot(test_dates, actual, label='实际价格 | Actual Price', color='blue', alpha=0.7)
    plt.plot(test_dates, predictions, label='预测价格 | Predicted Price', color='red', alpha=0.7)
    plt.title('股票价格预测结果 | Stock Price Prediction Results')
    plt.xlabel('日期 | Date')
    plt.ylabel('价格 | Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 训练损失曲线
    # Training loss curve
    plt.subplot(3, 3, 2)
    plt.plot(train_losses, label='训练损失 | Training Loss')
    plt.plot(test_losses, label='测试损失 | Test Loss')
    plt.title('损失曲线 | Loss Curves')
    plt.xlabel('轮次 | Epoch')
    plt.ylabel('损失 | Loss')
    plt.legend()
    plt.grid(True)
    
    # 预测误差分布
    # Prediction error distribution
    plt.subplot(3, 3, 3)
    errors = predictions - actual
    plt.hist(errors, bins=50, alpha=0.7, color='purple')
    plt.title('预测误差分布 | Prediction Error Distribution')
    plt.xlabel('误差 | Error')
    plt.ylabel('频数 | Frequency')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True)
    
    # 散点图：预测vs实际
    # Scatter plot: predictions vs actual
    plt.subplot(3, 3, 4)
    plt.scatter(actual, predictions, alpha=0.5, color='green')
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
    plt.title('预测vs实际散点图 | Predictions vs Actual Scatter Plot')
    plt.xlabel('实际价格 | Actual Price')
    plt.ylabel('预测价格 | Predicted Price')
    plt.grid(True)
    
    # 残差图
    # Residual plot
    plt.subplot(3, 3, 5)
    residuals = actual - predictions
    plt.scatter(range(len(residuals)), residuals, alpha=0.5, color='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('残差图 | Residual Plot')
    plt.xlabel('样本 | Sample')
    plt.ylabel('残差 | Residual')
    plt.grid(True)
    
    # 性能指标条形图
    # Performance metrics bar chart
    plt.subplot(3, 3, 6)
    metric_names = ['RMSE', 'MAE', 'MAPE', 'Direction_Acc']
    metric_values = [metrics['RMSE'], metrics['MAE'], metrics['MAPE'], metrics['Direction_Accuracy']]
    
    bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
    plt.title('性能指标 | Performance Metrics')
    plt.ylabel('数值 | Value')
    
    # 在柱子上添加数值
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    
    # 近期预测放大图
    # Recent predictions zoom-in
    plt.subplot(3, 3, 7)
    recent_days = 30
    recent_actual = actual[-recent_days:]
    recent_pred = predictions[-recent_days:]
    recent_dates = test_dates[-recent_days:]
    
    plt.plot(recent_dates, recent_actual, label='实际', marker='o', markersize=3)
    plt.plot(recent_dates, recent_pred, label='预测', marker='s', markersize=3)
    plt.title(f'最近{recent_days}天预测 | Recent {recent_days} Days Prediction')
    plt.xlabel('日期 | Date')
    plt.ylabel('价格 | Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 累积误差
    # Cumulative error
    plt.subplot(3, 3, 8)
    cumulative_error = np.cumsum(np.abs(errors))
    plt.plot(cumulative_error, color='purple')
    plt.title('累积绝对误差 | Cumulative Absolute Error')
    plt.xlabel('样本 | Sample')
    plt.ylabel('累积误差 | Cumulative Error')
    plt.grid(True)
    
    # 预测准确率趋势
    # Prediction accuracy trend
    plt.subplot(3, 3, 9)
    window_size = 20
    rolling_mape = []
    for i in range(window_size, len(predictions)):
        window_actual = actual[i-window_size:i]
        window_pred = predictions[i-window_size:i]
        window_mape = np.mean(np.abs((window_actual - window_pred) / window_actual)) * 100
        rolling_mape.append(window_mape)
    
    plt.plot(rolling_mape, color='brown')
    plt.title(f'滚动MAPE ({window_size}天) | Rolling MAPE ({window_size} days)')
    plt.xlabel('样本 | Sample')
    plt.ylabel('MAPE (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('stock_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("📈 LSTM股票价格预测项目 | LSTM Stock Price Prediction Project")
    print("=" * 70)
    
    set_random_seed(42)
    
    # 下载股票数据
    # Download stock data
    stock_data = download_stock_data('AAPL', '2y')
    
    # 准备数据
    # Prepare data
    (X_train, X_test, y_train, y_test, 
     feature_scaler, target_scaler, dates) = prepare_stock_data(stock_data)
    
    # 训练模型
    # Train model
    model, train_losses, test_losses = train_stock_model(
        X_train, y_train, X_test, y_test, X_train.shape[2]
    )
    
    # 评估模型
    # Evaluate model
    predictions, actual, metrics = evaluate_model(
        model, X_test, y_test, target_scaler, dates
    )
    
    # 可视化结果
    # Visualize results
    visualize_stock_predictions(predictions, actual, dates, train_losses, test_losses, metrics)
    
    print("\n🎉 股票预测项目完成！| Stock Prediction Project Completed!")
    print("📚 通过这个项目，你学会了：")
    print("📚 Through this project, you learned:")
    print("1. 如何处理真实的时间序列数据")
    print("   How to handle real time series data")
    print("2. 技术指标的计算和应用")
    print("   Calculation and application of technical indicators")
    print("3. LSTM在金融预测中的应用")
    print("   Application of LSTM in financial prediction")
    print("4. 模型评估和可视化的完整流程")
    print("   Complete process of model evaluation and visualization")
    
    print("\n⚠️ 重要提醒：")
    print("⚠️ Important reminder:")
    print("这个模型仅用于学习目的，不应用于实际投资决策！")
    print("This model is for learning purposes only and should not be used for actual investment decisions!") 