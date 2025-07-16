"""
从零实现GRU - 深入理解门控循环单元
GRU Implementation from Scratch - Deep Understanding of Gated Recurrent Units

GRU是LSTM的简化版本，只有两个门但效果相近，就像用更简单的机制实现类似的记忆功能。
GRU is a simplified version of LSTM with only two gates but similar performance, 
like achieving similar memory functionality with simpler mechanisms.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import sys
import os

# 添加上级目录到路径以导入工具函数
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import set_random_seed


class GRUCellFromScratch:
    """
    从零实现的GRU单元
    GRU Cell implemented from scratch
    
    GRU比LSTM更简单，只有两个门：
    - 重置门：决定忘记多少过去信息
    - 更新门：决定保留多少过去信息和接受多少新信息
    
    GRU is simpler than LSTM, with only two gates:
    - Reset gate: decides how much past information to forget
    - Update gate: decides how much past information to keep and how much new information to accept
    """
    
    def __init__(self, input_size: int, hidden_size: int, device: str = 'cpu'):
        """
        初始化GRU单元参数
        Initialize GRU cell parameters
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        self._init_weights()
        
    def _init_weights(self):
        """
        初始化权重和偏置
        Initialize weights and biases
        
        GRU只需要3组权重：重置门、更新门、候选状态
        GRU only needs 3 sets of weights: reset gate, update gate, candidate state
        """
        std = 1.0 / np.sqrt(self.hidden_size)
        
        # 重置门权重 | Reset gate weights
        self.W_r = torch.randn(self.hidden_size, self.input_size + self.hidden_size,
                              dtype=torch.float32, device=self.device) * std
        self.b_r = torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)
        
        # 更新门权重 | Update gate weights  
        self.W_z = torch.randn(self.hidden_size, self.input_size + self.hidden_size,
                              dtype=torch.float32, device=self.device) * std
        self.b_z = torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)
        
        # 候选状态权重 | Candidate state weights
        self.W_h = torch.randn(self.hidden_size, self.input_size + self.hidden_size,
                              dtype=torch.float32, device=self.device) * std
        self.b_h = torch.zeros(self.hidden_size, dtype=torch.float32, device=self.device)
        
        self.parameters = [self.W_r, self.b_r, self.W_z, self.b_z, self.W_h, self.b_h]
        
        for param in self.parameters:
            param.requires_grad_(True)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        GRU前向传播
        GRU forward propagation
        
        GRU的计算过程更简洁：
        1. 计算重置门和更新门
        2. 使用重置门计算候选状态
        3. 使用更新门组合旧状态和新状态
        
        GRU computation is more concise:
        1. Compute reset and update gates
        2. Use reset gate to compute candidate state
        3. Use update gate to combine old and new states
        """
        # 组合输入
        combined = torch.cat([x, h_prev], dim=1)
        
        # 步骤1：重置门 - "我应该忘记多少过去的信息？"
        # Step 1: Reset gate - "How much past information should I forget?"
        # r_t = σ(W_r · [h_{t-1}, x_t] + b_r)
        r_t = torch.sigmoid(combined @ self.W_r.T + self.b_r)
        
        # 步骤2：更新门 - "我应该更新多少状态？"
        # Step 2: Update gate - "How much state should I update?"
        # z_t = σ(W_z · [h_{t-1}, x_t] + b_z)
        z_t = torch.sigmoid(combined @ self.W_z.T + self.b_z)
        
        # 步骤3：候选状态 - "新的候选状态是什么？"
        # Step 3: Candidate state - "What is the new candidate state?"
        # 注意：这里使用重置门控制的旧状态
        # Note: Here we use reset gate controlled old state
        reset_combined = torch.cat([x, r_t * h_prev], dim=1)
        h_tilde = torch.tanh(reset_combined @ self.W_h.T + self.b_h)
        
        # 步骤4：最终状态 - "组合旧状态和新状态"
        # Step 4: Final state - "Combine old and new states"
        # h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
        # 这是GRU的核心创新：一个门同时控制遗忘和输入
        # This is GRU's core innovation: one gate controls both forgetting and input
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size, device=self.device)


class GRUFromScratch(nn.Module):
    """完整的GRU网络实现"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 output_size: int = 1, dropout: float = 0.0):
        super(GRUFromScratch, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 创建GRU单元层
        self.gru_cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.gru_cells.append(GRUCellFromScratch(layer_input_size, hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        seq_len, batch_size, _ = x.size()
        
        # 初始化隐藏状态
        if hidden is None:
            hidden_states = []
            for gru_cell in self.gru_cells:
                h_0 = gru_cell.init_hidden(batch_size)
                hidden_states.append(h_0)
        else:
            hidden_states = hidden
        
        outputs = []
        
        # 对每个时间步进行处理
        for t in range(seq_len):
            x_t = x[t]
            
            # 通过所有GRU层
            for layer_idx, gru_cell in enumerate(self.gru_cells):
                h_t = gru_cell.forward(x_t, hidden_states[layer_idx])
                hidden_states[layer_idx] = h_t
                x_t = h_t
                
                if self.dropout is not None and layer_idx < len(self.gru_cells) - 1:
                    x_t = self.dropout(x_t)
            
            # 通过输出层
            output_t = self.output_layer(x_t)
            outputs.append(output_t)
        
        outputs = torch.stack(outputs, dim=0)
        return outputs, hidden_states


def demonstrate_gru_gates():
    """演示GRU门控机制"""
    print("🧠 GRU门控机制演示 | GRU Gating Mechanism Demonstration")
    print("=" * 60)
    
    set_random_seed(42)
    
    input_size = 3
    hidden_size = 4
    batch_size = 1
    
    gru_cell = GRUCellFromScratch(input_size, hidden_size)
    
    x = torch.tensor([[1.0, 0.5, -0.3]], dtype=torch.float32)
    h_prev = torch.tensor([[0.2, -0.1, 0.3, 0.0]], dtype=torch.float32)
    
    print(f"输入 x: {x.numpy()}")
    print(f"前一隐藏状态 h_prev: {h_prev.numpy()}")
    print()
    
    # 手动计算每个门的值
    combined = torch.cat([x, h_prev], dim=1)
    
    # 重置门
    r_t = torch.sigmoid(combined @ gru_cell.W_r.T + gru_cell.b_r)
    print(f"🔄 重置门 r_t: {r_t.detach().numpy()}")
    print("   -> 值越接近0表示越要重置（忘记）过去信息")
    print("   -> Values closer to 0 mean more reset (forget) of past information")
    
    # 更新门
    z_t = torch.sigmoid(combined @ gru_cell.W_z.T + gru_cell.b_z)
    print(f"🔄 更新门 z_t: {z_t.detach().numpy()}")
    print("   -> 值越接近1表示越要保留过去信息，越接近0表示越要使用新信息")
    print("   -> Values closer to 1 mean more retention of past info, closer to 0 mean more use of new info")
    
    # 候选状态
    reset_combined = torch.cat([x, r_t * h_prev], dim=1)
    h_tilde = torch.tanh(reset_combined @ gru_cell.W_h.T + gru_cell.b_h)
    print(f"✨ 候选状态 h_tilde: {h_tilde.detach().numpy()}")
    print("   -> 基于重置后的过去信息和当前输入计算的新状态")
    print("   -> New state computed based on reset past information and current input")
    
    # 最终状态
    h_t = (1 - z_t) * h_prev + z_t * h_tilde
    print(f"🎯 新隐藏状态 h_t: {h_t.detach().numpy()}")
    print("   -> 更新门控制的过去状态和候选状态的加权组合")
    print("   -> Weighted combination of past state and candidate state controlled by update gate")
    
    print("\n" + "=" * 60)
    print("🔍 GRU vs LSTM 关键区别：")
    print("🔍 Key differences between GRU and LSTM:")
    print("1. GRU只有2个门，LSTM有3个门")
    print("   GRU has 2 gates, LSTM has 3 gates")
    print("2. GRU没有单独的细胞状态，LSTM有")
    print("   GRU has no separate cell state, LSTM does")
    print("3. GRU的更新门同时控制遗忘和输入")
    print("   GRU's update gate controls both forgetting and input")


def train_gru_sequence():
    """训练GRU进行序列预测"""
    print("\n📈 GRU序列预测训练演示 | GRU Sequence Prediction Training Demo")
    print("=" * 60)
    
    set_random_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成更复杂的序列数据 - 组合正弦波
    seq_length = 50
    num_samples = 1000
    
    t = np.linspace(0, 8*np.pi, num_samples)
    data = np.sin(t) + 0.5*np.sin(2*t) + 0.2*np.random.randn(num_samples)
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    
    X = np.array(X)
    y = np.array(y)
    
    X = torch.FloatTensor(X).unsqueeze(-1).to(device)
    y = torch.FloatTensor(y).unsqueeze(-1).to(device)
    X = X.transpose(0, 1)
    
    # 创建GRU模型
    model = GRUFromScratch(
        input_size=1,
        hidden_size=32,
        num_layers=2,
        output_size=1,
        dropout=0.1
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    num_epochs = 100
    losses = []
    
    print("开始训练GRU...")
    
    for epoch in range(num_epochs):
        model.train()
        
        outputs, _ = model(X)
        predictions = outputs[-1]
        loss = criterion(predictions, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'轮次 {epoch+1}/{num_epochs}, 损失: {loss.item():.6f}')
    
    # 测试和可视化
    model.eval()
    with torch.no_grad():
        test_outputs, _ = model(X)
        test_predictions = test_outputs[-1].cpu().numpy()
        actual_values = y.cpu().numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('GRU训练损失 | GRU Training Loss')
    plt.xlabel('轮次 | Epoch')
    plt.ylabel('损失 | Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    indices = range(200)
    plt.plot(indices, actual_values[:200], 'b-', label='实际值 | Actual', alpha=0.7)
    plt.plot(indices, test_predictions[:200], 'r-', label='预测值 | Predicted', alpha=0.7)
    plt.title('GRU预测结果 | GRU Prediction Results')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gru_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, losses


if __name__ == "__main__":
    print("🚀 开始GRU从零实现演示 | Starting GRU From Scratch Demonstration")
    print("=" * 80)
    
    # 演示GRU门控机制
    demonstrate_gru_gates()
    
    # 训练GRU序列预测
    model, losses = train_gru_sequence()
    
    print("\n🎉 GRU演示完成！")
    print("🎉 GRU demonstration completed!")
    print("\n📚 GRU的优势：")
    print("📚 Advantages of GRU:")
    print("1. 参数更少，训练更快")
    print("   Fewer parameters, faster training")
    print("2. 结构更简单，易于理解")
    print("   Simpler structure, easier to understand")  
    print("3. 在很多任务上效果接近LSTM")
    print("   Performance close to LSTM on many tasks") 