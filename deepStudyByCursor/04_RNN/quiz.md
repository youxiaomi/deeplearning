#!/usr/bin/env python3
"""
Quiz for Chapter 4: Recurrent Neural Networks
第四章测验：循环神经网络

This quiz covers:
- RNN basics and sequential data processing
- Vanishing gradient problem
- LSTM and GRU architectures
- Sequence-to-sequence models

本测验涵盖：
- RNN基础和序列数据处理
- 梯度消失问题
- LSTM和GRU架构
- 序列到序列模型
"""

import torch
import torch.nn as nn
import numpy as np

def multiple_choice_questions():
    """Multiple choice questions about RNNs"""
    print("=== Multiple Choice Questions ===")
    print("=== 选择题 ===\n")
    
    questions = [
        {
            "question": "What is the main advantage of RNNs over traditional feedforward networks?",
            "question_zh": "对于序列数据，RNN相比前馈网络的主要优势是什么？",
            "options": [
                "A) Faster computation",
                "B) Memory of previous inputs through hidden states",
                "C) Better gradient flow",
                "D) Fewer parameters"
            ],
            "answer": "C",
            "explanation": "RNNs maintain hidden states that carry information from previous time steps, allowing them to process sequences.",
            "explanation_zh": "RNN维护隐藏状态，携带来自先前时间步的信息，使其能够处理序列。"
        },
        {
            "question": "What is the role of hidden state h_t in RNNs?",
            "question_zh": "RNN中的隐藏状态h_t的作用是什么？",
            "options": [
                "A) Store current input information",
                "B) Act as network memory, carrying historical information",
                "C) Control gradient flow",
                "D) Reduce computational complexity"
            ],
            "answer": "B",
            "explanation": "Hidden state is the core of RNN, passing information between time steps as network memory.",
            "explanation_zh": "隐藏状态是RNN的核心，它在时间步之间传递信息，充当网络的记忆。"
        },
        {
            "question": "How does the vanishing gradient problem mainly manifest in RNNs?",
            "question_zh": "梯度消失问题在RNN中主要表现为什么？",
            "options": [
                "A) Training becomes slower",
                "B) Cannot learn long-term dependencies",
                "C) Memory usage increases",
                "D) Accuracy decreases"
            ],
            "answer": "B",
            "explanation": "Vanishing gradients make it difficult for RNNs to learn long-range dependencies.",
            "explanation_zh": "梯度消失导致RNN难以学习长距离的依赖关系，这是RNN的主要限制。"
        },
        {
            "question": "In the RNN formula h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b), what does W_hh represent?",
            "question_zh": "在RNN的数学公式h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)中，W_hh表示什么？",
            "options": [
                "A) Input-to-hidden weights",
                "B) Hidden-to-output weights",
                "C) Hidden-to-hidden weights",
                "D) Bias term"
            ],
            "answer": "C",
            "explanation": "W_hh是连接前一时间步隐藏状态到当前时间步隐藏状态的权重矩阵。",
            "explanation_zh": "W_hh connects the previous hidden state to the current hidden state."
        },
        {
            "question": "What is the advantage of bidirectional RNNs?",
            "question_zh": "双向RNN的优势是什么？",
            "options": [
                "A) Faster training",
                "B) Fewer parameters",
                "C) Can utilize future information",
                "D) Easier to converge"
            ],
            "answer": "C",
            "explanation": "Bidirectional RNNs can utilize both past and future information.",
            "explanation_zh": "双向RNN可以同时利用过去和未来的信息，提高模型性能。"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"问题{i}：{q['question_zh']}")
        for option in q['options']:
            print(f"  {option}")
        
        user_answer = input("Your answer (A/B/C/D): ").upper().strip()
        
        if user_answer == q['answer']:
            print("✓ Correct! 正确！")
            print(f"Explanation: {q['explanation']}")
            print(f"解释：{q['explanation_zh']}")
            score += 1
        else:
            print(f"✗ Incorrect. The correct answer is {q['answer']}")
            print(f"✗ 错误。正确答案是{q['answer']}")
            print(f"Explanation: {q['explanation']}")
            print(f"解释：{q['explanation_zh']}")
        print("-" * 50)
    
    print(f"Score: {score}/{len(questions)}")
    print(f"得分：{score}/{len(questions)}")

def fill_in_blanks():
    """Fill in the blank questions"""
    print("\n=== Fill in the Blanks ===")
    print("=== 填空题 ===\n")
    
    questions = [
        {
            "question": "The core idea of RNN is to share ______ at each time step, giving the network memory capability.",
            "question_zh": "RNN的核心思想是在每个时间步共享______，使网络具有记忆能力。",
            "answer": "参数/权重",
            "explanation": "RNN的核心思想是在每个时间步共享参数/权重，使网络具有记忆能力。",
            "explanation_zh": "RNN的核心思想是在每个时间步共享参数/权重，使网络具有记忆能力。"
        },
        {
            "question": "When processing the sequence \"hello\", RNN processes characters one by one, and the hidden state h_3 at the 3rd time step contains information from the first ______ characters.",
            "question_zh": "在处理序列\"hello\"时，RNN会逐个处理字符，第3个时间步的隐藏状态h_3包含了前______个字符的信息。",
            "answer": "3",
            "explanation": "在处理序列\"hello\"时，RNN会逐个处理字符，第3个时间步的隐藏状态h_3包含了前3个字符的信息。",
            "explanation_zh": "在处理序列\"hello\"时，RNN会逐个处理字符，第3个时间步的隐藏状态h_3包含了前3个字符的信息。"
        },
        {
            "question": "In Backpropagation Through Time (BPTT), gradients need to be backpropagated through ______ time steps.",
            "question_zh": "时间反向传播（BPTT）算法中，梯度需要通过______个时间步进行反向传播。",
            "answer": "多个/所有",
            "explanation": "时间反向传播（BPTT）算法中，梯度需要通过多个/所有时间步进行反向传播。",
            "explanation_zh": "时间反向传播（BPTT）算法中，梯度需要通过多个/所有时间步进行反向传播。"
        },
        {
            "question": "To alleviate the gradient explosion problem, a commonly used technique is ______.",
            "question_zh": "为了缓解梯度爆炸问题，常用的技术是______。",
            "answer": "梯度裁剪",
            "explanation": "为了缓解梯度爆炸问题，常用的技术是梯度裁剪。",
            "explanation_zh": "为了缓解梯度爆炸问题，常用的技术是梯度裁剪。"
        },
        {
            "question": "In language modeling tasks, the goal of RNN is to predict the ______ word in the sequence.",
            "question_zh": "在语言模型任务中，RNN的目标是预测序列中的______词。",
            "answer": "下一个",
            "explanation": "在语言模型任务中，RNN的目标是预测序列中的下一个词。",
            "explanation_zh": "在语言模型任务中，RNN的目标是预测序列中的下一个词。"
        }
    ]
    
    score = 0
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"问题{i}：{q['question_zh']}")
        
        user_answer = input("Your answer: ").strip().lower()
        correct_answers = [ans.strip().lower() for ans in q['answer'].split(',')]
        
        if any(ans in user_answer for ans in correct_answers):
            print("✓ Correct! 正确！")
            score += 1
        else:
            print(f"✗ Incorrect. The correct answer is: {q['answer']}")
            print(f"✗ 错误。正确答案是：{q['answer']}")
        
        print(f"Explanation: {q['explanation']}")
        print(f"解释：{q['explanation_zh']}")
        print("-" * 50)
    
    print(f"Score: {score}/{len(questions)}")
    print(f"得分：{score}/{len(questions)}")

def short_answer_questions():
    """Short answer questions"""
    print("\n=== Short Answer Questions ===")
    print("=== 简答题 ===\n")
    
    questions = [
        {
            "question": "Explain why traditional feedforward neural networks are not suitable for processing sequential data.",
            "question_zh": "解释为什么传统的前馈神经网络不适合处理序列数据？",
            "sample_answer": "Traditional feedforward networks have fixed input sizes and lack memory mechanisms, making them unsuitable for processing variable-length sequential data. They cannot model temporal dependencies in the input sequence, which is crucial for sequential tasks like language modeling, machine translation, or speech recognition.",
            "sample_answer_zh": "传统的前馈网络具有固定输入大小，缺乏记忆机制，使其不适合处理变长序列数据。它们无法在输入序列中建模时序依赖关系，这对于语言建模、机器翻译或语音识别等序列任务至关重要。"
        },
        {
            "question": "Describe in detail the causes of the vanishing gradient problem in RNNs and its impact on model performance.",
            "question_zh": "详细描述RNN中梯度消失问题的产生原因及其对模型性能的影响。",
            "sample_answer": "The vanishing gradient problem occurs in RNNs when gradients are backpropagated through time by repeatedly multiplying by weight matrices. If weights are small, gradients shrink exponentially with sequence length, making it hard to learn long-term dependencies. This severely impacts model performance, leading to poor learning of long-term dependencies and potentially causing the model to fail to capture important patterns in the input sequence. The vanishing gradient problem is particularly problematic for RNNs with recurrent connections, as it can lead to the loss of information about past inputs, making it difficult to maintain context and generate coherent outputs for long sequences.",
            "sample_answer_zh": "梯度消失问题发生在RNN中，当梯度通过时间反向传播时，需要重复乘以权重矩阵。如果权重很小，梯度会随序列长度呈指数收缩，使得难以学习长期依赖。这严重影响了模型性能，导致难以学习长距离依赖关系，并可能使模型无法捕捉输入序列中的重要模式。梯度消失问题是RNN中循环连接特别严重的问题，因为它可能导致关于过去输入的信息丢失，使得难以保持上下文并生成长序列的连贯输出。"
        },
        {
            "question": "Compare unidirectional RNNs and bidirectional RNNs, and explain their respective application scenarios.",
            "question_zh": "比较单向RNN和双向RNN的区别，并说明各自的适用场景。",
            "sample_answer": "Unidirectional RNNs only use past information, making them suitable for real-time tasks or when processing a single input sequence. Bidirectional RNNs use full sequence information, making them suitable for batch processing tasks or when the entire input sequence is available. Bidirectional RNNs can utilize both past and future information, which can be particularly beneficial for tasks like language modeling, where the model needs to consider the context of the entire sentence or paragraph. However, bidirectional RNNs require more computational resources and memory than unidirectional RNNs.",
            "sample_answer_zh": "单向RNN只能利用历史信息，使其适合实时任务或处理单个输入序列。双向RNN使用全序列信息，使其适合批处理任务或当整个输入序列可用时。双向RNN可以同时利用过去和未来的信息，这对于语言建模等任务特别有益，在这些任务中，模型需要考虑整个句子的上下文或段落。然而，双向RNN比单向RNN需要更多的计算资源和内存。"
        },
        {
            "question": "Explain the working principle of Truncated Backpropagation Through Time and its advantages.",
            "question_zh": "解释截断反向传播（Truncated BPTT）的工作原理及其优势。",
            "sample_answer": "Truncated Backpropagation Through Time (BPTT) is a technique used to handle long sequences by limiting the number of time steps that gradients are backpropagated through. This reduces computational complexity and helps prevent gradient problems. The truncated BPTT approach divides the sequence into smaller chunks, backpropagating gradients through each chunk independently. This allows for efficient training of long sequences while maintaining gradient flow. The main advantage is that it reduces the computational load and helps handle long sequences without encountering gradient explosion or vanishing gradient problems.",
            "sample_answer_zh": "截断反向传播（Truncated BPTT）是一种通过限制梯度反向传播的时间步数来处理长序列的技术。这减少了计算复杂度，并有助于防止梯度问题。截断BPTT方法将序列分成更小的块，独立地通过每个块反向传播梯度。这允许在保持梯度流动的同时，高效地训练长序列。主要优势是它减少了计算负载，并有助于处理长序列，而不会遇到梯度爆炸或梯度消失问题。"
        }
    ]
    
    for i, q in enumerate(questions, 1):
        print(f"Question {i}: {q['question']}")
        print(f"问题{i}：{q['question_zh']}")
        print("\nPlease provide your answer:")
        print("请提供您的答案：")
        
        user_answer = input().strip()
        
        print(f"\nSample Answer: {q['sample_answer']}")
        print(f"参考答案：{q['sample_answer_zh']}")
        print("-" * 50)

def programming_exercises():
    """Programming exercises"""
    print("\n=== Programming Exercises ===")
    print("=== 编程练习 ===\n")
    
    print("Exercise 1: Implement a simple RNN cell")
    print("练习1：实现一个简单的RNN单元")
    print()
    
    class SimpleRNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleRNN, self).__init__()
            self.hidden_size = hidden_size
            
            # 定义权重矩阵
            self.W_ih = nn.Linear(input_size, hidden_size)    # 输入到隐藏层
            self.W_hh = nn.Linear(hidden_size, hidden_size)   # 隐藏层到隐藏层
            self.W_ho = nn.Linear(hidden_size, output_size)   # 隐藏层到输出层
        
        def forward(self, input, hidden):
            """
            前向传播
            Args:
                input: 当前时间步输入 (batch_size, input_size)
                hidden: 前一时间步隐藏状态 (batch_size, hidden_size)
            Returns:
                output: 当前时间步输出 (batch_size, output_size)
                hidden: 当前时间步隐藏状态 (batch_size, hidden_size)
            """
            # 计算新的隐藏状态
            hidden = torch.tanh(self.W_ih(input) + self.W_hh(hidden))
            # 计算输出
            output = self.W_ho(hidden)
            return output, hidden
        
        def init_hidden(self, batch_size):
            """初始化隐藏状态"""
            return torch.zeros(batch_size, self.hidden_size)
    
    print("The SimpleRNN above demonstrates:")
    print("- Combining input and hidden state")
    print("- Computing new hidden state with tanh activation")
    print("- Producing output at each time step")
    print()
    print("上面的SimpleRNN演示了：")
    print("- 结合输入和隐藏状态")
    print("- 用tanh激活计算新隐藏状态")
    print("- 在每个时间步产生输出")
    print("-" * 50)
    
    print("Exercise 2: Implement LSTM cell")
    print("练习2：实现LSTM单元")
    print()
    
    class LSTMCell(nn.Module):
        def __init__(self, input_size, hidden_size):
            super(LSTMCell, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            
            # LSTM的四个门：遗忘门、输入门、候选值、输出门
            self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.candidate_gate = nn.Linear(input_size + hidden_size, hidden_size)
            self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        def forward(self, x, h_prev, c_prev):
            """
            LSTM前向传播
            Args:
                x: 当前输入 (batch_size, input_size)
                h_prev: 前一时间步隐藏状态 (batch_size, hidden_size)
                c_prev: 前一时间步细胞状态 (batch_size, hidden_size)
            Returns:
                h_new: 新的隐藏状态
                c_new: 新的细胞状态
            """
            # 拼接输入和前一隐藏状态
            combined = torch.cat([x, h_prev], dim=1)
            
            # 计算四个门
            f_t = torch.sigmoid(self.forget_gate(combined))      # 遗忘门
            i_t = torch.sigmoid(self.input_gate(combined))       # 输入门
            c_tilde = torch.tanh(self.candidate_gate(combined))  # 候选值
            o_t = torch.sigmoid(self.output_gate(combined))      # 输出门
            
            # 更新细胞状态
            c_new = f_t * c_prev + i_t * c_tilde
            
            # 计算新的隐藏状态
            h_new = o_t * torch.tanh(c_new)
            
            return h_new, c_new
    
    print("Key components of LSTM:")
    print("- Forget gate: decides what to discard")
    print("- Input gate: decides what new information to store")
    print("- Output gate: controls what parts of cell state to output")
    print("- Cell state: carries information across time steps")
    print()
    print("LSTM的关键组件：")
    print("- 遗忘门：决定丢弃什么")
    print("- 输入门：决定存储什么新信息")
    print("- 输出门：控制细胞状态的哪些部分输出")
    print("- 细胞状态：跨时间步携带信息")
    print("-" * 50)
    
    print("Exercise 3: Sequence prediction task")
    print("练习3：序列预测任务")
    print()
    
    def create_sequence_data(seq_length=10, num_sequences=1000):
        """
        创建序列预测数据：预测正弦波
        """
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # 随机起始点
            # Create sequence: sum of two numbers
            seq = torch.randint(0, 10, (seq_length,)).float()
            target = seq.sum().item()
            sequences.append(seq)
            targets.append(target)
        
        return torch.stack(sequences), torch.tensor(targets)
    
    class SequencePredictor(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super(SequencePredictor, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            
            # Forward propagate LSTM
            out, _ = self.lstm(x.unsqueeze(-1), (h0, c0))
            
            # Decode the hidden state of the last time step
            out = self.fc(out[:, -1, :])
            return out.squeeze()
    
    # Demo
    print("Example usage:")
    print("示例用法：")
    
    # Create sample data
    X, y = create_sequence_data(seq_length=5, num_sequences=10)
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Sample sequence: {X[0]}")
    print(f"Sample target (sum): {y[0]}")
    
    # Create model
    model = SequencePredictor()
    print(f"\nModel: {model}")
    
    # Forward pass
    with torch.no_grad():
        prediction = model(X[:1])
        print(f"Prediction: {prediction.item():.2f}")
        print(f"Actual: {y[0].item():.2f}")

def main():
    """Main function to run all quiz sections"""
    print("Welcome to Chapter 4 Quiz: Recurrent Neural Networks")
    print("欢迎来到第四章测验：循环神经网络")
    print("=" * 60)
    
    while True:
        print("\nChoose a section:")
        print("选择一个部分：")
        print("1. Multiple Choice Questions (选择题)")
        print("2. Fill in the Blanks (填空题)")
        print("3. Short Answer Questions (简答题)")
        print("4. Programming Exercises (编程练习)")
        print("5. Exit (退出)")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            multiple_choice_questions()
        elif choice == '2':
            fill_in_blanks()
        elif choice == '3':
            short_answer_questions()
        elif choice == '4':
            programming_exercises()
        elif choice == '5':
            print("Thank you for taking the quiz! 谢谢参与测验！")
            break
        else:
            print("Invalid choice. Please try again. 无效选择，请重试。")

if __name__ == "__main__":
    main() 