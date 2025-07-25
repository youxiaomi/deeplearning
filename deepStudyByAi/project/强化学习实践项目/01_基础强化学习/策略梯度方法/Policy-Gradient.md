## Policy Gradient Methods

Policy Gradient methods are a class of reinforcement learning algorithms that directly learn a policy function that maps states to actions, without necessarily learning a value function. The goal is to optimize the policy directly by estimating the gradient of the expected reward with respect to the policy parameters.

策略梯度方法是一类强化学习算法，它直接学习将状态映射到动作的策略函数，而不一定学习价值函数。目标是通过估计预期奖励相对于策略参数的梯度来直接优化策略。

### Core Concepts

#### 1. Policy (策略)

The policy, denoted as \( \pi(a|s; \theta) \), is a function that tells the agent what action to take in a given state. In policy gradient methods, this policy is often parameterized by a set of parameters \( \theta \) (e.g., weights of a neural network).

策略，表示为 \( \pi(a|s; \theta) \)，是一个告诉智能体在给定状态下采取什么动作的函数。在策略梯度方法中，这个策略通常由一组参数 \( \theta \)（例如，神经网络的权重）参数化。

**Example (例子):**
Imagine a robot learning to walk. Its policy would be a function that takes its current sensor readings (state) and outputs the motor commands (actions) to move its legs. The parameters \( \theta \) would tune how the robot interprets sensor data and executes movements.

想象一个正在学习走路的机器人。它的策略将是一个函数，接收其当前的传感器读数（状态）并输出控制其腿部移动的电机命令（动作）。参数 \( \theta \) 将调整机器人如何解释传感器数据和执行动作。

#### 2. Expected Return (预期回报)

The objective of policy gradient methods is to maximize the expected return (or expected cumulative reward). The expected return \( J(\theta) \) is the sum of rewards an agent expects to receive over an entire episode, starting from a given state, when following a policy \( \pi_\theta \).

策略梯度方法的目标是最大化预期回报（或预期累积奖励）。预期回报 \( J(\theta) \) 是智能体从给定状态开始，遵循策略 \( \pi_\theta \) 时，在一个完整回合中期望获得的总奖励。

\( J(\theta) = E_{\tau \sim \pi_\theta} [\sum_{t=0}^{T} r_t] \)

Where \( \tau \) is a trajectory (sequence of states, actions, and rewards).
其中 \( \tau \) 是一个轨迹（状态、动作和奖励的序列）。

#### 3. Policy Gradient Theorem (策略梯度定理)

The policy gradient theorem provides an analytical expression for the gradient of the expected return with respect to the policy parameters. This allows us to use gradient ascent to optimize the policy.

策略梯度定理提供了预期回报相对于策略参数的梯度的解析表达式。这使得我们能够使用梯度上升来优化策略。

For episodic tasks (tasks that end after a finite number of steps):
对于回合制任务（在有限步数后结束的任务）：

\( \nabla_{\theta} J(\theta) = E_{\tau \sim \pi_\theta} [\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) G_t] \)

Where:
其中：

*   \( \nabla_{\theta} J(\theta) \): The gradient of the expected return.
    预期回报的梯度。
*   \( \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) \): The gradient of the log-probability of taking action \( a_t \) in state \( s_t \) under policy \( \pi_{\theta} \). This is often called the "score function".
    在策略 \( \pi_{\theta} \) 下，在状态 \( s_t \) 中采取动作 \( a_t \) 的对数概率的梯度。这通常被称为“分数函数”。
*   \( G_t \): The cumulative discounted reward (return) from time step \( t \) onwards.
    从时间步 \( t \) 开始的累积折扣奖励（回报）。

### REINFORCE Algorithm

REINFORCE (also known as Monte Carlo Policy Gradient) is one of the simplest policy gradient algorithms. It works by running an agent for a full episode, collecting all the rewards, and then updating the policy parameters based on the observed returns.

REINFORCE（也称为蒙特卡洛策略梯度）是最简单的策略梯度算法之一。它通过让智能体运行一个完整的回合，收集所有奖励，然后根据观察到的回报更新策略参数。

**Algorithm Steps:**
**算法步骤：**

1.  **Initialize Policy Network:** Define a neural network that outputs probabilities for each action in a given state. Initialize its parameters \( \theta \).
    **初始化策略网络：** 定义一个神经网络，在给定状态下为每个动作输出概率。初始化其参数 \( \theta \)。
2.  **Generate an Episode:** Run the policy \( \pi_\theta \) for one complete episode, collecting the sequence of states, actions, and rewards: \( s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T, a_T, r_{T+1} \).
    **生成一个回合：** 运行策略 \( \pi_\theta \) 完成一个完整的回合，收集状态、动作和奖励的序列： \( s_0, a_0, r_1, s_1, a_1, r_2, ..., s_T, a_T, r_{T+1} \)。
3.  **Calculate Returns:** For each time step \( t \), calculate the cumulative discounted return \( G_t \) from that point onwards:
    **计算回报：** 对于每个时间步 \( t \)，计算从该点开始的累积折扣回报 \( G_t \)：
    \( G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1} \)
4.  **Compute Loss (or Gradient):** The objective is to maximize the expected return. In PyTorch, we typically minimize the negative of this objective. The loss for REINFORCE is often formulated as:
    **计算损失（或梯度）：** 目标是最大化预期回报。在 PyTorch 中，我们通常最小化该目标的负值。REINFORCE 的损失通常表示为：
    \( L(\theta) = - \sum_{t=0}^{T} \log \pi_{\theta}(a_t|s_t) G_t \)
    This loss function, when minimized, performs gradient ascent on \( J(\theta) \).
    这个损失函数在最小化时，对 \( J(\theta) \) 执行梯度上升。
5.  **Update Policy Parameters:** Use an optimizer (e.g., Adam) to update the policy parameters \( \theta \) using the computed gradients.
    **更新策略参数：** 使用优化器（例如，Adam）根据计算出的梯度更新策略参数 \( \theta \)。
6.  **Repeat:** Repeat steps 2-5 for many episodes until the policy converges or reaches a desired performance level.
    **重复：** 重复步骤 2-5 多个回合，直到策略收敛或达到所需的性能水平。

### Advantages and Disadvantages of Policy Gradient

**Advantages (优点):**

*   **Can learn stochastic policies:** Policy gradients can learn policies that are inherently probabilistic, which is useful in environments where an optimal deterministic policy might not exist or where exploration is crucial.
    **可以学习随机策略：** 策略梯度可以学习本质上是概率性的策略，这在不存在最优确定性策略或探索至关重要的环境中非常有用。
*   **Handles continuous action spaces:** Policy gradients can naturally handle continuous action spaces by parameterizing the policy with a distribution (e.g., Gaussian distribution), whereas value-based methods often struggle with this.
    **处理连续动作空间：** 策略梯度可以通过用分布（例如，高斯分布）参数化策略来自然地处理连续动作空间，而基于价值的方法通常在这方面遇到困难。
*   **Directly optimize the goal:** They directly optimize the objective of maximizing cumulative reward, which can sometimes lead to more stable learning compared to value-based methods that optimize an indirect objective (Q-values).
    **直接优化目标：** 它们直接优化最大化累积奖励的目标，与优化间接目标（Q 值）的基于价值的方法相比，这有时可以导致更稳定的学习。

**Disadvantages (缺点):**

*   **High variance:** The gradient estimates can have high variance, leading to slow convergence or unstable training. This is a common issue with Monte Carlo methods because they rely on full episode returns.
    **高方差：** 梯度估计可能具有高方差，导致收敛缓慢或训练不稳定。这是蒙特卡洛方法的常见问题，因为它们依赖于完整的回合回报。
*   **Requires full episodes:** REINFORCE specifically requires waiting until the end of an episode to calculate returns, which can be inefficient for long episodes.
    **需要完整的回合：** REINFORCE 特别需要等到回合结束才能计算回报，这对于长的回合来说效率可能很低。

### Conclusion

Policy Gradient methods, particularly REINFORCE, offer a direct approach to optimizing control policies in reinforcement learning. While they can suffer from high variance, they form the foundation for more advanced and stable algorithms like Actor-Critic methods, which combine the strengths of both policy-based and value-based approaches.

策略梯度方法，特别是 REINFORCE，为强化学习中优化控制策略提供了一种直接的方法。尽管它们可能受到高方差的影响，但它们为更高级和更稳定的算法（如 Actor-Critic 方法）奠定了基础，这些方法结合了基于策略和基于价值的方法的优点。 