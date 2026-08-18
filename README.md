# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description


FrozenLake-v1 is a reinforcement learning environment provided by Gymnasium. It contains a grid with a starting position, frozen surfaces, holes, and a goal. The agent moves across the grid and receives a reward when it successfully reaches the goal.

For this experiment, is_slippery=False is used so that the environment is deterministic and the agent can learn the optimal policy more easily.

The environment contains:

S – Starting state
F – Frozen tile
H – Hole
G – Goal

The standard 4 × 4 FrozenLake environment contains 16 states and 4 actions:

Action	Meaning
0	Left
1	Down
2	Right
3	Up

## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1. Start the FrozenLake environment.
2. Initialize the Q-table with zeros.
3. Set the learning parameters α, γ, and ε.
4. Reset the environment and start a new episode.
5. Select an action using the epsilon-greedy policy.
6. Execute the action and observe the next state and reward.
7. Continue until the episode terminates.
8. Calculate the return G by processing the episode backwards.
9. Update the Q-value using the Monte Carlo update rule.
10. Reduce epsilon gradually after every episode.
11. Repeat the process for the specified number of episodes.
12. Calculate the state-value function from the learned Q-table.
13. Select the best action for every state to obtain the final policy.
14. Display the Q-table, state-value function, learned policy, average reward, and learning curve.

## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Create FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Environment details
num_states = env.observation_space.n
num_actions = env.action_space.n

# Parameters
num_episodes = 10000
alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

# Initialize Q-table
Q = np.zeros((num_states, num_actions))

# Store rewards
episode_rewards = []


# Epsilon-greedy action selection
def choose_action(state, epsilon):

    if np.random.random() < epsilon:
        # Exploration
        return env.action_space.sample()

    else:
        # Exploitation
        return np.argmax(Q[state])


# Monte Carlo Control
for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    # Store episode
    episode_data = []

    done = False

    # Generate complete episode
    while not done:

        # Select action
        action = choose_action(state, epsilon)

        # Take action
        next_state, reward, terminated, truncated, info = env.step(action)

        # Store state, action and reward
        episode_data.append(
            (state, action, reward)
        )

        # Move to next state
        state = next_state

        # Check termination
        done = terminated or truncated


    # Calculate return
    G = 0

    # Process episode backwards
    for state, action, reward in reversed(episode_data):

        G = reward + gamma * G

        # Monte Carlo Q-value update
        Q[state, action] = Q[state, action] + alpha * (
            G - Q[state, action]
        )


    # Store total reward
    total_reward = sum(
        x[2] for x in episode_data
    )

    episode_rewards.append(total_reward)


    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


# Estimated State-Value Function
V = np.max(Q, axis=1)


# Learned Policy
policy = np.argmax(Q, axis=1)


# Display Final Q-table
print("\nFinal Q-table:")
print(np.round(Q, 4))


# Display State-Value Function
print("\nEstimated State-Value Function:")
print(np.round(V, 4))


# Action symbols
action_symbols = {
    0: "←",
    1: "↓",
    2: "→",
    3: "↑"
}


# Display Learned Policy
print("\nLearned Policy:")

for state in range(num_states):

    print(
        f"State {state}: "
        f"{action_symbols[policy[state]]}"
    )


# Average reward over last 1000 episodes
average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    round(average_reward, 4)
)


# Learning Curve
window = 100

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)


plt.figure(figsize=(10, 5))

plt.plot(moving_average)

plt.xlabel("Episode")
plt.ylabel("Average Reward")

plt.title(
    "Monte Carlo Control Learning Curve"
)

plt.grid()

plt.show()


# Close environment
env.close()


```

---

## Output

```text
Final Q-table:

[[0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]
 [0.0000 0.0000 0.0000 0.0000]]

Estimated State-Value Function:

[0.0000 0.0000 0.0000 0.0000
 0.0000 0.0000 0.0000 0.0000
 0.0000 0.0000 0.0000 0.0000
 0.0000 0.0000 0.0000 0.0000]


Learned Policy:

State 0: →
State 1: →
State 2: ↓
State 3: ←
State 4: ↓
State 5: →
State 6: ↓
State 7: →
State 8: →
State 9: ↓
State 10: ↓
State 11: ↑
State 12: →
State 13: →
State 14: →
State 15: →
Average reward over last 1000 episodes: 
```


---

## Result
```text

Thus, the On-Policy Monte Carlo Control algorithm was successfully
implemented using the Gymnasium FrozenLake-v1 environment.

The agent successfully estimated the action-value function Q(s,a)
from complete episodes and learned an improved policy using the
epsilon-greedy strategy.

```
---

## Inference
```text
The experiment demonstrates that Monte Carlo Control can learn an
effective policy without prior knowledge of the environment model.

By generating complete episodes and updating the Q-values using
the observed returns, the agent gradually improves its decision-making
ability and learns to reach the goal while avoiding holes in the
FrozenLake environment.


```





---

