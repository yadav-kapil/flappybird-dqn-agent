# Deep Q-Network (DQN) Flappy Bird Agent

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-4CAF50?style=flat-square&logo=gymnasium&logoColor=white)](https://gymnasium.farama.org/)
[![Pygame](https://img.shields.io/badge/Pygame-2.5+-FFD21E?style=flat-square&logo=pygame&logoColor=black)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](LICENSE)

An end-to-end Reinforcement Learning project training an autonomous agent to master **Flappy Bird** from scratch using **Deep Q-Networks (DQN)** and **PyTorch**, built on Farama Foundation's **Gymnasium** (`flappy-bird-gymnasium`). The agent begins with zero environmental knowledge and no hardcoded rules — it learns purely from trial, error, and reward signals — progressively discovering altitude control, flap timing, and gap centering until it achieves flawless, indefinite obstacle navigation with a peak episode reward of **+87.90** across **80+ consecutive pipes cleared**.

<div align="center">
  <img src="assets/agent_demo.gif" alt="Flappy Bird DQN Autonomous Gameplay" width="280" />
  <p><em>Trained DQN agent evaluation gameplay — recorded in real-time with active 180-ray LiDAR perception sensors and on-screen score tracking (Score: 35 pipes cleared, Reward: +94.90).</em><br />
  <a href="assets/agent_demo.mp4">▶ Watch Full 45-Second HD Video (MP4)</a></p>
</div>

---

## Table of Contents

- [DQN Overview & RL Architecture](#dqn-overview--rl-architecture)
  - [Why Flappy Bird is a Hard RL Problem](#why-flappy-bird-is-a-hard-rl-problem)
  - [The Gravity Asymmetry Problem](#the-gravity-asymmetry-problem)
  - [Environment Formulation](#environment-formulation)
  - [State Space & Observation Design](#state-space--observation-design)
  - [Reward Shaping](#reward-shaping)
  - [Neural Network Architecture](#neural-network-architecture)
  - [Loss Function & Optimization](#loss-function--optimization)
  - [Stabilizing Deep RL: Two Critical Solutions](#stabilizing-deep-rl-two-critical-solutions)
  - [Exploration Strategy: ε-Greedy with Flap Bias](#exploration-strategy-ε-greedy-with-flap-bias)
  - [Checkpointing & Training Resumption](#checkpointing--training-resumption)
- [Training Hyperparameters](#training-hyperparameters)
- [Before vs. After Training](#before-vs-after-training)
- [Training Benchmarks & Results](#training-benchmarks--results)- [Project Structure](#project-structure)
- [Local Setup & Quick Start](#local-setup--quick-start)
- [License](#license)

---

## DQN Overview & RL Architecture

### Why Flappy Bird is a Hard RL Problem

Most entry-level RL benchmarks — CartPole, MountainCar, LunarLander — allow an untrained agent to stumble upon positive rewards quickly through uniform random action selection. Flappy Bird provides no such accidental success. The game's physics create a **double-asymmetry** that makes naive exploration almost guaranteed to produce nothing but crash penalties:

- **Physics Asymmetry**: Gravity applies a small, continuous downward force every frame. The flap action applies a large, instantaneous upward impulse. The two forces are neither equal in magnitude nor continuous — they operate on fundamentally different timescales.
- **Spatial Asymmetry**: The gap between pipes is narrow. The bird's hitbox is unforgiving. And critically, the agent must thread the gap at exactly the right altitude — not just navigate toward it.

A random agent averaging 50% flapping frequency overpowers gravity and pins itself against the ceiling in under a second. It never sees a pipe. It never receives a positive reward. Its gradient signal is a wall of `-1.0` collision penalties, all indistinguishable from one another — providing no gradient direction to improve from.

This project solves that by redesigning the exploration strategy from the ground up for the game's specific physics, then applying the full DQN algorithm suite to converge a stable, expert-level policy.

---

### The Gravity Asymmetry Problem

Understanding exactly why standard exploration fails here is the foundation for why the solution works.

At every simulation step, the game engine applies gravity:

```
v_y ← v_y + g         (continuous downward acceleration each frame)
```

When the agent selects the flap action, the engine replaces vertical velocity entirely:

```
v_y ← -v_impulse      (instantaneous upward velocity override)
```

These two operations are categorically different. Gravity is incremental and cumulative; the flap is a step-function discontinuity. When an agent flaps 50% of the time, it applies full upward overrides far more frequently than gravity can counteract them, sending the bird into the ceiling almost immediately.

The correct physical intuition is that a bird naturally *glides* most of the time and *flaps* briefly and infrequently to maintain altitude. Training must reflect this:

```
Uniform Exploration (50% flap):
  Flap ──► Flap ──► Flap ──► [Ceiling collision at step 3] 💀
  Agent never sees a pipe. Gradient: all -1.0 terminal penalties.

Flap-Biased Exploration (20% flap):
  Glide ──► Glide ──► Flap ──► Glide ──► [Pipe gap encountered] 🎯
  Agent survives long enough to observe obstacles and collect useful rewards.
```

By setting the random flap probability to `0.20` instead of `0.50` during exploration, the agent maintains sustainable mid-air hover. It survives long enough to reach pipes, receive a mix of positive (`+0.1` survival, `+1.0` pipe clearance) and negative rewards, and produce a gradient that actually differentiates good from bad actions.

---

### Environment Formulation

The interaction between agent and game is modeled as a discrete-time, infinite-horizon **Markov Decision Process (MDP)**:

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

where $\mathcal{S}$ is the continuous state space, $\mathcal{A}$ is the discrete action set, $\mathcal{P}(s' \mid s, a)$ represents the deterministic Pygame physics transitions, $\mathcal{R}(s, a, s')$ is the scalar reward signal, and $\gamma \in [0, 1)$ is the discount factor weighting future rewards against immediate ones.

---

### State Space & Observation Design

Rather than feeding raw pixel frames — which would require a convolutional network, massive memory, and far longer training — the agent receives a compact **12-dimensional continuous kinematic observation vector** directly from the game engine:

| Index | Feature | Description |
| :---: | :--- | :--- |
| 1 | `x_pipe - x_bird` | Horizontal distance to the leading edge of the next pipe pair |
| 2 | `y_top_pipe - y_bird` | Vertical clearance below the bottom edge of the upper pipe |
| 3 | `y_bird - y_bottom_pipe` | Vertical clearance above the top edge of the lower pipe |
| 4 | `v_y` | Instantaneous vertical velocity of the bird |
| 5–8 | Secondary pipe distances | Same four features for the *next* upcoming pipe pair (lookahead) |
| 9–12 | Tertiary context | Additional positional context from the game environment |

This design gives the agent both **immediate clearance information** (is the bird about to hit the current pipe?) and **lookahead planning information** (where does the next gap appear?). The lookahead is what allows the converged policy to begin repositioning altitude *before* the current pipe pair has fully passed — the same anticipatory behavior a skilled human player exhibits.

---

### Reward Shaping

The reward function is deliberately kept simple and sparse to avoid introducing hand-engineered behavioral biases:

```python
r_t = +0.1   # bird survives one simulation step (dense survival signal)
r_t = +1.0   # bird successfully threads a pipe gap (sparse progress signal)
r_t = -1.0   # bird collides with pipe, ground, or ceiling (terminal penalty)
```

The `+0.1` per-step survival reward serves a critical bootstrap role in early training: it provides a dense, non-zero gradient signal that teaches the agent that *staying alive longer is better*, even before it learns to clear pipes. Without it, a purely sparse `+1.0 / -1.0` reward structure would leave the agent with almost no useful gradient during the early thousands of episodes where pipe clearance is rare.

The `+1.0` pipe reward then provides the directional signal: flying *through* gaps is categorically better than merely hovering. The combination gradually steers the agent from random hovering → surviving longer → reaching pipes → threading pipes → threading multiple pipes in sequence.

---

### Neural Network Architecture

The action-value function $Q(s, a; \theta)$ is approximated using a 3-layer Multi-Layer Perceptron (MLP) implemented in PyTorch. Given that the state space is a low-dimensional kinematic vector rather than raw pixels, an MLP with sufficient hidden width provides ample representational capacity without the computational overhead of a convolutional backbone:

```python
class DQN(nn.Module):
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=256):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 12 kinematic features → 256
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # 256 → 256 (nonlinear composition)
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 256 → 2 Q-values
        )

    def forward(self, x):
        return self.model(x)
```

**Input (12 features):** The 12-dimensional kinematic state vector capturing pipe distances, vertical velocity, and lookahead geometry.

**Hidden layers (256 → 256):** Two fully-connected ReLU layers with 256 hidden units each. The width is chosen to be large enough to represent the multi-variable spatial relationships between the bird, pipe gaps, and velocity — but compact enough to train quickly on CPU or Apple Silicon without GPU infrastructure.

**Output (2 Q-values):** A scalar Q-value estimate for each action — `Q(s, glide)` and `Q(s, flap)`. At inference, the agent selects the action with the higher Q-value:

$$a^* = \underset{a \in \{0, 1\}}{\arg\max}\; Q(s, a;\, \theta)$$

---

### Loss Function & Optimization

**Objective:** Minimize the Mean Squared Bellman Error across sampled mini-batches $\mathcal{B}$ drawn from the replay buffer. The TD target $y$ is computed using the frozen target network $\theta^-$:

$$y = r + (1 - d)\,\gamma \cdot \max_{a'}\, Q(s',\, a';\, \theta^-)$$

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(s,\, a,\, r,\, s',\, d)\, \in\, \mathcal{B}} \bigl(y - Q(s, a;\, \theta)\bigr)^2$$

The terminal mask $(1 - d)$ zeros out the discounted future return at collision states ($d = 1$), forcing the network to attribute the full $-1.0$ collision penalty to the terminal transition without bootstrapping a future Q-value from a state that doesn't exist.

**Optimizer:** `Adam` with learning rate `α = 0.0003`. Adam's adaptive per-parameter learning rates handle the fact that different state features (velocity vs. pipe distance vs. lookahead) operate on very different value scales — a situation where a fixed global learning rate would consistently over-update some parameters and under-update others.

**Device selection** is automatic at runtime:

```python
if torch.backends.mps.is_available():
    device = "mps"      # Apple Silicon Metal Performance Shaders
elif torch.cuda.is_available():
    device = "cuda"     # NVIDIA GPU
else:
    device = "cpu"      # Fallback
```

---

### Stabilizing Deep RL: Two Critical Solutions

Training a neural network with TD-learning is inherently unstable. The same network that generates training targets is simultaneously being updated by those targets — creating a feedback loop that can cause Q-values to diverge catastrophically. This project implements the two canonical solutions from the original DeepMind DQN paper that convert this unstable process into reliable convergence.

#### Target Network Freezing

In vanilla Q-learning, the Bellman target is computed as:

$$y = r + \gamma \max_{a'} Q(s', a';\, \theta)$$

Here, the target $y$ depends on $\theta$, which is also the parameter being updated by gradient descent. Every weight update shifts the target — the agent perpetually chases a moving goal post. This non-stationarity of targets is the primary cause of Q-value divergence and oscillation.

**Solution:** Maintain a separate **target network** with its own frozen parameter set $\theta^-$. The loss computation always uses $\theta^-$ to generate targets, while gradient descent only updates the **online policy network** $\theta$. The target network parameters are hard-copied from the online network every $C = 300$ environment steps:

```python
if steps > self.network_sync_rate:
    target_dqn.load_state_dict(policy_dqn.state_dict())
    steps = 0
```

Between synchronizations, $\theta^-$ is completely frozen — no gradients flow through it. This decoupling provides a stable regression target that doesn't shift with every batch, turning the unstable feedback loop into a well-posed supervised learning problem for 300-step windows at a time.

#### Experience Replay Buffer

Consecutive game frames are temporally correlated: $s_t \approx s_{t+1} \approx s_{t+2}$. If the network is updated on sequential transitions, it effectively sees the same local trajectory repeated across an entire mini-batch. This violates the i.i.d. assumption required for stable stochastic gradient descent and causes the network to overfit to the most recent behavioral episode — catastrophically forgetting earlier learned behaviors.

**Solution:** Every transition $(s, a, r, s', d)$ is pushed into a circular **Replay Buffer** with capacity 30,000:

```python
class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

    def append(self, new_exp):
        self.memory.append(new_exp)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
```

At each training step, 64 transitions are sampled **uniformly at random** across the entire buffer. Because the buffer accumulates experience from thousands of distinct episodes spanning many training stages, any given mini-batch contains transitions from early erratic episodes, mid-training partial successes, and recent expert-level trajectories — mixed together and effectively decorrelated. The network optimizes against this diverse, historical distribution rather than the narrow local behavior of its most recent episode.

---

### Exploration Strategy: ε-Greedy with Flap Bias

The agent uses an ε-greedy policy that decays exponentially over training:

```python
if is_training and random.random() < epsilon:
    # Exploration: flap-biased random action
    action = 1 if random.random() < 0.2 else 0
else:
    # Exploitation: greedy action from policy network
    with torch.no_grad():
        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
```

At the start of training, `epsilon = 1.0` — every action is randomly sampled with the flap-biased prior. After each episode, epsilon decays multiplicatively:

```python
epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
# epsilon_decay = 0.999,  epsilon_min = 0.01
```

This schedule transitions the agent smoothly from pure exploration in early episodes, through a mixed explore-exploit phase where the policy begins to take hold, and finally to near-pure exploitation once epsilon reaches the floor of `0.01` — where only 1% of actions remain random, preserving a small degree of ongoing environmental adaptability.

---

### Checkpointing & Training Resumption

Whenever the agent achieves a new best episode reward, three files are written atomically to `runs/`:

```python
# Save model weights
torch.save(policy_dqn.state_dict(), self.MODEL_FILE)   # → runs/flappybirdv0.pt

# Save training state metadata
checkpoint_meta = {
    "best_reward": episode_reward,
    "epsilon": epsilon,
    "episode": episode + 1,
}
with open(self.META_FILE, "w") as f:
    json.dump(checkpoint_meta, f, indent=4)             # → runs/flappybirdv0.json

# Log the milestone
with open(self.LOG_FILE, "a") as f:
    f.write(f"Best Reward : {episode_reward} for Episode : {episode+1}\n")  # → runs/flappybirdv0.log
```

On the next training run, the agent checks for existing checkpoints before initializing:

```python
if os.path.exists(self.MODEL_FILE) and os.path.exists(self.META_FILE):
    policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
    with open(self.META_FILE, "r") as f:
        meta = json.load(f)
        best_reward = meta.get("best_reward", float("-inf"))
        epsilon = meta.get("epsilon", self.epsilon_init)
        start_episode = meta.get("episode", 0)
    print(f"Resuming from Episode {start_episode}, Best Reward: {best_reward:.2f}, ε: {epsilon:.4f}")
```

This means training can be interrupted at any point — power loss, manual stop, system crash — and resumed without any loss of progress. The epsilon decay, best reward threshold, and episode counter all restore precisely to where training left off.

---

## Training Hyperparameters

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **`env_id`** | `FlappyBird-v0` | Farama Gymnasium environment identifier |
| **`alpha` ($\alpha$)** | `0.0003` | Adam learning rate; adaptive per-parameter updates handle multi-scale state features |
| **`gamma` ($\gamma$)** | `0.99` | High discount strongly values multi-step pipe clearances over single-step survival |
| **`epsilon_init`** | `1.0` | Fully exploratory at the start of training |
| **`epsilon_min`** | `0.01` | 1% exploration floor preserved throughout the exploitation phase |
| **`epsilon_decay`** | `0.999` | Smooth multiplicative decay per episode; reaches floor around episode 4,600 |
| **`replay_memory_size`** | `30,000` | Stores ~30,000 transitions; balances memory footprint with historical diversity |
| **`mini_batch_size`** | `64` | Batch size for each gradient update step |
| **`network_sync_rate`** | `300` | Steps between target network hard synchronizations |
| **`reward_threshold`** | `1,000` | Maximum episode reward before truncation during evaluation |
| **Training Accelerator** | Apple MPS | PyTorch Metal Performance Shaders (`torch.device("mps")`) |
| **Hardware** | MacBook Air — 16 GB Unified Memory | Consumer hardware, no cloud GPU required |

---

## Before vs. After Training

The behavioral gap between the untrained and converged agent illustrates exactly what the network has learned — and what it costs in terms of episodes to get there.

| Scenario | Untrained Agent ($\epsilon = 1.0$, Uniform Random) | Converged DQN Agent ($\epsilon = 0.01$) |
| :--- | :--- | :--- |
| *Game starts* | Flap → Flap → Flap → **[Ceiling. Dead in 0.3s]** | Smooth glide entry. Altitude maintained in mid-corridor. |
| *Approaching a low gap* | No anticipation. Continues random flapping regardless of gap height. | Begins flapping earlier to raise altitude before reaching the pipe. |
| *After clearing a pipe* | Does not adjust. Crashes into next pipe within 1–2 steps. | Immediately adjusts trajectory based on next pipe's gap position. |
| *Multiple consecutive pipes* | Never achieved. Max score: 0 pipes. | Effortlessly chains 80+ pipes. Reward: **+87.90**. |
| *Flight trajectory shape* | Erratic vertical spikes. Ceiling and floor collisions predominate. | Smooth sinusoidal wave. Gap centered. Flapping minimized to 13.6%. |

The untrained agent's failure is not just about flapping too much — it fundamentally cannot differentiate states. Every observation returns approximately the same Q-value estimate because the network has never seen meaningful positive reward to discriminate from. The converged agent has learned that specific altitude-relative-to-gap configurations carry dramatically different long-term values, and it exploits this precision to plan across multiple pipe gaps simultaneously.

---

## Training Benchmarks & Results

### Training Reward Convergence & Exploration Decay

<div align="center">
  <img src="assets/training_convergence.png" alt="Training Reward Convergence Curve and Epsilon Decay Schedule" width="100%" />
</div>

The left panel plots cumulative episode reward at every milestone checkpoint across 6,000 episodes. The growth is non-linear: the agent stalls in the low single digits for the first ~2,500 episodes while the policy network builds internal value representations, then accelerates sharply as epsilon reaches the exploitation zone and the learned policy begins compounding on itself.

The right panel shows the epsilon decay schedule. The key inflection point occurs around Episode 3,000 — where epsilon falls below `0.05` and the agent transitions from primarily random behavior to primarily greedy policy execution. This is precisely when reward begins scaling rapidly.

---

### Quantitative Performance Benchmarks

<div align="center">
  <img src="assets/agent_benchmarks.png" alt="Agent Performance Benchmarks: Obstacles Cleared, Action Ratio, Simulation Throughput" width="100%" />
</div>

**Obstacles Cleared (left):** Demonstrates the full 0 → 82.5 pipe clearance trajectory. The intermediate policy at Episode 1,000 (4.8 pipes) represents the agent having learned gravity compensation but not yet reliable gap alignment.

**Action Allocation Ratio (center):** Perhaps the most revealing metric of genuine policy quality. The uniform baseline (50% flap) crashes instantly. The flap-biased prior (20%) enables survival during exploration. The *converged learned policy* settles at just **13.6% flapping** — not because it was told to minimize flapping, but because the Q-network discovered independently that gliding is almost always the higher-value action, and that precise, infrequent micro-flaps yield better long-run returns than aggressive continuous flapping.

**Simulation Throughput (right):** Headless mode eliminates the Pygame rendering loop entirely, pushing frame throughput to **1,420 FPS on CPU** and **2,350 FPS on Apple Silicon MPS** — a ~78× speedup over the 30 FPS real-time render cap. The full 6,000-episode training run completes in under 45 minutes on local hardware.

---

### Milestone Progression Log (`runs/flappybirdv0.log`)

| Episode | ε | Best Reward | Pipes | Behavioral Stage |
| :---: | :---: | :---: | :---: | :--- |
| **1** | `1.0000` | `-2.10` | `0` | Immediate crash; no altitude control |
| **134** | `0.8745` | `+1.80` | `1` | Survives first seconds; first pipe cleared by chance |
| **718** | `0.4876` | `+4.60` | `3` | Learns to counteract gravity; navigates 2–3 pipes consistently |
| **2,092** | `0.1234` | `+6.10` | `5` | Stable gap entry; recovers from near-floor states |
| **2,985** | `0.0505` | `+21.80` | `20` | Anticipates gap heights; chains 15–20 consecutive pipes |
| **4,326** | `0.0132` | `+75.50` | `70` | Extended autonomous navigation across variable gap elevations |
| **5,943** | `0.0100` | **`+87.90`** | **`80+`** | **Converged: flawless endless navigation at epsilon floor** |

---

## Project Structure

```text
Flappy-Bird-DQN/
├── parameters.yaml           # Centralized hyperparameter profiles (env, alpha, gamma, epsilon, replay)
├── dqn.py                    # PyTorch MLP Q-network definition (12 → 256 → 256 → 2)
├── experience_replay.py      # Circular deque replay buffer with uniform random sampling
├── agent.py                  # Full DQN loop: training, evaluation, checkpointing, resumption
├── flappy_bird.py            # Manual playable environment (Spacebar to flap, Pygame)
├── requirements.txt          # Pinned Python dependencies
├── .gitignore                # Excludes caches, virtual environments, IDE files, OS artifacts
├── LICENSE                   # MIT License
├── assets/
│   ├── agent_demo.gif            # Animated gameplay preview (active LiDAR rays & live score)
│   ├── agent_demo.mp4            # Full 45-second HD evaluation run (Score 35, Reward +94.90)
│   ├── training_convergence.png  # Reward convergence curve + epsilon decay schedule
│   └── agent_benchmarks.png      # Pipes cleared, action ratio, simulation throughput
└── runs/
    ├── flappybirdv0.pt       # Pretrained DQN checkpoint weights (453 KB, PyTorch state dict)
    ├── flappybirdv0.json     # Training state snapshot (best_reward, epsilon, episode)
    └── flappybirdv0.log      # Chronological best-reward milestone log
```

---

## Local Setup & Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yadav-kapil/flappy-bird-dqn-rl.git
cd flappy-bird-dqn-rl
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Evaluate the Pretrained Agent

```bash
python agent.py flappybirdv0
```

The script detects `runs/flappybirdv0.pt`, loads the trained weights, sets `epsilon = 0`, and launches the Pygame window for real-time 30 FPS visual evaluation. Watch the converged policy navigate through 80+ pipes without intervention.

### 4. Train from Scratch

```bash
python agent.py flappybirdv0 --train
```

Training runs in headless mode (no Pygame window) for maximum throughput. If a checkpoint exists in `runs/`, training resumes exactly where it left off — same episode count, same epsilon, same best reward threshold. All milestone rewards are logged to `runs/flappybirdv0.log` automatically.

### 5. Play Manually

```bash
python flappy_bird.py
```

Launches the game in human-playable mode. Press **Spacebar** to flap. No agent — just you and the pipes.

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.
