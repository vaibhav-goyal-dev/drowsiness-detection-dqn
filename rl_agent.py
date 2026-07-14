"""
rl_agent.py
-----------
DQN decision agent. Key design principle:

  The CNN is the expert. The RL agent's job is NOT to override the CNN —
  it is to decide WHEN to trigger an alarm given the CNN's confidence
  plus contextual signals (PERCLOS, MAR).

  The RL agent starts conservative (epsilon=0.6) and gradually trusts
  itself as it accumulates experience. Until the buffer is large enough,
  it simply follows the CNN probability directly.

State  [5]: [P_drowsy, EAR, MAR, PERCLOS, consecutive_drowsy_frames_norm]
Actions [2]: 0=ALERT, 1=DROWSY
Reward: -10 for missed drowsy, -2 for false alarm, +10 correct drowsy, +2 correct alert
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from typing import Optional, Tuple


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN that augments CNN decisions rather than replacing them.

    Critical fix vs previous version:
      - CNN probability is part of state, not bypassed
      - min_buffer_before_train: RL doesn't update until it has enough data
      - When epsilon is high, falls back to CNN threshold directly
      - Reward is asymmetric: missing drowsy (-10) >> false alarm (-2)
    """

    def __init__(self, config: dict):
        self.sd   = config['state_dim']
        self.ad   = config['action_dim']
        self.gamma       = config['gamma']
        self.bs          = config['batch_size']
        self.target_freq = config['target_update_freq']
        self.min_buf     = config.get('min_buffer_before_train', 500)

        self.eps     = config['epsilon_start']
        self.eps_end = config['epsilon_end']
        self.eps_dec = config['epsilon_decay']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.online = QNet(self.sd, self.ad, config['hidden_dim']).to(self.device)
        self.target = QNet(self.sd, self.ad, config['hidden_dim']).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt    = optim.Adam(self.online.parameters(), lr=config['learning_rate'])
        self.buf    = ReplayBuffer(config['replay_buffer_size'])
        self.steps  = 0

        # High-Precision Balanced Reward System
        self.R_correct_drowsy = 15.0
        self.R_missed_drowsy  = -20.0
        self.R_correct_alert  = 15.0
        self.R_false_alarm    = -20.0

    # ── Action ────────────────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      p_drowsy: float,
                      cnn_threshold: float = 0.55) -> int:
        """
        Select action. Key behavior:
          - If buffer too small → trust CNN directly (no RL noise)
          - If exploring (epsilon) → use CNN probability as fallback
          - If exploiting → use Q-network
        """
        # Before enough data: just follow CNN
        if len(self.buf) < self.min_buf:
            return 1 if p_drowsy >= cnn_threshold else 0

        # Epsilon-greedy: fallback is CNN, not random
        if random.random() < self.eps:
            return 1 if p_drowsy >= cnn_threshold else 0

        s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.online(s)
        return int(q.argmax(dim=1).item())

    # ── Reward ────────────────────────────────────────────────────────────────

    def compute_reward(self, action: int, p_drowsy: float) -> float:
        """
        Proxy reward using CNN confidence as ground truth signal.
        The CNN has seen millions of training images — treat it as oracle.
        """
        if action == 1:   # said DROWSY
            return (self.R_correct_drowsy * p_drowsy +
                    self.R_false_alarm   * (1 - p_drowsy))
        else:             # said ALERT
            return (self.R_correct_alert  * (1 - p_drowsy) +
                    self.R_missed_drowsy  * p_drowsy)

    # ── Learning ──────────────────────────────────────────────────────────────

    def store(self, s, a, r, s2, done):
        self.buf.push(s, a, r, s2, done)

    def update(self) -> Optional[float]:
        if len(self.buf) < max(self.bs, self.min_buf):
            return None

        s, a, r, s2, d = self.buf.sample(self.bs)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)
        r  = torch.FloatTensor(r).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d  = torch.FloatTensor(d).to(self.device)

        # Double DQN
        curr_q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.online(s2).argmax(dim=1)
            next_q = self.target(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            tgt    = r + self.gamma * next_q * (1 - d)

        loss = nn.SmoothL1Loss()(curr_q, tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.opt.step()

        # Decay epsilon
        self.eps = max(self.eps_end, self.eps * self.eps_dec)

        # Sync target
        self.steps += 1
        if self.steps % self.target_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item()

    # ── Save / load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'online':  self.online.state_dict(),
            'target':  self.target.state_dict(),
            'opt':     self.opt.state_dict(),
            'eps':     self.eps,
            'steps':   self.steps,
        }, path)
        print(f"[RL] Saved → {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[RL] No weights at {path} — starting fresh.")
            return
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck['online'])
        self.target.load_state_dict(ck['target'])
        self.opt.load_state_dict(ck['opt'])
        self.eps   = ck.get('eps',   self.eps_end)
        self.steps = ck.get('steps', 0)
        print(f"[RL] Loaded {path}  steps={self.steps}  eps={self.eps:.3f}")
