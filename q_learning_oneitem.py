import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# -----------------------------
# 1) Configuration (inputs you provide)
# -----------------------------
@dataclass
class InventoryQLConfig:
    '''parameters'''
    # Costs (paper's reward is cost; we will MINIMIZE cost by maximizing -cost)
    h: float = 1.0      # holding cost per unit per period
    b: float = 5.0      # backorder cost per unit per period
    O: float = 20.0     # fixed cost per order (single order issued)

    # Fixed order quantity (policy design in the paper)
    OQ: int = 10        # fixed replenishment quantity added when action=1

    # RL hyperparameters (paper values)
    alpha: float = 0.5  # learning rate
    gamma: float = 0.5  # discount factor

    # Training setup (paper mentions 1000 days, 1000 iterations)
    episode_length: int = 1000
    num_episodes: int = 1000

    # Exploration (paper doesn't specify; we need one to learn)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99  # multiplicative decay per episode

    # Inventory position bounds for tabular Q
    # (You must bound IP to keep the Q-table finite.)
    ip_min: int = -200
    ip_max: int = 200


# -----------------------------
# 2) Environment driven by your demand data
# -----------------------------
class InventoryEnv:
    """
    Paper environment:
      state: IP_t (inventory position)
      actions: 0 (no order), 1 (order)
      lead time LT = 1
      transition:
        IP_{t+1} = IP_t - D_{t+1} + (OQ if order else 0)
      cost:
        holding if IP_t >= 0: h * IP_t
        backorder if IP_t < 0: b * abs(IP_t)
        ordering if order: O
    """
    def __init__(
        self,
        demand_series: np.ndarray,
        config: InventoryQLConfig,
        demand_mode: str = "bootstrap",  # "bootstrap" or "sequential"
        seed: Optional[int] = 42,
        initial_ip: int = 0,
    ):
        self.demand_series = np.asarray(demand_series, dtype=float)
        if self.demand_series.ndim != 1:
            raise ValueError("demand_series must be a 1D array of demands.")
        if len(self.demand_series) < 2:
            raise ValueError("demand_series must contain at least 2 demand values.")

        self.cfg = config
        self.rng = np.random.default_rng(seed)
        self.demand_mode = demand_mode
        self.initial_ip = int(initial_ip)

        self.t = 0
        self.ip = self.initial_ip

        # for sequential mode
        self._seq_idx = 0

    def reset(self) -> int:
        self.t = 0
        self.ip = int(self.initial_ip)
        # choose a random start point for sequential mode to avoid always same slice
        if self.demand_mode == "sequential":
            self._seq_idx = int(self.rng.integers(0, len(self.demand_series)))
        return self.ip

    def _sample_demand_next(self) -> float:
        """
        We interpret D_{t+1} as the demand realized during the next period.
        """
        if self.demand_mode == "bootstrap": # randomly samples from your historical demand values (for training)
            return float(self.rng.choice(self.demand_series))
        elif self.demand_mode == "sequential": # walks through the dataset in order (for evaluation)
            d = float(self.demand_series[self._seq_idx])
            self._seq_idx = (self._seq_idx + 1) % len(self.demand_series)
            return d
        else:
            raise ValueError("demand_mode must be 'bootstrap' or 'sequential'.")

    def step(self, action: int) -> Tuple[int, float, float]:
        """
        Returns: (next_ip, reward, cost)
        reward = -cost so we can use standard max-Q.
        """
        if action not in (0, 1):
            raise ValueError("action must be 0 (no order) or 1 (order).")

        # 1) compute cost at time t based on current IP_t and action
        holding_cost = self.cfg.h * self.ip if self.ip >= 0 else 0.0
        backorder_cost = self.cfg.b * abs(self.ip) if self.ip < 0 else 0.0
        ordering_cost = self.cfg.O if action == 1 else 0.0
        cost = holding_cost + backorder_cost + ordering_cost

        # 2) sample next period demand D_{t+1}
        demand_next = self._sample_demand_next()

        # 3) apply paper transition with LT = 1
        next_ip = self.ip - demand_next + (self.cfg.OQ if action == 1 else 0)
        # inventory position can be fractional if demand is fractional; usually demand is integer
        next_ip = int(np.round(next_ip))

        # 4) clip to bounds so Q-table stays finite
        next_ip = int(np.clip(next_ip, self.cfg.ip_min, self.cfg.ip_max))

        # 5) advance time and state
        self.ip = next_ip
        self.t += 1

        # reward for Q-learning (maximize reward == minimize cost)
        reward = -cost
        return next_ip, reward, cost


# -----------------------------
# 3) Tabular Q-learning (Eq. 5)
# -----------------------------
class QLearningAgent:
    """
    Q-table over IP states and 2 actions.
    Implements:
      Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
    """
    def __init__(self, config: InventoryQLConfig):
        self.cfg = config

        # Map integer IP in [ip_min, ip_max] -> index 0..N-1
        self.num_states = self.cfg.ip_max - self.cfg.ip_min + 1
        self.num_actions = 2

        self.Q = np.zeros((self.num_states, self.num_actions), dtype=float)

        # exploration parameters
        self.epsilon = self.cfg.epsilon_start

    def _s_to_idx(self, ip: int) -> int:
        return int(ip - self.cfg.ip_min)

    def act(self, ip: int, greedy: bool = False) -> int:
        """
        epsilon-greedy action selection during training.
        greedy=True forces exploitation (for evaluation / final policy).
        """
        s = self._s_to_idx(ip)

        if greedy or (np.random.rand() > self.epsilon):
            return int(np.argmax(self.Q[s]))
        else:
            return int(np.random.randint(0, self.num_actions))

    def update(self, ip: int, action: int, reward: float, next_ip: int):
        s = self._s_to_idx(ip)
        s_next = self._s_to_idx(next_ip)

        td_target = reward + self.cfg.gamma * np.max(self.Q[s_next])
        td_error = td_target - self.Q[s, action]
        self.Q[s, action] = self.Q[s, action] + self.cfg.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def policy(self) -> Dict[int, int]:
        """
        Returns the learned greedy action for every IP in the table.
        """
        pol = {}
        for ip in range(self.cfg.ip_min, self.cfg.ip_max + 1):
            s = self._s_to_idx(ip)
            pol[ip] = int(np.argmax(self.Q[s]))
        return pol


# -----------------------------
# 4) Training loop
# -----------------------------
def train_q_learning(
    demand_series: np.ndarray,
    config: InventoryQLConfig,
    demand_mode: str = "bootstrap",
    seed: int = 42,
    initial_ip: int = 0,
) -> Tuple[QLearningAgent, Dict[str, List[float]]]:
    env = InventoryEnv(
        demand_series=demand_series,
        config=config,
        demand_mode=demand_mode,
        seed=seed,
        initial_ip=initial_ip,
    )
    agent = QLearningAgent(config)

    history = {"episode_total_cost": [], "episode_total_reward": [], "epsilon": []}

    for ep in range(config.num_episodes):
        ip = env.reset()
        total_cost = 0.0
        total_reward = 0.0

        for t in range(config.episode_length):
            action = agent.act(ip)
            next_ip, reward, cost = env.step(action)

            agent.update(ip, action, reward, next_ip)

            ip = next_ip
            total_cost += cost
            total_reward += reward

        agent.decay_epsilon()

        history["episode_total_cost"].append(total_cost)
        history["episode_total_reward"].append(total_reward)
        history["epsilon"].append(agent.epsilon)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{config.num_episodes} | total_cost={total_cost:.2f} | epsilon={agent.epsilon:.3f}")

    return agent, history


# -----------------------------
# 5) Evaluation (simulate with learned greedy policy)
# -----------------------------
def evaluate_policy(
    agent: QLearningAgent,
    demand_series: np.ndarray,
    config: InventoryQLConfig,
    demand_mode: str = "sequential",
    seed: int = 123,
    initial_ip: int = 0,
    horizon: int = 1000,
) -> Dict[str, float]:
    env = InventoryEnv(
        demand_series=demand_series,
        config=config,
        demand_mode=demand_mode,
        seed=seed,
        initial_ip=initial_ip,
    )
    ip = env.reset()

    total_cost = 0.0
    orders = 0

    for _ in range(horizon):
        action = agent.act(ip, greedy = True)
        orders += int(action == 1)
        ip, _, cost = env.step(action)
        total_cost += cost

    return {
        "avg_cost_per_period": total_cost / horizon, # Average cost per day, horizon is a 1000 days
        "order_rate": orders / horizon, # The percentage of times an agent places an order per day. For eg order_rate = 0.5 -> The agent places an order about 50.1% of the time aka once every 2 days
        "total_cost": total_cost,
    }


# -----------------------------
# 6) How you plug in your demand data
# -----------------------------
def load_demand_from_csv(csv_path: str, demand_column: str) -> np.ndarray:
    """
    Required input: a CSV file with a column containing demand per period.
    Example: demand per day.
    """
    df = pd.read_csv(csv_path)
    if demand_column not in df.columns:
        raise ValueError(f"Column '{demand_column}' not found. Available columns: {list(df.columns)}")

    demand = df[demand_column].dropna().values
    # optional: enforce non-negative
    demand = np.asarray(demand, dtype=float)
    if np.any(demand < 0):
        raise ValueError("Demand contains negative values. Check your data.")
    return demand

# -----------------------------
# 7) Sanity Check
# -----------------------------
def sanity_check(results, demand, cfg):
    mean_demand_per_day = sum(demand) / len(demand)
    average_order_per_day = results['order_rate'] * cfg.OQ 
    print("mean demand per day", mean_demand_per_day)
    print("average_order_per_day", average_order_per_day)


if __name__ == "__main__":
    # Example usage (replace with your own file/column)
   # demand = load_demand_from_csv("/Users/charlottevedrines/Documents/Capstone/Q-Learning/one_item_demand.csv", "Order_Qty")

    # OR if you already have a numpy array:
    # demand = np.array([3, 5, 2, 7, 4, 6, ...])

    # Dummy demand for illustration:
    demand = np.random.poisson(lam=5, size=5000)

    cfg = InventoryQLConfig(
        h=1.0,
        b=5.0,
        O=20.0,
        OQ=10,        # fixed quantity (policy design), typical order size historically
        alpha=0.5,
        gamma=0.5,
        episode_length=1000,
        num_episodes=1000,  # start smaller; increase later
        ip_min=-200,
        ip_max=200,
    )

    agent, hist = train_q_learning(demand, cfg, demand_mode="bootstrap", seed=42, initial_ip=0)

    results = evaluate_policy(agent, demand, cfg, demand_mode="sequential", horizon=1000)
    print("Evaluation:", results)

    # Assuming demand has two columns: one with the date and the second with the number of orders of OQ
    sanity_check_var = sanity_check(results, demand, cfg) 

    # The learned policy is a mapping IP -> action
    pol = agent.policy()
    # Example: show ordering decision around low inventory
    for ip in range(-20, 21):
        print(ip, "ORDER" if pol[ip] == 1 else "NO ORDER")