import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional


# ============================================================
# 0) CONFIGS
# ============================================================
@dataclass
class SpaceConfig:
    """
    Warehouse-level shared constraint.
    """
    warehouse_capacity: float = 10000.0   # total available space
    use_on_hand_for_space: bool = True    # use max(IP,0) to compute space usage



@dataclass
class InventoryQLConfig:
    """
    Paper-like Q-learning config for ONE SKU.
    We will instantiate one agent per SKU.
    """
    # Costs
    h: float = 1.0      # holding cost per unit per day
    b: float = 5.0      # backorder cost per unit per day
    O: float = 20.0     # fixed cost per order event

    # Fixed order quantity (policy design in paper)
    OQ_default: int = 10

    # Q-learning params (paper values)
    alpha: float = 0.5
    gamma: float = 0.5

    # Training
    episode_length: int = 1000
    num_episodes: int = 500  # you can increase to 1000 later

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99

    # Q-table inventory bounds
    ip_min: int = -200
    ip_max: int = 200


# ============================================================
# 1) DEMAND STORAGE (NO MATRIX NEEDED)
# ============================================================
class DemandStore:
    """
    Holds demand data in long format:
        date | sku_id | demand

    Provides:
      - per-sku demand series for training
      - daily demand vector for simulation
    """
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        sku_col: str = "sku_id",
        demand_col: str = "demand"
    ):
        self.date_col = date_col
        self.sku_col = sku_col
        self.demand_col = demand_col

        # Basic checks
        for c in [date_col, sku_col, demand_col]:
            if c not in df.columns:
                raise ValueError(f"Column '{c}' not found in dataframe.")

        # Ensure proper types
        ddf = df.copy()
        ddf[date_col] = pd.to_datetime(ddf[date_col])

        # Aggregate if multiple rows per (date, sku) exist
        ddf = (
            ddf.groupby([date_col, sku_col], as_index=False)[demand_col]
            .sum()
        )

        self.df = ddf

        # sorted unique lists
        self.dates = np.array(sorted(ddf[date_col].unique()))
        self.skus = np.array(sorted(ddf[sku_col].unique()))

        # maps
        self.sku_to_idx = {sku: i for i, sku in enumerate(self.skus)}
        self.idx_to_sku = {i: sku for sku, i in self.sku_to_idx.items()}

        # Build fast lookup: for each date, get demand vector length N
        self.N = len(self.skus)
        self.T = len(self.dates)
        self._date_to_vec: Dict[pd.Timestamp, np.ndarray] = {}

        for date in self.dates:
            sub = ddf[ddf[date_col] == date]
            vec = np.zeros(self.N, dtype=float)
            idxs = sub[sku_col].map(self.sku_to_idx).values
            vec[idxs] = sub[demand_col].values
            self._date_to_vec[pd.Timestamp(date)] = vec

        # Build per-sku demand series for training
        # (use all observed demands ordered by date; missing days treated as 0)
        self._sku_series: Dict[str, np.ndarray] = {}
        for sku in self.skus:
            sub = ddf[ddf[sku_col] == sku].set_index(date_col).sort_index()
            full = pd.Series(index=self.dates, dtype=float).fillna(0.0)
            full.loc[sub.index.values] = sub[demand_col].values
            self._sku_series[sku] = full.values.astype(float)

    def get_sku_series(self, sku_id: str) -> np.ndarray:
        return self._sku_series[sku_id]

    def get_daily_vector(self, date: pd.Timestamp) -> np.ndarray:
        return self._date_to_vec[pd.Timestamp(date)]

    def get_all_skus(self) -> np.ndarray:
        return self.skus

    def get_all_dates(self) -> np.ndarray:
        return self.dates


# ============================================================
# 2) SINGLE-SKU ENVIRONMENT (PAPER TRANSITION)
# ============================================================

class InventoryEnvOneSKU:
    """
    Paper setup:
      State S(t)=IP_t
      Actions: 0 (no order), 1 (order)
      LT = 1
      If order -> IP_{t+1} = IP_t - D_{t+1} + OQ
      else     IP_{t+1} = IP_t - D_{t+1}
      Cost = holding/backorder + ordering
      Reward used in Q-learning = -cost (so we maximize reward)
    """
    def __init__(
        self,
        demand_series: np.ndarray,
        cfg: InventoryQLConfig,
        OQ: int,
        demand_mode: str = "bootstrap",
        seed: int = 42,
        initial_ip: int = 0
    ):
        self.demand_series = np.asarray(demand_series, dtype=float)
        if self.demand_series.ndim != 1:
            raise ValueError("demand_series must be 1D.")
        if len(self.demand_series) < 2:
            raise ValueError("demand_series too short.")

        self.cfg = cfg
        self.OQ = int(OQ)

        self.rng = np.random.default_rng(seed)
        self.demand_mode = demand_mode
        self.initial_ip = int(initial_ip)

        self.ip = self.initial_ip
        self._seq_idx = 0

    def reset(self) -> int:
        self.ip = self.initial_ip
        if self.demand_mode == "sequential":
            self._seq_idx = int(self.rng.integers(0, len(self.demand_series)))
        return self.ip

    def _sample_demand(self) -> float:
        if self.demand_mode == "bootstrap":
            return float(self.rng.choice(self.demand_series))
        elif self.demand_mode == "sequential":
            d = float(self.demand_series[self._seq_idx])
            self._seq_idx = (self._seq_idx + 1) % len(self.demand_series)
            return d
        else:
            raise ValueError("demand_mode must be bootstrap/sequential")

    def step(self, action: int) -> Tuple[int, float, float]:
        if action not in (0, 1):
            raise ValueError("Action must be 0 or 1.")

        # cost at time t
        holding = self.cfg.h * self.ip if self.ip >= 0 else 0.0
        backorder = self.cfg.b * abs(self.ip) if self.ip < 0 else 0.0
        ordering = self.cfg.O if action == 1 else 0.0
        cost = holding + backorder + ordering

        # demand next period
        d_next = self._sample_demand()

        # transition (LT=1)
        next_ip = self.ip - d_next + (self.OQ if action == 1 else 0)
        next_ip = int(np.round(next_ip))
        next_ip = int(np.clip(next_ip, self.cfg.ip_min, self.cfg.ip_max))

        self.ip = next_ip
        reward = -cost
        return next_ip, reward, cost

# -----------------------------
# 3) Tabular Q-learning (Eq. 5)
# -----------------------------
class QLearningAgentOneSKU:
    """
    Q-table for one SKU:
      Q(IP, action)
    """
    def __init__(self, cfg: InventoryQLConfig):
        self.cfg = cfg
        self.num_states = cfg.ip_max - cfg.ip_min + 1
        self.num_actions = 2
        self.Q = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.epsilon = cfg.epsilon_start

    def _s_to_idx(self, ip: int) -> int:
        return int(ip - self.cfg.ip_min)

    def act(self, ip: int, greedy: bool = False) -> int:
        s = self._s_to_idx(ip)

        # epsilon-greedy exploration
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

    def advantage_order_vs_noorder(self, ip: int) -> float:
        s = self._s_to_idx(ip)
        return float(self.Q[s, 1] - self.Q[s, 0])

    def policy(self) -> Dict[int, int]:
        """
        Returns the learned greedy action for every IP in the table.
        """
        pol = {}
        for ip in range(self.cfg.ip_min, self.cfg.ip_max + 1):
            s = self._s_to_idx(ip)
            pol[ip] = int(np.argmax(self.Q[s]))
        return pol

def train_agent_for_sku(
    demand_series: np.ndarray,
    cfg: InventoryQLConfig,
    OQ: int,
    seed: int = 42,
    initial_ip: int = 0
) -> QLearningAgentOneSKU:
    env = InventoryEnvOneSKU(
        demand_series=demand_series,
        cfg=cfg,
        OQ=OQ,
        demand_mode="bootstrap",
        seed=seed,
        initial_ip=initial_ip
    )
    agent = QLearningAgentOneSKU(cfg)

    for ep in range(cfg.num_episodes):
        ip = env.reset()
        for _ in range(cfg.episode_length):
            a = agent.act(ip, greedy=False)
            next_ip, r, _ = env.step(a)
            agent.update(ip, a, r, next_ip)
            ip = next_ip
        agent.decay_epsilon()

    return agent


# ============================================================
# 4) WAREHOUSE SPACE + KNAPSACK SELECTION
# ============================================================

def compute_space_used(IP: np.ndarray, v: np.ndarray, use_on_hand: bool = True) -> float:
    if use_on_hand:
        return float(np.sum(np.maximum(IP, 0) * v))
    return float(np.sum(IP * v))


def greedy_knapsack_selection(
    candidate_ids: np.ndarray,
    advantages: np.ndarray,
    volumes: np.ndarray,
    remaining_capacity: float
) -> np.ndarray:
    """
    Greedy knapsack: rank by advantage/volume.
    """
    volumes = np.maximum(volumes, 1e-9)
    score = advantages / volumes
    order = np.argsort(score)[::-1]

    selected = []
    used = 0.0
    for k in order:
        sku = candidate_ids[k]
        vol = volumes[k]
        if used + vol <= remaining_capacity:
            selected.append(sku)
            used += vol

    return np.array(selected, dtype=int)


def decide_orders_with_space(
    agents: List[QLearningAgentOneSKU],
    IP: np.ndarray,
    v: np.ndarray,
    OQ: np.ndarray,
    space_cfg: SpaceConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      proposed_orders: shape (N,)
      accepted_orders: shape (N,)
    """
    N = len(agents)
    proposed = np.zeros(N, dtype=int)
    advantages = np.zeros(N, dtype=float)

    used = compute_space_used(IP, v, space_cfg.use_on_hand_for_space)
    remaining = max(0.0, space_cfg.warehouse_capacity - used)

    for i in range(N):
        action = agents[i].act(int(IP[i]), greedy=True)
        proposed[i] = action
        if action == 1:
            advantages[i] = agents[i].advantage_order_vs_noorder(int(IP[i]))

    candidates = np.where(proposed == 1)[0]
    if len(candidates) == 0:
        return proposed, proposed.copy()

    volumes = v[candidates] * OQ[candidates]

    # If fits, accept all proposed
    if float(np.sum(volumes)) <= remaining:
        return proposed, proposed.copy()

    # Otherwise knapsack selection
    selected = greedy_knapsack_selection(
        candidate_ids=candidates,
        advantages=advantages[candidates],
        volumes=volumes,
        remaining_capacity=remaining
    )

    accepted = np.zeros(N, dtype=int)
    accepted[selected] = 1
    return proposed, accepted

# ============================================================
# 5) DAILY SYSTEM SIMULATION (DEMAND INPUT DAILY)
# ============================================================

def simulate_daily_orders(
    demand_store: DemandStore,
    agents: List[QLearningAgentOneSKU],
    cfg: InventoryQLConfig,
    space_cfg: SpaceConfig,
    v: np.ndarray,
    OQ: np.ndarray,
    initial_IP: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    dates = demand_store.get_all_dates()
    N = len(agents)

    if initial_IP is None:
        IP = np.zeros(N, dtype=int)
    else:
        IP = initial_IP.astype(int).copy()

    # Traces
    proposed_trace = np.zeros((len(dates), N), dtype=int)
    accepted_trace = np.zeros((len(dates), N), dtype=int)
    space_used_trace = np.zeros(len(dates), dtype=float)
    total_cost_trace = np.zeros(len(dates), dtype=float)
    backorder_units_trace = np.zeros(len(dates), dtype=float)
    total_demand_trace = np.zeros(len(dates), dtype=float)

    for t, date in enumerate(dates):
        demand_today = demand_store.get_daily_vector(date)
        total_demand = float(np.sum(demand_today))
        total_demand_trace[t] = total_demand

        # decide orders
        proposed_orders, accepted_orders = decide_orders_with_space(
            agents=agents, IP=IP, v=v, OQ=OQ, space_cfg=space_cfg
        )
        proposed_trace[t] = proposed_orders
        accepted_trace[t] = accepted_orders

        # space usage (before update)
        space_used_trace[t] = compute_space_used(IP, v, space_cfg.use_on_hand_for_space)

        # compute costs for this day based on IP_t and accepted orders
        holding = cfg.h * np.maximum(IP, 0)
        backorder = cfg.b * np.maximum(-IP, 0)
        ordering = cfg.O * accepted_orders
        daily_cost = float(np.sum(holding + backorder + ordering))
        total_cost_trace[t] = daily_cost

        # backorder units at time t
        backorder_units_trace[t] = float(np.sum(np.maximum(-IP, 0)))

        # update IP to next day
        IP = IP - demand_today + accepted_orders * OQ
        IP = np.clip(IP, cfg.ip_min, cfg.ip_max).astype(int)

    # -----------------------------
    # KPIs
    # -----------------------------
    proposed_total = float(np.sum(proposed_trace))
    accepted_total = float(np.sum(accepted_trace))
    blocked_total = proposed_total - accepted_total

    total_demand_all = float(np.sum(total_demand_trace))
    total_backorders_all = float(np.sum(backorder_units_trace))

    backorder_rate = (total_backorders_all / total_demand_all) if total_demand_all > 0 else 0.0
    fill_rate = 1.0 - backorder_rate

    avg_cost_per_day = float(np.mean(total_cost_trace))

    space_block_rate = (blocked_total / proposed_total) if proposed_total > 0 else 0.0
    avg_blocked_orders_per_day = float(np.mean(np.sum((proposed_trace == 1) & (accepted_trace == 0), axis=1)))

    metrics = {
        "avg_cost_per_day": avg_cost_per_day,
        "backorder_rate_units": backorder_rate,
        "fill_rate": fill_rate,
        "orders_proposed_total": proposed_total,
        "orders_accepted_total": accepted_total,
        "orders_blocked_total": blocked_total,
        "space_block_rate": space_block_rate,
        "avg_blocked_orders_per_day": avg_blocked_orders_per_day,
    }

    return {
        "proposed_orders_trace": proposed_trace,
        "accepted_orders_trace": accepted_trace,
        "space_used_trace": space_used_trace,
        "total_cost_trace": total_cost_trace,
        "backorder_units_trace": backorder_units_trace,
        "total_demand_trace": total_demand_trace,
        "final_IP": IP,
        "dates": dates,
        "metrics": metrics
    }


# ============================================================
# 6) MAIN: LOAD DATA, TRAIN, RUN
# ============================================================

def load_inputs(
    demand_csv_path: str,
    space_csv_path: str,
    date_col: str = "date",
    sku_col: str = "sku_id",
    demand_col: str = "demand",
    space_col: str = "space_per_unit",
    oq_col: Optional[str] = None,  # if you have per sku OQ; else use default
) -> Tuple[DemandStore, np.ndarray, np.ndarray]:
    """
    demand_csv must have: date, sku_id, demand
    space_csv must have: sku_id, space_per_unit (+ optionally OQ)
    """
    df_demand = pd.read_csv(demand_csv_path)
    df_space = pd.read_csv(space_csv_path)

    df_demand[date_col] = pd.to_datetime(df_demand[date_col])

    # Build demand store
    store = DemandStore(df_demand, date_col=date_col, sku_col=sku_col, demand_col=demand_col)
    skus = store.get_all_skus()

    # Align space file to SKU list
    if sku_col not in df_space.columns or space_col not in df_space.columns:
        raise ValueError(f"space_csv must have columns: {sku_col}, {space_col}")

    df_space = df_space.set_index(sku_col)

    missing = [sku for sku in skus if sku not in df_space.index]
    if missing:
        raise ValueError(f"Missing {len(missing)} SKUs in space file. Example missing: {missing[:5]}")

    v = df_space.loc[skus, space_col].astype(float).values

    if oq_col is not None:
        if oq_col not in df_space.columns:
            raise ValueError(f"oq_col={oq_col} not found in space file.")
        OQ = df_space.loc[skus, oq_col].astype(int).values
    else:
        OQ = None  # will be filled with cfg default in main

    return store, v, OQ


if __name__ == "__main__":
    # ------------------------------------------------------------
    # USER INPUTS
    # ------------------------------------------------------------
    # 1) Your data files
    # demand_csv_path = "demand.csv"      # columns: date, sku_id, demand
    # space_csv_path = "space.csv"        # columns: sku_id, space_per_unit (and optionally OQ)

    # For demonstration, we'll generate fake data instead
    # ------------------------------------------------------------
    print("Demo mode: generating fake data...")

    rng = np.random.default_rng(0)
    N = 200  # set to 5000 for your real case
    T = 120  # days

    skus = [f"SKU_{i}" for i in range(N)]
    dates = pd.date_range("2025-01-01", periods=T, freq="D")

    # Fake long-format demand
    demand_rows = []
    for d in dates:
        # Poisson demand per SKU
        demands = rng.poisson(lam=3, size=N)
        for i, sku in enumerate(skus):
            if demands[i] > 0:
                demand_rows.append((d, sku, float(demands[i])))

    df_demand = pd.DataFrame(demand_rows, columns=["date", "sku_id", "demand"])

    # Fake space per unit
    df_space = pd.DataFrame({
        "sku_id": skus,
        "space_per_unit": rng.uniform(0.5, 2.0, size=N),
        # optional per-sku order quantities
        "OQ": rng.integers(5, 25, size=N)
    })

    store = DemandStore(df_demand, date_col="date", sku_col="sku_id", demand_col="demand")
    v = df_space.set_index("sku_id").loc[store.get_all_skus(), "space_per_unit"].values
    OQ_vec = df_space.set_index("sku_id").loc[store.get_all_skus(), "OQ"].values.astype(int)

    # ------------------------------------------------------------
    # CONFIGS
    # ------------------------------------------------------------
    inv_cfg = InventoryQLConfig(
        h=1.0,
        b=5.0,
        O=20.0,
        OQ_default=10,
        alpha=0.5,
        gamma=0.5,
        episode_length=800,
        num_episodes=200,
        ip_min=-200,
        ip_max=200,
    )

    space_cfg = SpaceConfig(
        warehouse_capacity=15000.0,  # try changing this and see effect
        use_on_hand_for_space=True
    )

    # If you don't have OQ per SKU, use inv_cfg.OQ_default:
    if OQ_vec is None:
        OQ_vec = np.full(len(store.get_all_skus()), inv_cfg.OQ_default, dtype=int)

    # ------------------------------------------------------------
    # TRAIN ONE AGENT PER SKU
    # ------------------------------------------------------------
    print("Training Q-learning agents per SKU...")
    agents: List[QLearningAgentOneSKU] = []
    all_skus = store.get_all_skus()

    for idx, sku in enumerate(all_skus):
        series = store.get_sku_series(sku)
        agent = train_agent_for_sku(
            demand_series=series,
            cfg=inv_cfg,
            OQ=int(OQ_vec[idx]),
            seed=42 + idx,
            initial_ip=0
        )
        agents.append(agent)

        if (idx + 1) % 50 == 0:
            print(f"Trained {idx+1}/{len(all_skus)} agents")

    # ------------------------------------------------------------
    # RUN DAILY ORDERING WITH SPACE KNAPSACK
    # ------------------------------------------------------------
    print("Simulating daily orders with space constraint...")
    sim = simulate_daily_orders(
        demand_store=store,
        agents=agents,
        cfg=inv_cfg,
        space_cfg=space_cfg,
        v=v,
        OQ=OQ_vec,
        initial_IP=np.zeros(len(all_skus), dtype=int)
    )

    space_used = sim["space_used_trace"]
    orders = sim["orders_trace"]

    print("\nDone.")
    print(f"Average space used: {space_used.mean():.2f} / capacity {space_cfg.warehouse_capacity:.2f}")
    print(f"Avg #orders/day: {orders.sum(axis=1).mean():.1f} out of {len(all_skus)} SKUs")

    # Example: show how many SKUs ordered on the last day
    print(f"Orders last day: {orders[-1].sum()}")

    print("\n--- KPIs ---")
    for k, v in sim["metrics"].items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


