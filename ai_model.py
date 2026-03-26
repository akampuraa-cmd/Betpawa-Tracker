"""
AI module: Genetic Algorithm + Q-Learning (Reinforcement Learning).

Both models predict the outcome of the next MUN match:
  0 = Loss  |  1 = Draw  |  2 = Win

Genetic Algorithm
-----------------
Each *individual* is a weight vector that drives a simple softmax classifier
over hand-crafted features derived from recent match history.

Fitness = prediction accuracy on a leave-one-out sliding window over the
historical data stored in the database.

Q-Learning (Reinforcement Learning)
------------------------------------
The *state* is the tuple of the last QL_STATE_WINDOW outcomes encoded as
integers (0/1/2).  The agent picks an action (predicted outcome) using an
ε-greedy policy and receives a reward of +1 (correct) or -1 (wrong).
"""

import math
import pickle
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config

logger = logging.getLogger(__name__)

# Action space: three possible predictions
ACTIONS = [config.RESULT_LOSS, config.RESULT_DRAW, config.RESULT_WIN]
N_ACTIONS = len(ACTIONS)
MODEL_STATE_VERSION = 1


# ── Feature engineering ───────────────────────────────────────────────────────

def _build_features(
    outcomes: List[int],
    goals: List[Tuple[int, int]],
    window: int,
    odds: Optional[Tuple[float, float, float]] = None,
    include_odds: bool = False,
) -> np.ndarray:
    """
    Build a feature vector from the last *window* matches.

    Features (length = window * 2 + 4 + 3 if include_odds else window * 2 + 4):
      - last *window* outcomes encoded as floats in [0, 1]   (0=L, 0.5=D, 1=W)
      - last *window* MUN goal counts (normalised by 5)
      - rolling mean goals scored   (last window)
      - rolling mean goals conceded (last window)
      - win rate   (last window)
      - draw rate  (last window)
      - home_odds, draw_odds, away_odds (if include_odds is True, evaluated/padded)
    """
    def _encode(o: int) -> float:
        return {config.RESULT_LOSS: 0.0, config.RESULT_DRAW: 0.5, config.RESULT_WIN: 1.0}[o]

    padded_out = ([config.RESULT_DRAW] * window + outcomes)[-window:]
    padded_goals = ([(0, 0)] * window + goals)[-window:]

    out_enc = np.array([_encode(o) for o in padded_out], dtype=float)
    mun_g = np.array([g[0] / 5.0 for g in padded_goals], dtype=float)
    opp_g = np.array([g[1] / 5.0 for g in padded_goals], dtype=float)

    mean_mun = float(np.mean(mun_g))
    mean_opp = float(np.mean(opp_g))
    win_rate = float(np.mean([1.0 if o == config.RESULT_WIN else 0.0 for o in padded_out]))
    draw_rate = float(np.mean([1.0 if o == config.RESULT_DRAW else 0.0 for o in padded_out]))

    features = [out_enc, mun_g, [mean_mun, mean_opp, win_rate, draw_rate]]

    if include_odds:
        if odds:
            # Normalize odds by dividing by 10 (assuming typical odds <10)
            norm_odds = [o / 10.0 if o else 0.0 for o in odds]
            features.append(norm_odds)
        else:
            features.append([0.0, 0.0, 0.0])

    return np.concatenate(features)


def _feature_size(window: int, include_odds: bool = False) -> int:
    base = window * 2 + 4
    return base + 3 if include_odds else base


# ── Genetic Algorithm ─────────────────────────────────────────────────────────

class GeneticAlgorithm:
    """
    Evolves a population of linear classifiers to predict match outcomes.

    Each individual is a weight matrix W of shape (N_ACTIONS, n_features)
    plus a bias vector b of shape (N_ACTIONS,), flattened into a 1-D array.
    """

    def __init__(
        self,
        population_size: int = config.GA_POPULATION_SIZE,
        generations: int = config.GA_GENERATIONS,
        mutation_rate: float = config.GA_MUTATION_RATE,
        crossover_rate: float = config.GA_CROSSOVER_RATE,
        elitism: int = config.GA_ELITISM_COUNT,
        feature_window: int = config.GA_FEATURE_WINDOW,
        include_odds: bool = False,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.feature_window = feature_window
        self.include_odds = include_odds
        self.n_features = _feature_size(feature_window, include_odds)
        self.gene_length = N_ACTIONS * (self.n_features + 1)  # W + b

        self.population: List[np.ndarray] = []
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = 0.0
        self.generation: int = 0
        self.fitness_history: List[float] = []

        self._init_population()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_population(self) -> None:
        self.population = [
            np.random.randn(self.gene_length) * 0.5
            for _ in range(self.population_size)
        ]

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _decode(self, individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (W, b) from the flat gene array."""
        w_size = N_ACTIONS * self.n_features
        W = individual[:w_size].reshape(N_ACTIONS, self.n_features)
        b = individual[w_size:]
        return W, b

    def _predict_proba(self, individual: np.ndarray, features: np.ndarray) -> np.ndarray:
        W, b = self._decode(individual)
        logits = W @ features + b
        # Stable softmax
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    def predict(self, features: np.ndarray) -> int:
        """Predict outcome using the best individual found so far."""
        if self.best_individual is None:
            return random.choice(ACTIONS)
        proba = self._predict_proba(self.best_individual, features)
        return ACTIONS[int(np.argmax(proba))]

    # ── Fitness ──────────────────────────────────────────────────────────────

    def evaluate_fitness(
        self,
        individual: np.ndarray,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
    ) -> float:
        """
        Leave-one-out accuracy: for each position i ≥ 1, build features from
        outcomes[:i] (padding fills missing history) and predict outcomes[i].

        Uses at least 1 training point as long as there are ≥ 2 samples.
        """
        if len(outcomes) < 2:
            return 0.0

        # Start from max(1, window) when enough data is available; fall back
        # to 1 when we have fewer samples than the configured window.
        start = min(self.feature_window, len(outcomes) - 1)

        correct = 0
        total = 0
        for i in range(start, len(outcomes)):
            # _build_features always returns a vector of self.n_features size
            feats = _build_features(outcomes[:i], goals[:i], self.feature_window, include_odds=self.include_odds)
            pred = ACTIONS[int(np.argmax(self._predict_proba(individual, feats)))]
            if pred == outcomes[i]:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    # ── Genetic operators ─────────────────────────────────────────────────────

    def _select(self, fitnesses: List[float]) -> np.ndarray:
        """Tournament selection (k=3)."""
        k = 3
        idxs = random.sample(range(len(self.population)), k)
        best_idx = max(idxs, key=lambda i: fitnesses[i])
        return self.population[best_idx].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        point = random.randint(1, len(p1) - 1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] += np.random.randn() * 0.1
        return individual

    # ── Training ─────────────────────────────────────────────────────────────

    def train_one_generation(
        self,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
    ) -> float:
        """Run one generation.  Returns the best fitness of this generation."""
        fitnesses = [
            self.evaluate_fitness(ind, outcomes, goals)
            for ind in self.population
        ]

        # Elitism: carry over best individuals unchanged
        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        new_pop = [self.population[i].copy() for i in sorted_idx[: self.elitism]]

        # Fill the rest with children
        while len(new_pop) < self.population_size:
            p1 = self._select(fitnesses)
            p2 = self._select(fitnesses)
            c1, c2 = self._crossover(p1, p2)
            new_pop.append(self._mutate(c1))
            if len(new_pop) < self.population_size:
                new_pop.append(self._mutate(c2))

        self.population = new_pop[:self.population_size]
        self.generation += 1

        best_idx = int(np.argmax(fitnesses))
        gen_best = fitnesses[best_idx]
        if gen_best >= self.best_fitness:
            self.best_fitness = gen_best
            self.best_individual = self.population[0].copy()

        self.fitness_history.append(gen_best)
        return gen_best

    def train(
        self,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
        generations: Optional[int] = None,
        callback=None,
    ) -> None:
        """Train for *generations* generations (default: self.generations)."""
        n = generations or self.generations
        for g in range(n):
            fit = self.train_one_generation(outcomes, goals)
            if callback:
                callback(generation=self.generation, best_fitness=fit)

    def get_features(self, outcomes: List[int], goals: List[Tuple[int, int]], odds: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        return _build_features(outcomes, goals, self.feature_window, odds=odds, include_odds=self.include_odds)

    def predict_next(self, outcomes: List[int], goals: List[Tuple[int, int]], odds: Optional[Tuple[float, float, float]] = None) -> int:
        feats = self.get_features(outcomes, goals, odds)
        return self.predict(feats)

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": MODEL_STATE_VERSION,
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "elitism": self.elitism,
            "feature_window": self.feature_window,
            "include_odds": self.include_odds,
            "n_features": self.n_features,
            "gene_length": self.gene_length,
            "population": [ind.copy() for ind in self.population],
            "best_individual": None if self.best_individual is None else self.best_individual.copy(),
            "best_fitness": self.best_fitness,
            "generation": self.generation,
            "fitness_history": list(self.fitness_history),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("version") != MODEL_STATE_VERSION:
            raise ValueError("Unsupported GeneticAlgorithm checkpoint version")
        if int(state["population_size"]) != self.population_size:
            raise ValueError("Population size mismatch in checkpoint")
        if int(state["feature_window"]) != self.feature_window or bool(state["include_odds"]) != self.include_odds:
            raise ValueError("Feature configuration mismatch in checkpoint")

        population = [np.array(ind, dtype=float) for ind in state["population"]]
        if not population:
            raise ValueError("Checkpoint does not contain a population")

        self.population = population
        self.best_individual = None if state["best_individual"] is None else np.array(state["best_individual"], dtype=float)
        self.best_fitness = float(state["best_fitness"])
        self.generation = int(state["generation"])
        self.fitness_history = [float(v) for v in state.get("fitness_history", [])]

    def save(self, path: str) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> bool:
        checkpoint = Path(path)
        if not checkpoint.exists():
            return False
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        self.load_state_dict(state)
        return True


# ── Q-Learning ────────────────────────────────────────────────────────────────

State = Tuple[int, ...]


class QLearning:
    """
    Tabular Q-Learning agent for predicting the next MUN match outcome.

    State  : last *state_window* outcomes as a tuple of ints
    Actions: ACTIONS = [LOSS, DRAW, WIN]
    Reward : +1 correct prediction, -1 wrong
    """

    def __init__(
        self,
        learning_rate: float = config.QL_LEARNING_RATE,
        discount: float = config.QL_DISCOUNT_FACTOR,
        epsilon: float = config.QL_EPSILON,
        epsilon_decay: float = config.QL_EPSILON_DECAY,
        epsilon_min: float = config.QL_EPSILON_MIN,
        state_window: int = config.QL_STATE_WINDOW,
    ):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_window = state_window

        self.q_table: Dict[Tuple[State, int], float] = {}
        self.total_steps = 0
        self.total_reward = 0.0
        self.correct_predictions = 0
        self.reward_history: List[float] = []

    # ── Q-table helpers ──────────────────────────────────────────────────────

    def _q(self, state: State, action: int) -> float:
        return self.q_table.get((state, action), 0.0)

    def _set_q(self, state: State, action: int, value: float) -> None:
        self.q_table[(state, action)] = value

    def _best_action(self, state: State) -> int:
        q_vals = {a: self._q(state, a) for a in ACTIONS}
        return max(q_vals, key=q_vals.__getitem__)

    # ── State encoding ───────────────────────────────────────────────────────

    def _get_state(self, outcomes: List[int]) -> State:
        w = self.state_window
        padded = ([config.RESULT_DRAW] * w + outcomes)[-w:]
        return tuple(padded)

    # ── Policy ───────────────────────────────────────────────────────────────

    def choose_action(self, state: State) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return self._best_action(state)

    # ── Update ───────────────────────────────────────────────────────────────

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
    ) -> None:
        """Bellman update."""
        best_next = max(self._q(next_state, a) for a in ACTIONS)
        old_q = self._q(state, action)
        new_q = old_q + self.lr * (reward + self.gamma * best_next - old_q)
        self._set_q(state, action, new_q)

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        outcomes: List[int],
        callback=None,
    ) -> None:
        """
        Replay all stored outcomes sequentially to train the Q-table.
        """
        w = self.state_window
        for i in range(w, len(outcomes)):
            state = self._get_state(outcomes[:i])
            action = self.choose_action(state)
            actual = outcomes[i]
            reward = 1.0 if action == actual else -1.0

            next_state = self._get_state(outcomes[: i + 1])
            self.update(state, action, reward, next_state)

            self.total_steps += 1
            self.total_reward += reward
            if reward > 0:
                self.correct_predictions += 1
            self.reward_history.append(reward)
            self._decay_epsilon()

            if callback:
                callback(
                    step=self.total_steps,
                    reward=reward,
                    epsilon=self.epsilon,
                    q_table_size=len(self.q_table),
                )

    def predict_next(self, outcomes: List[int]) -> int:
        """Predict the next outcome using the greedy policy."""
        state = self._get_state(outcomes)
        return self._best_action(state)

    @property
    def accuracy(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.correct_predictions / self.total_steps

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": MODEL_STATE_VERSION,
            "learning_rate": self.lr,
            "discount": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "state_window": self.state_window,
            "q_table": dict(self.q_table),
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "correct_predictions": self.correct_predictions,
            "reward_history": list(self.reward_history),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("version") != MODEL_STATE_VERSION:
            raise ValueError("Unsupported QLearning checkpoint version")
        if int(state["state_window"]) != self.state_window:
            raise ValueError("State window mismatch in checkpoint")

        self.lr = float(state["learning_rate"])
        self.gamma = float(state["discount"])
        self.epsilon = float(state["epsilon"])
        self.epsilon_decay = float(state["epsilon_decay"])
        self.epsilon_min = float(state["epsilon_min"])
        self.q_table = dict(state["q_table"])
        self.total_steps = int(state["total_steps"])
        self.total_reward = float(state["total_reward"])
        self.correct_predictions = int(state["correct_predictions"])
        self.reward_history = [float(v) for v in state.get("reward_history", [])]

    def save(self, path: str) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> bool:
        checkpoint = Path(path)
        if not checkpoint.exists():
            return False
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        self.load_state_dict(state)
        return True


# ── LSTM ──────────────────────────────────────────────────────────────────────

class PyTorchLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMPredictor:
    def __init__(
        self,
        seq_length: int = getattr(config, "LSTM_SEQUENCE_LENGTH", 10),
        hidden_size: int = getattr(config, "LSTM_HIDDEN_SIZE", 32),
        num_layers: int = getattr(config, "LSTM_NUM_LAYERS", 2),
        lr: float = getattr(config, "LSTM_LEARNING_RATE", 0.005),
    ):
        self.seq_length = seq_length
        self.input_size = 3  # encoded outcome, mun goals, opp goals
        
        self.model = PyTorchLSTM(self.input_size, hidden_size, num_layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _step_feature(self, outcome: int, mun_g: int, opp_g: int) -> List[float]:
        enc = {config.RESULT_LOSS: 0.0, config.RESULT_DRAW: 0.5, config.RESULT_WIN: 1.0}[outcome]
        return [enc, mun_g / 5.0, opp_g / 5.0]

    def train(self, outcomes: List[int], goals: List[Tuple[int, int]], epochs: int = 50) -> None:
        if len(outcomes) <= self.seq_length:
            return

        X, y = [], []
        for i in range(len(outcomes) - self.seq_length):
            seq_X = [
                self._step_feature(outcomes[j], goals[j][0], goals[j][1])
                for j in range(i, i + self.seq_length)
            ]
            X.append(seq_X)
            y.append(outcomes[i + self.seq_length])
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict_next(self, outcomes: List[int], goals: List[Tuple[int, int]]) -> Tuple[int, float]:
        if len(outcomes) == 0:
            return config.RESULT_DRAW, 0.0
        
        self.model.eval()
        seq_X = []
        
        start_idx = max(0, len(outcomes) - self.seq_length)
        pad_count = self.seq_length - (len(outcomes) - start_idx)
        
        for _ in range(pad_count):
            seq_X.append(self._step_feature(config.RESULT_DRAW, 0, 0)) # Default padding
        
        for i in range(start_idx, len(outcomes)):
            seq_X.append(self._step_feature(outcomes[i], goals[i][0], goals[i][1]))
        
        with torch.no_grad():
            x_tensor = torch.tensor([seq_X], dtype=torch.float32)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
            
        return ACTIONS[pred], conf

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": MODEL_STATE_VERSION,
            "seq_length": self.seq_length,
            "input_size": self.input_size,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("version") != MODEL_STATE_VERSION:
            raise ValueError("Unsupported LSTMPredictor checkpoint version")
        if int(state["seq_length"]) != self.seq_length or int(state["input_size"]) != self.input_size:
            raise ValueError("LSTM configuration mismatch in checkpoint")

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def save(self, path: str) -> None:
        checkpoint = Path(path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> bool:
        checkpoint = Path(path)
        if not checkpoint.exists():
            return False
        with checkpoint.open("rb") as handle:
            state = pickle.load(handle)
        self.load_state_dict(state)
        return True


# ── Combined AI interface ─────────────────────────────────────────────────────

class BetpawaAI:
    """
    Convenience wrapper combining the GA, Q-Learning, and LSTM models.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, auto_load: bool = True):
        self.checkpoint_path = checkpoint_path or getattr(config, "MODEL_CHECKPOINT_PATH", None)
        self.ga = GeneticAlgorithm(include_odds=True)
        self.ql = QLearning()
        self.lstm = LSTMPredictor()

        if auto_load:
            self.load()

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": MODEL_STATE_VERSION,
            "ga": self.ga.state_dict(),
            "ql": self.ql.state_dict(),
            "lstm": self.lstm.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("version") != MODEL_STATE_VERSION:
            raise ValueError("Unsupported BetpawaAI checkpoint version")

        self.ga.load_state_dict(state["ga"])
        self.ql.load_state_dict(state["ql"])
        self.lstm.load_state_dict(state["lstm"])

    def save(self, path: Optional[str] = None) -> bool:
        checkpoint_path = path or self.checkpoint_path
        if not checkpoint_path:
            return False

        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def load(self, path: Optional[str] = None) -> bool:
        checkpoint_path = path or self.checkpoint_path
        if not checkpoint_path:
            return False

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            return False

        try:
            with checkpoint.open("rb") as handle:
                state = pickle.load(handle)
            self.load_state_dict(state)
        except Exception as exc:
            logger.warning("Could not load AI checkpoint from %s: %s", checkpoint, exc)
            return False

        logger.info("Loaded AI checkpoint from %s", checkpoint)
        return True

    def train(
        self,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
        ga_generations: Optional[int] = None,
        lstm_epochs: Optional[int] = None,
        ga_callback=None,
        ql_callback=None,
    ) -> None:
        """Train all models on historical data."""
        if len(outcomes) < 2:
            logger.info("Not enough data to train (need ≥ 2 results)")
            return

        logger.info("Training Genetic Algorithm …")
        self.ga.train(outcomes, goals, generations=ga_generations, callback=ga_callback)

        logger.info("Training Q-Learning agent …")
        self.ql.train(outcomes, callback=ql_callback)

        logger.info("Training LSTM agent …")
        epochs = lstm_epochs or getattr(config, "LSTM_EPOCHS", 50)
        self.lstm.train(outcomes, goals, epochs=epochs)

        self.save()

    def predict(
        self,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
        odds: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, object]:
        """
        Return a combined prediction dict:
          {
            "ga_prediction": int,
            "ga_label": str,
            "ga_confidence": float,
            "ql_prediction": int,
            "ql_label": str,
            "lstm_prediction": int,
            "lstm_label": str,
            "lstm_confidence": float,
            "consensus": int,
            "consensus_label": str,
          }
        """
        ga_pred = self.ga.predict_next(outcomes, goals, odds)
        ql_pred = self.ql.predict_next(outcomes)
        lstm_pred, lstm_conf = self.lstm.predict_next(outcomes, goals)

        # GA confidence = best individual's softmax probability
        if self.ga.best_individual is not None:
            feats = self.ga.get_features(outcomes, goals, odds)
            proba = self.ga._predict_proba(self.ga.best_individual, feats)
            ga_conf = float(np.max(proba))
        else:
            ga_conf = 0.0

        # Consensus logic: Majority vote among GA, QL, LSTM
        preds = [ga_pred, ql_pred, lstm_pred]
        votes = {config.RESULT_WIN: 0, config.RESULT_DRAW: 0, config.RESULT_LOSS: 0}
        for p in preds:
            votes[p] += 1
        
        max_votes = max(votes.values())
        tied = [k for k, v in votes.items() if v == max_votes]
        
        if len(tied) == 1:
            consensus = tied[0]
        else:
            # Disagreement (3 different votes, or tie) -> trust the one with max confidence
            if lstm_conf >= ga_conf:
                consensus = lstm_pred
            else:
                consensus = ga_pred

        return {
            "ga_prediction": ga_pred,
            "ga_label": config.RESULT_LABELS[ga_pred],
            "ga_confidence": ga_conf,
            "ql_prediction": ql_pred,
            "ql_label": config.RESULT_LABELS[ql_pred],
            "lstm_prediction": lstm_pred,
            "lstm_label": config.RESULT_LABELS[lstm_pred],
            "lstm_confidence": lstm_conf,
            "consensus": consensus,
            "consensus_label": config.RESULT_LABELS[consensus],
        }

    def summary(self) -> Dict[str, object]:
        return {
            "ga_generation": self.ga.generation,
            "ga_best_fitness": round(self.ga.best_fitness, 4),
            "ql_steps": self.ql.total_steps,
            "ql_accuracy": round(self.ql.accuracy, 4),
            "ql_epsilon": round(self.ql.epsilon, 4),
            "ql_q_table_size": len(self.ql.q_table),
            "lstm_params": sum(p.numel() for p in self.lstm.model.parameters())
        }
