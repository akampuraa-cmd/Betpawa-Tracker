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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config

logger = logging.getLogger(__name__)

# Action space: three possible predictions
ACTIONS = [config.RESULT_LOSS, config.RESULT_DRAW, config.RESULT_WIN]
N_ACTIONS = len(ACTIONS)
MODEL_STATE_VERSION = 2


def _as_int(value: object) -> int:
    return int(cast(Any, value))


def _as_float(value: object) -> float:
    return float(cast(Any, value))


def _as_list(value: object) -> List[object]:
    if not isinstance(value, list):
        raise ValueError("Expected checkpoint list")
    return value


def _as_dict(value: object) -> Dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("Expected checkpoint dict")
    return value


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

    Features (length = window * 4 + 4 + 4 if include_odds else window * 4 + 4):
      - last *window* outcomes one-hot encoded as [L, D, W] blocks
      - last *window* MUN goal counts (normalised by 5)
      - rolling mean goals scored   (last window)
      - rolling mean goals conceded (last window)
      - win rate   (last window)
      - draw rate  (last window)
      - odds_present, home_odds, draw_odds, away_odds (if include_odds is True)
    """
    def _encode(o: int) -> np.ndarray:
        if o == config.RESULT_LOSS:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if o == config.RESULT_DRAW:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return np.array([0.0, 0.0, 1.0], dtype=float)

    padded_out = ([config.RESULT_DRAW] * window + outcomes)[-window:]
    padded_goals = ([(0, 0)] * window + goals)[-window:]

    out_enc = np.concatenate([_encode(o) for o in padded_out])
    mun_g = np.array([min(g[0], 5) / 5.0 for g in padded_goals], dtype=float)
    opp_g = np.array([min(g[1], 5) / 5.0 for g in padded_goals], dtype=float)

    mean_mun = float(np.mean(mun_g))
    mean_opp = float(np.mean(opp_g))
    win_rate = float(np.mean([1.0 if o == config.RESULT_WIN else 0.0 for o in padded_out]))
    draw_rate = float(np.mean([1.0 if o == config.RESULT_DRAW else 0.0 for o in padded_out]))

    features = [out_enc, mun_g, [mean_mun, mean_opp, win_rate, draw_rate]]

    if include_odds:
        if odds:
            # Normalize odds and add an explicit presence flag.
            norm_odds = [o / 10.0 if o else 0.0 for o in odds]
            features.append([1.0, *norm_odds])
        else:
            features.append([0.0, 0.0, 0.0, 0.0])

    return np.concatenate(features)


def _feature_size(window: int, include_odds: bool = False) -> int:
    base = window * 4 + 4
    return base + 4 if include_odds else base


def _build_training_matrices(
    outcomes: List[int],
    goals: List[Tuple[int, int]],
    window: int,
    include_odds: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the cached feature matrix and labels used by GA fitness."""
    if len(outcomes) < 2:
        return np.empty((0, _feature_size(window, include_odds))), np.empty((0,), dtype=int)

    start = min(window, len(outcomes) - 1)
    features = [
        _build_features(outcomes[:i], goals[:i], window, include_odds=include_odds)
        for i in range(start, len(outcomes))
    ]
    return np.vstack(features), np.array(outcomes[start:], dtype=int)


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
        seed: Optional[int] = None,
        mutation_rate_floor: float = getattr(config, "GA_MUTATION_RATE_FLOOR", 0.01),
        mutation_rate_ceiling: float = getattr(config, "GA_MUTATION_RATE_CEILING", 0.35),
        early_stop_patience: int = getattr(config, "GA_EARLY_STOP_PATIENCE", 40),
        early_stop_min_delta: float = getattr(config, "GA_EARLY_STOP_MIN_DELTA", 0.0001),
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.feature_window = feature_window
        self.include_odds = include_odds
        self.seed = seed
        self.mutation_rate_floor = mutation_rate_floor
        self.mutation_rate_ceiling = mutation_rate_ceiling
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.n_features = _feature_size(feature_window, include_odds)
        self.gene_length = N_ACTIONS * (self.n_features + 1)  # W + b

        self.population: List[np.ndarray] = []
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = 0.0
        self.generation: int = 0
        self.fitness_history: List[float] = []
        self._stagnation_generations: int = 0
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)

        self._init_population()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_population(self) -> None:
        self.population = [
            self.np_rng.standard_normal(self.gene_length) * 0.5
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

    def _evaluate_population_fitness(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        if feature_matrix.size == 0:
            return np.zeros(len(self.population), dtype=float)

        population = np.vstack(self.population)
        w_size = N_ACTIONS * self.n_features
        weights = population[:, :w_size].reshape(len(self.population), N_ACTIONS, self.n_features)
        biases = population[:, w_size:]
        logits = np.einsum("nf,paf->pna", feature_matrix, weights) + biases[:, None, :]
        preds = np.argmax(logits, axis=2)
        return np.mean(preds == labels, axis=1)

    def predict(self, features: np.ndarray) -> int:
        """Predict outcome using the best individual found so far."""
        if self.best_individual is None:
            return self.rng.choice(ACTIONS)
        proba = self._predict_proba(self.best_individual, features)
        return ACTIONS[int(np.argmax(proba))]

    # ── Fitness ──────────────────────────────────────────────────────────────

    def evaluate_fitness(
        self,
        individual: np.ndarray,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Vectorized leave-one-out accuracy over a cached feature matrix.
        """
        if feature_matrix.size == 0:
            return 0.0

        W, b = self._decode(individual)
        logits = feature_matrix @ W.T + b
        preds = np.argmax(logits, axis=1)
        return float(np.mean(preds == labels))

    # ── Genetic operators ─────────────────────────────────────────────────────

    def _select(self, fitnesses: Sequence[float]) -> np.ndarray:
        """Tournament selection (k=3)."""
        k = min(3, len(self.population))
        idxs = self.rng.sample(range(len(self.population)), k)
        best_idx = max(idxs, key=lambda i: fitnesses[i])
        return self.population[best_idx].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()
        blend = self.np_rng.random(len(p1))
        c1 = blend * p1 + (1.0 - blend) * p2
        c2 = blend * p2 + (1.0 - blend) * p1
        return c1, c2

    def _current_mutation_rate(self) -> float:
        rate = self.mutation_rate * (1.0 + 0.05 * self._stagnation_generations)
        return min(self.mutation_rate_ceiling, max(self.mutation_rate_floor, rate))

    def _mutate(self, individual: np.ndarray, mutation_rate: Optional[float] = None) -> np.ndarray:
        rate = self.mutation_rate if mutation_rate is None else mutation_rate
        if rate <= 0.0:
            return individual
        mask = self.np_rng.random(individual.size) < rate
        if np.any(mask):
            individual[mask] += self.np_rng.standard_normal(mask.sum()) * 0.1
        return individual

    # ── Training ─────────────────────────────────────────────────────────────

    def train_one_generation(
        self,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
    ) -> float:
        """Run one generation.  Returns the best fitness of this generation."""
        feature_matrix, labels = _build_training_matrices(
            outcomes,
            goals,
            self.feature_window,
            include_odds=self.include_odds,
        )
        return self.train_one_generation_from_cache(feature_matrix, labels)

    def train(
        self,
        outcomes: List[int],
        goals: List[Tuple[int, int]],
        generations: Optional[int] = None,
        callback=None,
    ) -> None:
        """Train for *generations* generations (default: self.generations)."""
        n = generations or self.generations
        feature_matrix, labels = _build_training_matrices(
            outcomes,
            goals,
            self.feature_window,
            include_odds=self.include_odds,
        )
        for g in range(n):
            fit = self.train_one_generation_from_cache(feature_matrix, labels)
            if callback:
                callback(generation=self.generation, best_fitness=fit)
            if self._stagnation_generations >= self.early_stop_patience:
                logger.info(
                    "GA early stopping after low improvement for %s generations",
                    self._stagnation_generations,
                )
                break

    def train_one_generation_from_cache(
        self,
        feature_matrix: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Run one generation using cached training features and labels."""
        if feature_matrix.size == 0:
            return 0.0

        fitnesses = self._evaluate_population_fitness(feature_matrix, labels)
        fitness_list = fitnesses.tolist()

        elite_count = min(self.elitism, len(self.population))
        elite_idx = np.argpartition(fitnesses, -elite_count)[-elite_count:]
        elite_idx = elite_idx[np.argsort(fitnesses[elite_idx])[::-1]]
        new_pop = [self.population[i].copy() for i in elite_idx]

        best_idx = int(np.argmax(fitnesses))
        gen_best = float(fitnesses[best_idx])
        best_candidate = self.population[best_idx].copy()
        mutation_rate = self._current_mutation_rate()

        while len(new_pop) < self.population_size:
            p1 = self._select(fitness_list)
            p2 = self._select(fitness_list)
            c1, c2 = self._crossover(p1, p2)
            new_pop.append(self._mutate(c1, mutation_rate))
            if len(new_pop) < self.population_size:
                new_pop.append(self._mutate(c2, mutation_rate))

        self.population = new_pop[:self.population_size]
        self.generation += 1

        if self.best_individual is None or gen_best > self.best_fitness + self.early_stop_min_delta:
            self.best_fitness = gen_best
            self.best_individual = best_candidate
            self._stagnation_generations = 0
        else:
            self._stagnation_generations += 1

        self.fitness_history.append(gen_best)
        return gen_best

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
        if _as_int(state["population_size"]) != self.population_size:
            raise ValueError("Population size mismatch in checkpoint")
        if _as_int(state["feature_window"]) != self.feature_window or bool(state["include_odds"]) != self.include_odds:
            raise ValueError("Feature configuration mismatch in checkpoint")
        if _as_int(state.get("n_features", self.n_features)) != self.n_features:
            raise ValueError("Feature count mismatch in checkpoint")
        if _as_int(state.get("gene_length", self.gene_length)) != self.gene_length:
            raise ValueError("Gene length mismatch in checkpoint")

        population_data = _as_list(state["population"])
        population = [np.array(ind, dtype=float) for ind in population_data]
        if not population:
            raise ValueError("Checkpoint does not contain a population")
        if any(ind.size != self.gene_length for ind in population):
            raise ValueError("Population shape mismatch in checkpoint")

        self.population = population
        if state["best_individual"] is None:
            self.best_individual = None
        else:
            best_individual = np.array(state["best_individual"], dtype=float)
            if best_individual.size != self.gene_length:
                raise ValueError("Best individual shape mismatch in checkpoint")
            self.best_individual = best_individual
        self.best_fitness = _as_float(state["best_fitness"])
        self.generation = _as_int(state["generation"])
        self.fitness_history = [_as_float(v) for v in _as_list(state.get("fitness_history", []))]

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
        replay_passes: int = getattr(config, "QL_REPLAY_PASSES", 3),
        replay_min_td_error: float = getattr(config, "QL_REPLAY_MIN_TD_ERROR", 0.0001),
        replay_early_stop_patience: int = getattr(config, "QL_REPLAY_EARLY_STOP_PATIENCE", 2),
    ):
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_window = state_window
        self.replay_passes = replay_passes
        self.replay_min_td_error = replay_min_td_error
        self.replay_early_stop_patience = replay_early_stop_patience

        self.q_table: Dict[Tuple[State, int], float] = {}
        self.total_steps = 0
        self.total_reward = 0.0
        self.correct_predictions = 0
        self.reward_history: List[float] = []
        self._replay_stagnation: int = 0

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

    def _build_replay_transitions(self, outcomes: List[int]) -> List[Tuple[State, int, float, State]]:
        w = self.state_window
        transitions: List[Tuple[State, int, float, State]] = []
        for i in range(w, len(outcomes)):
            state = self._get_state(outcomes[:i])
            actual = outcomes[i]
            next_state = self._get_state(outcomes[: i + 1])
            transitions.append((state, actual, 1.0, next_state))
        return transitions

    def _train_on_transition_batch(
        self,
        transitions: List[Tuple[State, int, float, State]],
        callback=None,
    ) -> float:
        if not transitions:
            return 0.0

        td_errors: List[float] = []
        for state, actual, reward, next_state in transitions:
            action = self.choose_action(state)
            reward_value = reward if action == actual else -reward
            best_next = max(self._q(next_state, a) for a in ACTIONS)
            old_q = self._q(state, action)
            target = reward_value + self.gamma * best_next
            td_error = target - old_q
            new_q = old_q + self.lr * td_error
            self._set_q(state, action, new_q)

            self.total_steps += 1
            self.total_reward += reward_value
            if reward_value > 0:
                self.correct_predictions += 1
            self.reward_history.append(reward_value)
            self._decay_epsilon()
            td_errors.append(abs(td_error))

            if callback:
                callback(
                    step=self.total_steps,
                    reward=reward_value,
                    epsilon=self.epsilon,
                    q_table_size=len(self.q_table),
                )

        return float(np.mean(td_errors))

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        outcomes: List[int],
        callback=None,
    ) -> None:
        """
        Replay all stored outcomes sequentially to train the Q-table.
        """
        transitions = self._build_replay_transitions(outcomes)
        passes = max(1, self.replay_passes)
        for _ in range(passes):
            mean_td_error = self._train_on_transition_batch(transitions, callback=callback)
            if mean_td_error <= self.replay_min_td_error:
                self._replay_stagnation += 1
            else:
                self._replay_stagnation = 0
            if self._replay_stagnation >= self.replay_early_stop_patience:
                logger.info(
                    "QL early stopping after low TD error (%.6f)",
                    mean_td_error,
                )
                break

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
        if state.get("version") not in {1, MODEL_STATE_VERSION}:
            raise ValueError("Unsupported QLearning checkpoint version")
        if _as_int(state["state_window"]) != self.state_window:
            raise ValueError("State window mismatch in checkpoint")

        self.lr = _as_float(state["learning_rate"])
        self.gamma = _as_float(state["discount"])
        self.epsilon = _as_float(state["epsilon"])
        self.epsilon_decay = _as_float(state["epsilon_decay"])
        self.epsilon_min = _as_float(state["epsilon_min"])

        q_table_data = _as_dict(state["q_table"])
        self.q_table = {
            cast(Tuple[State, int], key): _as_float(value)
            for key, value in q_table_data.items()
        }
        self.total_steps = _as_int(state["total_steps"])
        self.total_reward = _as_float(state["total_reward"])
        self.correct_predictions = _as_int(state["correct_predictions"])
        self.reward_history = [_as_float(v) for v in _as_list(state.get("reward_history", []))]

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
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int = 3, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
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
        dropout: float = getattr(config, "LSTM_DROPOUT", 0.2),
        batch_size: int = getattr(config, "LSTM_BATCH_SIZE", 32),
        validation_split: float = getattr(config, "LSTM_VALIDATION_SPLIT", 0.2),
        early_stop_patience: int = getattr(config, "LSTM_EARLY_STOP_PATIENCE", 12),
        min_delta: float = getattr(config, "LSTM_MIN_DELTA", 0.0001),
        lr_factor: float = getattr(config, "LSTM_LR_FACTOR", 0.5),
        lr_patience: int = getattr(config, "LSTM_LR_PATIENCE", 4),
    ):
        self.seq_length = seq_length
        self.input_size = 3  # encoded outcome, mun goals, opp goals
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stop_patience = early_stop_patience
        self.min_delta = min_delta
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        
        self.model = PyTorchLSTM(self.input_size, hidden_size, num_layers, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=1e-5,
        )

    def _step_feature(self, outcome: int, mun_g: int, opp_g: int) -> List[float]:
        enc = {config.RESULT_LOSS: 0.0, config.RESULT_DRAW: 0.5, config.RESULT_WIN: 1.0}[outcome]
        return [enc, mun_g / 5.0, opp_g / 5.0]

    def _goal_or_default(self, goals: List[Tuple[int, int]], index: int) -> Tuple[int, int]:
        if 0 <= index < len(goals):
            return goals[index]
        return (0, 0)

    def _build_dataset(self, outcomes: List[int], goals: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = [], []
        for i in range(len(outcomes) - self.seq_length):
            seq_X = [
                self._step_feature(outcomes[j], *self._goal_or_default(goals, j))
                for j in range(i, i + self.seq_length)
            ]
            X.append(seq_X)
            y.append(outcomes[i + self.seq_length])

        if not X:
            return torch.empty((0, self.seq_length, self.input_size), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return X_tensor, y_tensor

    def train(self, outcomes: List[int], goals: List[Tuple[int, int]], epochs: int = 50) -> None:
        if len(outcomes) <= self.seq_length:
            return

        X_tensor, y_tensor = self._build_dataset(outcomes, goals)
        if X_tensor.size(0) == 0:
            return

        split_idx = int(len(X_tensor) * (1.0 - self.validation_split))
        split_idx = max(1, min(split_idx, len(X_tensor) - 1)) if len(X_tensor) > 1 else len(X_tensor)
        train_X, val_X = X_tensor[:split_idx], X_tensor[split_idx:]
        train_y, val_y = y_tensor[:split_idx], y_tensor[split_idx:]

        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=False,
        )

        self.model.train()
        best_val_loss = float("inf")
        stalled_epochs = 0
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_X)

            train_loss = epoch_loss / len(train_dataset)
            if len(val_X) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(val_X)
                    val_loss = float(self.criterion(val_logits, val_y).item())
                self.scheduler.step(val_loss)
                if val_loss + self.min_delta < best_val_loss:
                    best_val_loss = val_loss
                    stalled_epochs = 0
                else:
                    stalled_epochs += 1
                self.model.train()
                if stalled_epochs >= self.early_stop_patience:
                    break
            else:
                self.scheduler.step(train_loss)

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
            seq_X.append(self._step_feature(outcomes[i], *self._goal_or_default(goals, i)))

        try:
            with torch.no_grad():
                x_tensor = torch.tensor([seq_X], dtype=torch.float32)
                logits = self.model(x_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                pred = int(torch.argmax(probs).item())
                conf = float(probs[pred].item())
            return ACTIONS[pred], conf
        except Exception as exc:
            logger.warning("LSTM prediction failed: %s", exc)
            return config.RESULT_DRAW, 0.0

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": MODEL_STATE_VERSION,
            "seq_length": self.seq_length,
            "input_size": self.input_size,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if state.get("version") not in {1, MODEL_STATE_VERSION}:
            raise ValueError("Unsupported LSTMPredictor checkpoint version")
        if _as_int(state["seq_length"]) != self.seq_length or _as_int(state["input_size"]) != self.input_size:
            raise ValueError("LSTM configuration mismatch in checkpoint")

        self.model.load_state_dict(cast(Mapping[str, Any], _as_dict(state["model_state_dict"])))
        self.optimizer.load_state_dict(cast(Dict[str, Any], _as_dict(state["optimizer_state_dict"])))
        scheduler_state = state.get("scheduler_state_dict")
        if scheduler_state is not None:
            self.scheduler.load_state_dict(cast(Dict[str, Any], _as_dict(scheduler_state)))

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
        version = state.get("version")
        if version not in {1, MODEL_STATE_VERSION}:
            raise ValueError("Unsupported BetpawaAI checkpoint version")

        if version == MODEL_STATE_VERSION:
            self.ga.load_state_dict(_as_dict(state["ga"]))
            self.ql.load_state_dict(_as_dict(state["ql"]))
            self.lstm.load_state_dict(_as_dict(state["lstm"]))
            return

        # Legacy checkpoints may contain GA weights that are incompatible with
        # the current feature layout. Restore Q-learning and LSTM state, and
        # keep the freshly initialised GA if the old GA block cannot load.
        try:
            self.ga.load_state_dict(_as_dict(state["ga"]))
        except Exception as exc:
            logger.warning("Skipping legacy GA checkpoint: %s", exc)

        self.ql.load_state_dict(_as_dict(state["ql"]))
        self.lstm.load_state_dict(_as_dict(state["lstm"]))

    def save(self, path: Optional[str] = None) -> bool:
        checkpoint_path = path or self.checkpoint_path
        if not checkpoint_path:
            return False

        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def checkpoint_exists(self, path: Optional[str] = None) -> bool:
        checkpoint_path = path or self.checkpoint_path
        if not checkpoint_path:
            return False
        return Path(checkpoint_path).exists()

    def checkpoint_info(self, path: Optional[str] = None) -> Dict[str, object]:
        checkpoint_path = path or self.checkpoint_path
        if not checkpoint_path:
            return {
                "exists": False,
                "path": None,
                "size_bytes": 0,
            }

        checkpoint = Path(checkpoint_path)
        exists = checkpoint.exists()
        return {
            "exists": exists,
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size if exists else 0,
        }

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
        for _ in range(3):
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
