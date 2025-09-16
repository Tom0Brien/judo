# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from scipy.stats import levy_stable

from judo.gui import slider
from judo.optimizers.base import Optimizer, OptimizerConfig


@slider("exploration_rate", 0.01, 0.5, 0.05)
@slider("diversity_threshold", 0.1, 2.0, 0.1)
@slider("memory_factor", 0.1, 0.9, 0.05)
@dataclass
class AMBSConfig(OptimizerConfig):
    """Configuration for Adaptive Multi-Armed Bandit Sampling."""
    
    # Bandit parameters
    exploration_rate: float = 0.1  # Epsilon for epsilon-greedy strategy selection
    ucb_confidence: float = 2.0    # UCB confidence parameter
    
    # Sampling strategy parameters
    gaussian_sigma: float = 0.1     # Standard Gaussian sampling
    uniform_range: float = 0.2      # Uniform sampling range
    levy_alpha: float = 1.5         # Lévy flight alpha parameter
    cauchy_gamma: float = 0.05      # Cauchy distribution scale
    
    # Diversity and adaptation parameters
    diversity_threshold: float = 0.5  # Minimum diversity to maintain
    memory_factor: float = 0.7        # How much to weight historical performance
    adaptation_window: int = 10       # Window for performance evaluation


class AMBSSampler:
    """Individual sampling strategy for AMBS."""
    
    def __init__(self, name: str, sample_func, sigma_param: str):
        self.name = name
        self.sample_func = sample_func
        self.sigma_param = sigma_param
        self.total_reward = 0.0
        self.num_samples = 0
        self.recent_rewards: List[float] = []
        
    def get_average_reward(self) -> float:
        """Get average reward for this sampler."""
        if self.num_samples == 0:
            return 0.0
        return self.total_reward / self.num_samples
    
    def update_performance(self, reward: float, memory_factor: float, window_size: int):
        """Update performance metrics."""
        self.total_reward = self.total_reward * memory_factor + reward
        self.num_samples = self.num_samples * memory_factor + 1
        
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > window_size:
            self.recent_rewards.pop(0)


class AMBS(Optimizer[AMBSConfig]):
    """Adaptive Multi-Armed Bandit Sampling optimizer.
    
    This optimizer employs multiple diverse sampling strategies simultaneously:
    1. Gaussian sampling (like MPPI)
    2. Uniform sampling (for exploration)
    3. Lévy flight sampling (for rare long jumps)
    4. Cauchy sampling (heavy-tailed for escaping local optima)
    5. Elite-guided sampling (like CEM)
    
    A multi-armed bandit approach adaptively weights these strategies based on
    their performance, with diversity preservation mechanisms.
    """

    def __init__(self, config: AMBSConfig, nu: int) -> None:
        """Initialize the AMBS optimizer."""
        super().__init__(config, nu)
        
        # Initialize sampling strategies
        self.samplers = {
            'gaussian': AMBSSampler('gaussian', self._gaussian_sample, 'gaussian_sigma'),
            'uniform': AMBSSampler('uniform', self._uniform_sample, 'uniform_range'),
            'levy': AMBSSampler('levy', self._levy_sample, 'levy_alpha'),
            'cauchy': AMBSSampler('cauchy', self._cauchy_sample, 'cauchy_gamma'),
            'elite': AMBSSampler('elite', self._elite_sample, 'gaussian_sigma')
        }
        
        # Strategy weights (will be adapted over time)
        self.strategy_weights = np.ones(len(self.samplers)) / len(self.samplers)
        
        # Elite memory for elite-guided sampling
        self.elite_knots = None
        self.elite_diversity = 1.0
        
        # Performance tracking
        self.iteration_count = 0

    def _gaussian_sample(self, nominal_knots: np.ndarray, n_samples: int) -> np.ndarray:
        """Standard Gaussian sampling around nominal."""
        sigma = self.config.gaussian_sigma
        if self.use_noise_ramp:
            ramp = self.noise_ramp * np.linspace(1 / self.num_nodes, 1, self.num_nodes)[:, None]
            sigma = ramp * sigma
        return nominal_knots + sigma * np.random.randn(n_samples, self.num_nodes, self.nu)

    def _uniform_sample(self, nominal_knots: np.ndarray, n_samples: int) -> np.ndarray:
        """Uniform sampling around nominal."""
        range_val = self.config.uniform_range
        noise = np.random.uniform(-range_val, range_val, (n_samples, self.num_nodes, self.nu))
        return nominal_knots + noise

    def _levy_sample(self, nominal_knots: np.ndarray, n_samples: int) -> np.ndarray:
        """Lévy flight sampling for rare long jumps."""
        alpha = self.config.levy_alpha
        # Generate Lévy stable random variables
        noise = levy_stable.rvs(alpha, 0, size=(n_samples, self.num_nodes, self.nu)) * 0.1
        return nominal_knots + noise

    def _cauchy_sample(self, nominal_knots: np.ndarray, n_samples: int) -> np.ndarray:
        """Cauchy sampling for heavy-tailed exploration."""
        gamma = self.config.cauchy_gamma
        noise = np.random.standard_cauchy((n_samples, self.num_nodes, self.nu)) * gamma
        return nominal_knots + noise

    def _elite_sample(self, nominal_knots: np.ndarray, n_samples: int) -> np.ndarray:
        """Elite-guided sampling based on historical good solutions."""
        if self.elite_knots is None:
            # Fall back to Gaussian if no elites yet
            return self._gaussian_sample(nominal_knots, n_samples)
        
        # Sample around elite solutions with some diversity
        elite_idx = np.random.choice(len(self.elite_knots), n_samples)
        selected_elites = self.elite_knots[elite_idx]
        noise = np.random.randn(n_samples, self.num_nodes, self.nu) * self.config.gaussian_sigma
        return selected_elites + noise

    def _calculate_diversity(self, samples: np.ndarray) -> float:
        """Calculate diversity of samples using average pairwise distance."""
        if len(samples) < 2:
            return 1.0
        
        flat_samples = samples.reshape(len(samples), -1)
        distances = []
        for i in range(len(flat_samples)):
            for j in range(i + 1, len(flat_samples)):
                distances.append(np.linalg.norm(flat_samples[i] - flat_samples[j]))
        
        return np.mean(distances) if distances else 1.0

    def _update_strategy_weights(self, strategy_performances: Dict[str, float]):
        """Update strategy weights using UCB1 bandit algorithm."""
        total_trials = self.iteration_count + 1
        
        # Calculate UCB values for each strategy
        ucb_values = []
        for i, (name, sampler) in enumerate(self.samplers.items()):
            if sampler.num_samples == 0:
                ucb_values.append(float('inf'))  # Unsampled strategies get priority
            else:
                avg_reward = sampler.get_average_reward()
                confidence = self.config.ucb_confidence * np.sqrt(np.log(total_trials) / sampler.num_samples)
                ucb_values.append(avg_reward + confidence)
        
        # Epsilon-greedy with UCB
        if np.random.random() < self.config.exploration_rate:
            # Exploration: uniform random selection
            self.strategy_weights = np.ones(len(self.samplers)) / len(self.samplers)
        else:
            # Exploitation: weight based on UCB values
            ucb_values = np.array(ucb_values)
            ucb_values = np.where(np.isinf(ucb_values), np.max(ucb_values[np.isfinite(ucb_values)]) + 1, ucb_values)
            
            # Softmax to convert to probabilities
            exp_values = np.exp(ucb_values - np.max(ucb_values))
            self.strategy_weights = exp_values / np.sum(exp_values)

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Sample control knots using adaptive multi-armed bandit approach."""
        num_rollouts = self.num_rollouts
        
        # Allocate samples to strategies based on weights
        n_per_strategy = np.random.multinomial(num_rollouts - 1, self.strategy_weights)
        
        sampled_knots = [nominal_knots[None]]  # Always include nominal
        strategy_assignments = [len(self.samplers)]  # Nominal gets special index
        
        # Sample from each strategy
        for i, (name, sampler) in enumerate(self.samplers.items()):
            if n_per_strategy[i] > 0:
                samples = sampler.sample_func(nominal_knots, n_per_strategy[i])
                sampled_knots.append(samples)
                strategy_assignments.extend([i] * n_per_strategy[i])
        
        self.last_strategy_assignments = np.array(strategy_assignments)
        return np.concatenate(sampled_knots)

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update nominal knots and strategy weights based on performance."""
        
        # Update strategy performance tracking
        strategy_performances = {}
        for i, (name, sampler) in enumerate(self.samplers.items()):
            strategy_mask = self.last_strategy_assignments == i
            if np.any(strategy_mask):
                strategy_reward = np.mean(rewards[strategy_mask])
                sampler.update_performance(
                    strategy_reward, 
                    self.config.memory_factor, 
                    self.config.adaptation_window
                )
                strategy_performances[name] = strategy_reward
        
        # Update strategy weights
        self._update_strategy_weights(strategy_performances)
        
        # Elite selection for future guidance
        n_elites = max(2, self.num_rollouts // 8)
        elite_indices = np.argsort(rewards)[-n_elites:]
        self.elite_knots = sampled_knots[elite_indices]
        
        # Calculate diversity
        diversity = self._calculate_diversity(sampled_knots)
        self.elite_diversity = diversity
        
        # Weighted combination based on rewards with diversity preservation
        if diversity < self.config.diversity_threshold:
            # Low diversity: use more exploration
            weights = np.ones(len(rewards)) / len(rewards)
        else:
            # Good diversity: use reward-weighted combination
            costs = -rewards
            beta = np.min(costs)
            weights = np.exp(-(costs - beta) / 0.1)  # Temperature = 0.1
            weights = weights / np.sum(weights)
        
        nominal_knots = np.sum(weights[:, None, None] * sampled_knots, axis=0)
        
        self.iteration_count += 1
        return nominal_knots

    def stop_cond(self) -> bool:
        """Stop if diversity becomes too low and performance stagnates."""
        if self.iteration_count < 5:  # Don't stop too early
            return False
        
        # Check if all strategies are performing similarly (convergence)
        performances = [sampler.get_average_reward() for sampler in self.samplers.values() 
                       if sampler.num_samples > 0]
        
        if len(performances) > 1:
            performance_std = np.std(performances)
            return (performance_std < 0.01 and 
                   self.elite_diversity < self.config.diversity_threshold * 0.5)
        
        return False 