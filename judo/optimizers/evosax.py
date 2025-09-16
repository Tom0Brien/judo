# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Type

import numpy as np
import jax
import jax.numpy as jnp
from evosax.algorithms.base import EvolutionaryAlgorithm
from typing import Literal

from judo.gui import slider
from judo.optimizers.base import Optimizer, OptimizerConfig


@slider("sigma", 0.001, 1.0, 0.01)
@dataclass
class EvosaxConfig(OptimizerConfig):
    """Configuration for evosax-based optimizers."""
    algorithm_name: Literal["CMA_ES", "OpenAI_ES", "xNES", "SNES", "RandomSearch", "SimulatedAnnealing", "PGPE", "ARS", "Sep_CMA_ES", "GradientlessDescent", "SAMR_GA", "SimpleGA", "DifferentialEvolution", "PSO"] = "SAMR_GA"
    sigma: float = 0.1
    algorithm_kwargs: dict = field(default_factory=dict)


class Evosax(Optimizer[EvosaxConfig]):
    """Generic evosax optimizer wrapper for Judo.
    
    This optimizer provides a wrapper around evosax evolutionary algorithms,
    allowing them to be used within the Judo framework. The optimizer uses
    JAX for computation but converts between numpy and JAX arrays as needed
    to maintain compatibility with Judo's numpy-based interface.
    
    Supported algorithms include CMA_ES, OpenAI_ES, xNES, SNES, RandomSearch,
    SimulatedAnnealing, and many others from the evosax library.
    """

    def __init__(self, config: EvosaxConfig, nu: int) -> None:
        """Initialize the evosax optimizer.
        
        Args:
            config: Configuration for the evosax optimizer
            nu: Number of control dimensions
        """
        print("Initializing evosax optimizer")
        
        # Initialize config attribute to avoid issues during setup
        self._config = config
        
        # Store the algorithm name to detect changes
        self._current_algorithm_name = config.algorithm_name
        
        # Initialize the evosax strategy before calling super().__init__
        # to avoid circular dependency with config setter
        self._init_strategy(config, nu)
        
        # Call super().__init__ after strategy is initialized
        super().__init__(config, nu)

    def _init_strategy(self, config: EvosaxConfig = None, nu: int = None) -> None:
        """Initialize or reinitialize the evosax strategy."""
        if config is None:
            config = self.config
        if nu is None:
            nu = self.nu
        
        # Initialize JAX random key
        self.rng_key = jax.random.PRNGKey(42)
        
        # Get the evosax algorithm
        algorithm_class = self._get_algorithm_class(config.algorithm_name)
        
        # Initialize the evolution strategy
        # Following the Hydrax pattern for evosax initialization
        self.strategy = algorithm_class(
            population_size=config.num_rollouts,
            # Only to inform the dimension to evosax 
            solution=jnp.zeros(config.num_nodes * nu), 
            **config.algorithm_kwargs
        )
        
        # Get default parameters
        self.es_params = self.strategy.default_params
        
        # Initialize optimizer state
        self.rng_key, init_key = jax.random.split(self.rng_key)
        initial_mean = jnp.zeros(config.num_nodes * nu)
        
        # Different algorithm types have different initialization APIs
        if self._is_population_based_algorithm(algorithm_class):
            # Population-based algorithms need an initial population and fitness
            # Create random initial population
            population_shape = (config.num_rollouts, config.num_nodes * nu)
            initial_population = jax.random.normal(init_key, population_shape) * config.sigma
            # Create dummy fitness values (will be updated in first iteration)
            initial_fitness = jnp.zeros(config.num_rollouts)
            
            self.es_state = self.strategy.init(
                key=init_key,
                population=initial_population,
                fitness=initial_fitness,
                params=self.es_params
            )
        else:
            # Distribution-based algorithms use mean in init
            self.es_state = self.strategy.init(
                key=init_key, 
                mean=initial_mean,
                params=self.es_params
            )

    @property
    def config(self) -> EvosaxConfig:
        """Get the current config."""
        return self._config

    @config.setter
    def config(self, new_config: EvosaxConfig) -> None:
        """Set the config and reinitialize strategy if algorithm changed."""
        old_algorithm = getattr(self, '_current_algorithm_name', None)
        
        # Update the config
        self._config = new_config
        
        # Check if algorithm changed and if we're not in initial setup
        if (old_algorithm is not None and 
            old_algorithm != new_config.algorithm_name and 
            hasattr(self, 'nu')):
            print(f"Algorithm changed from {old_algorithm} to {new_config.algorithm_name}, reinitializing...")
            self._current_algorithm_name = new_config.algorithm_name
            self._init_strategy(new_config, self.nu)

    def _get_algorithm_class(self, algorithm_name: str) -> Type[EvolutionaryAlgorithm]:
        """Get the evosax algorithm class by name.
        
        Args:
            algorithm_name: Name of the algorithm (e.g., 'CMA_ES', 'OpenAI_ES')
            
        Returns:
            The evosax algorithm class
            
        Raises:
            ImportError: If the algorithm is not available
        """
        try:
            if algorithm_name == "CMA_ES":
                from evosax.algorithms.distribution_based import CMA_ES
                return CMA_ES
            elif algorithm_name == "OpenAI_ES":
                from evosax.algorithms.distribution_based import OpenAI_ES
                return OpenAI_ES
            elif algorithm_name == "xNES":
                from evosax.algorithms.distribution_based import xNES
                return xNES
            elif algorithm_name == "SNES":
                from evosax.algorithms.distribution_based import SNES
                return SNES
            elif algorithm_name == "RandomSearch":
                from evosax.algorithms.distribution_based import RandomSearch
                return RandomSearch
            elif algorithm_name == "SimulatedAnnealing":
                from evosax.algorithms.distribution_based import SimulatedAnnealing
                return SimulatedAnnealing
            elif algorithm_name == "PGPE":
                from evosax.algorithms.distribution_based import PGPE
                return PGPE
            elif algorithm_name == "ARS":
                from evosax.algorithms.distribution_based import ARS
                return ARS
            elif algorithm_name == "Sep_CMA_ES":
                from evosax.algorithms.distribution_based import Sep_CMA_ES
                return Sep_CMA_ES
            elif algorithm_name == "GradientlessDescent":
                from evosax.algorithms.distribution_based import GradientlessDescent
                return GradientlessDescent
            elif algorithm_name == "SAMR_GA":
                from evosax.algorithms.population_based import SAMR_GA
                return SAMR_GA
            elif algorithm_name == "SimpleGA":
                from evosax.algorithms.population_based import SimpleGA
                return SimpleGA
            elif algorithm_name == "DifferentialEvolution":
                from evosax.algorithms.population_based import DifferentialEvolution
                return DifferentialEvolution
            elif algorithm_name == "PSO":
                from evosax.algorithms.population_based import PSO
                return PSO
            else:
                raise ImportError(f"Unknown evosax algorithm: {algorithm_name}")
        except ImportError as e:
            raise ImportError(f"Failed to import evosax algorithm {algorithm_name}: {e}")

    def _is_population_based_algorithm(self, algorithm_class: Type[EvolutionaryAlgorithm]) -> bool:
        """Check if the algorithm is population-based (vs distribution-based).
        
        Args:
            algorithm_class: The evosax algorithm class
            
        Returns:
            True if the algorithm is population-based, False otherwise
        """
        # Check if the algorithm is from the population_based module
        module_name = algorithm_class.__module__
        return "population_based" in module_name

    @property
    def sigma(self) -> float:
        """Get the sigma value."""
        return self.config.sigma

    @property
    def algorithm_name(self) -> str:
        """Get the algorithm name."""
        return self.config.algorithm_name

    def sample_control_knots(self, nominal_knots: np.ndarray) -> np.ndarray:
        """Sample control knots using evosax algorithm.

        Args:
            nominal_knots: The nominal control input to sample from. Shape=(num_nodes, nu).

        Returns:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
        """
        # Convert numpy to JAX array and flatten
        nominal_flat = jnp.array(nominal_knots.flatten())
        
        # Handle sampling differently for distribution-based vs population-based algorithms
        if self._is_population_based_algorithm(type(self.strategy)):
            # For population-based algorithms, we can't update the mean
            # Just sample from the current state
            pass
        else:
            # For distribution-based algorithms, update the mean
            self.es_state = self.es_state.replace(mean=nominal_flat)
        
        # Sample from the evosax strategy
        self.rng_key, sample_key = jax.random.split(self.rng_key)
        samples, self.es_state = self.strategy.ask(
            sample_key,
            self.es_state,
            self.es_params
        )
        
        # Reshape to (num_rollouts, num_nodes, nu) and convert to numpy
        num_rollouts = self.num_rollouts
        num_nodes = self.num_nodes
        sampled_knots = np.array(samples).reshape(num_rollouts, num_nodes, self.nu)
        
        return sampled_knots

    def update_nominal_knots(self, sampled_knots: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """Update the nominal control knots based on sampled controls and rewards.

        Args:
            sampled_knots: The sampled control input. Shape=(num_rollouts, num_nodes, nu).
            rewards: The rewards for each sampled control input. Shape=(num_rollouts,).

        Returns:
            nominal_knots: The updated nominal control input. Shape=(num_nodes, nu).
        """
        # Convert rewards to costs (evosax minimizes, Judo maximizes)
        costs = -np.array(rewards)
        
        # Flatten sampled knots for evosax
        population = jnp.array(sampled_knots.reshape(self.num_rollouts, -1))
        fitness = jnp.array(costs)
        
        # Update the evosax state
        self.rng_key, tell_key = jax.random.split(self.rng_key)
        self.es_state, _ = self.strategy.tell(
            key=tell_key,
            population=population,
            fitness=fitness,
            state=self.es_state,
            params=self.es_params
        )
        
        # Handle getting nominal knots differently for different algorithm types
        if self._is_population_based_algorithm(type(self.strategy)):
            # For population-based algorithms, return the best individual from current population
            best_idx = jnp.argmin(fitness)  # Best fitness (lowest cost)
            updated_solution = population[best_idx]
            nominal_knots = np.array(updated_solution).reshape(self.num_nodes, self.nu)
        else:
            # For distribution-based algorithms, get the updated mean
            updated_mean = np.array(self.es_state.mean)
            nominal_knots = updated_mean.reshape(self.num_nodes, self.nu)
        
        return nominal_knots

    def stop_cond(self) -> bool:
        """Check if the optimization should stop.
        
        Returns:
            True if the optimization should stop based on evosax stopping criteria.
        """
        # Most evosax algorithms don't have built-in stopping criteria
        # We could implement some basic ones here if needed
        return False 