# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Type, Literal

import numpy as np
import jax
import jax.numpy as jnp
from evosax.algorithms.base import EvolutionaryAlgorithm

from judo.gui import slider
from judo.optimizers.base import Optimizer, OptimizerConfig


@slider("sigma", 0.001, 1.0, 0.01)
@slider("c_c", 0.001, 1.0, 0.01)
@slider("c_1", 0.001, 1.0, 0.01)
@slider("c_mu", 0.001, 2.0, 0.01)
@slider("c_sigma", 0.001, 1.0, 0.01)
@slider("d_sigma", 0.1, 5.0, 0.1)
@slider("cm", 0.1, 2.0, 0.1)
@slider("temperature", 0.001, 1.0, 0.01)
@slider("mutation_rate", 0.001, 0.5, 0.01)
@slider("elite_ratio", 0.05, 0.5, 0.05)
@slider("learning_rate", 0.001, 1.0, 0.01)
@dataclass
class EvosaxConfig(OptimizerConfig):
    """Configuration for evosax-based optimizers with individual algorithm parameters."""

    # Basic parameters
    sigma: float = 0.1
    algorithm_name: Literal[
        "CMA_ES", "Sep_CMA_ES", "xNES", "SNES", "PGPE", "ARS", 
        "SimulatedAnnealing", "GradientlessDescent",
        "DifferentialEvolution", "PSO", "SAMR_GA"
    ] = "CMA_ES"
    
    # CMA-ES specific parameters
    c_c: float = 0.0  # Cumulation parameter for covariance matrix (0 = auto)
    c_1: float = 0.0  # Learning rate for rank-one update (0 = auto)
    c_mu: float = 0.0  # Learning rate for rank-mu update (0 = auto)
    c_sigma: float = 0.0  # Cumulation parameter for step-size control (0 = auto)
    d_sigma: float = 0.0  # Damping factor for step-size adaptation (0 = auto)
    cm: float = 1.0  # Learning rate for mean update
    
    # Evolution Strategy parameters
    temperature: float = 0.1  # Temperature for MPPI-style algorithms
    mutation_rate: float = 0.1  # Mutation rate for GA/ES algorithms
    elite_ratio: float = 0.2  # Fraction of population to use as elites
    learning_rate: float = 0.01  # Learning rate for gradient-based ES
    
    # Advanced parameters
    use_antithetic_sampling: bool = False  # Use antithetic sampling for variance reduction
    use_fitness_shaping: bool = True  # Apply fitness shaping/ranking
    restart_strategy: str = "none"  # Restart strategy: "none", "ipop", "bipop"
    
    def get_algorithm_kwargs(self) -> dict:
        """Build algorithm-specific kwargs from individual parameters."""
        kwargs = {}
        
        # For now, be conservative and only pass parameters we know work
        # Most evosax algorithms don't accept custom parameters in their constructors
        # Instead, they use default_params that can be modified after initialization
        
        # CMA-ES variants typically don't accept custom parameters in constructor
        # The parameters are handled through the params object instead
        
        # Most parameters will be handled through the evosax default_params mechanism
        # For now, keep the constructor calls minimal to avoid parameter errors
        
        return kwargs


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
        algorithm_kwargs = config.get_algorithm_kwargs()
        self.strategy = algorithm_class(
            population_size=config.num_rollouts,
            # Only to inform the dimension to evosax 
            solution=jnp.zeros(config.num_nodes * nu), 
            **algorithm_kwargs
        )
        
        # Get default parameters
        self.es_params = self.strategy.default_params
        
        # Apply custom parameters to the evosax params object
        self._apply_custom_params(config)
        
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

    def _apply_custom_params(self, config: EvosaxConfig) -> None:
        """Apply custom parameters to the evosax params object dynamically.
        
        This method inspects the actual evosax params structure and only applies
        parameters that exist, making it robust across different algorithms.
        """
        # Get available parameter names from the evosax params object
        available_params = set()
        if hasattr(self.es_params, '__dataclass_fields__'):
            available_params = set(self.es_params.__dataclass_fields__.keys())
        elif hasattr(self.es_params, '_fields'):  # namedtuple
            available_params = set(self.es_params._fields)
        
        # Create a mapping from our config parameters to actual evosax parameter names
        # Based on inspection of real evosax algorithms
        param_mapping = {
            'sigma': ['std_init'],  # Main step-size parameter for most algorithms
            'c_c': ['c_c'],  # CMA-ES covariance cumulation
            'c_1': ['c_1'],  # CMA-ES rank-one learning rate
            'c_mu': ['c_mu'],  # CMA-ES rank-mu learning rate  
            'c_sigma': ['c_std'],  # CMA-ES step-size learning rate (actual name: c_std)
            'd_sigma': ['d_std'],  # CMA-ES step-size damping (actual name: d_std)
            'cm': ['c_mean'],  # CMA-ES mean learning rate (actual name: c_mean)
            'temperature': ['temperature_init'],  # Simulated Annealing
            'learning_rate': ['lr_std_init', 'std_lr'],  # Various ES learning rates
            'mutation_rate': ['differential_weight', 'crossover_rate'],  # DE parameters
            'elite_ratio': ['elitism'],  # Population-based algorithms
        }
        
        # Only apply parameters that exist and have non-default values
        params_to_update = {}
        
        for config_param, evosax_names in param_mapping.items():
            if hasattr(config, config_param):
                config_value = getattr(config, config_param)
                
                # Check if this is a non-default value worth applying
                if self._should_apply_param(config_param, config_value):
                    # Find the matching evosax parameter name
                    for evosax_name in evosax_names:
                        if evosax_name in available_params:
                            params_to_update[evosax_name] = config_value
                            break
        
        # Apply all valid parameter updates at once
        if params_to_update:
            try:
                self.es_params = self.es_params.replace(**params_to_update)
                print(f"Applied {len(params_to_update)} custom parameters for {config.algorithm_name}: {list(params_to_update.keys())}")
            except Exception as e:
                print(f"Warning: Could not apply some parameters for {config.algorithm_name}: {e}")
        else:
            print(f"No custom parameters applied for {config.algorithm_name} (using defaults)")

    def _should_apply_param(self, param_name: str, value: Any) -> bool:
        """Check if a parameter value should be applied (i.e., is non-default)."""
        # Define what constitutes "default" values that should be skipped
        defaults = {
            'sigma': 0.1,
            'c_c': 0.0,
            'c_1': 0.0,
            'c_mu': 0.0,
            'c_sigma': 0.0,
            'd_sigma': 0.0,
            'cm': 1.0,
            'temperature': 0.1,
            'learning_rate': 0.01,
            'mutation_rate': 0.1,
            'elite_ratio': 0.2,
        }
        
        default_value = defaults.get(param_name)
        if default_value is None:
            return True  # Unknown parameter, let it through
            
        # For numeric parameters, check if significantly different from default
        if isinstance(value, (int, float)):
            return abs(value - default_value) > 1e-6
        
        return value != default_value

    def get_supported_parameters(self) -> dict:
        """Get the parameters supported by the current algorithm.
        
        Returns:
            Dictionary mapping parameter names to their current values.
        """
        if not hasattr(self, 'es_params'):
            return {}
            
        supported = {}
        if hasattr(self.es_params, '__dataclass_fields__'):
            for field_name, field in self.es_params.__dataclass_fields__.items():
                value = getattr(self.es_params, field_name)
                supported[field_name] = value
        elif hasattr(self.es_params, '_fields'):  # namedtuple
            for field_name in self.es_params._fields:
                value = getattr(self.es_params, field_name)
                supported[field_name] = value
                
        return supported

    @property
    def config(self) -> EvosaxConfig:
        """Get the current config."""
        return self._config

    @config.setter
    def config(self, new_config: EvosaxConfig) -> None:
        """Set the config and update evosax parameters online."""
        old_config = getattr(self, '_config', None)
        old_algorithm = getattr(self, '_current_algorithm_name', None)
        
        # Update the config
        self._config = new_config
        
        # Check if algorithm changed and if we're not in initial setup
        algorithm_changed = (old_algorithm is not None and 
                           old_algorithm != new_config.algorithm_name and 
                           hasattr(self, 'nu'))
        
        if algorithm_changed:
            print(f"Algorithm changed from {old_algorithm} to {new_config.algorithm_name}, reinitializing...")
            self._current_algorithm_name = new_config.algorithm_name
            self._init_strategy(new_config, self.nu)
        elif old_config is not None and hasattr(self, 'es_params'):
            # Algorithm didn't change, but other parameters might have
            # Apply parameter updates online without reinitializing strategy
            if self._config_parameters_changed(old_config, new_config):
                print("Parameters changed, updating evosax strategy online...")
                self._apply_custom_params(new_config)

    def _config_parameters_changed(self, old_config: EvosaxConfig, new_config: EvosaxConfig) -> bool:
        """Check if any evosax-relevant parameters changed between configs.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            
        Returns:
            True if any parameters that affect evosax strategy changed
        """
        # List of parameters that affect evosax strategy behavior
        evosax_params = [
            'sigma', 'c_c', 'c_1', 'c_mu', 'c_sigma', 'd_sigma', 'cm',
            'temperature', 'learning_rate', 'mutation_rate', 'elite_ratio',
            'use_antithetic_sampling', 'use_fitness_shaping'
        ]
        
        for param in evosax_params:
            old_value = getattr(old_config, param, None)
            new_value = getattr(new_config, param, None)
            if old_value != new_value:
                print(f"  Parameter '{param}' changed: {old_value} → {new_value}")
                return True
        
        return False

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

            elif algorithm_name == "xNES":
                from evosax.algorithms.distribution_based import xNES
                return xNES
            elif algorithm_name == "SNES":
                from evosax.algorithms.distribution_based import SNES
                return SNES

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