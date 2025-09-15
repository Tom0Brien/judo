# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np
import optuna
import pyarrow as pa
from dora_utils.dataclasses import from_arrow, to_arrow
from dora_utils.node import DoraNode, on_event
from rich.console import Console

from judo.app.structs import MujocoState
from judo.config import set_config_overrides
from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks


class TunerNode(DoraNode):
    """A node that tunes task/optimizer parameters using Optuna.

    Supports single-objective optimization (plan_time, reward) or multi-objective
    optimization (multiobjective) to find Pareto-optimal solutions.
    """

    def __init__(
        self,
        node_id: str = "tuner",
        max_workers: int | None = None,
        n_trials: int = 50,
        n_samples_per_trial: int = 20,
        target_tasks: Union[str, List[str]] = "cylinder_push",
        target_optimizers: Union[str, List[str]] = "mppi",
        objective: Literal["plan_time", "reward", "multiobjective"] = "plan_time",
        trial_duration: float = 10.0,
        study_name: str | None = None,
        storage: str | None = None,
        results_dir: str = "tune_results",
    ) -> None:
        """Initialize the optimizer node."""
        super().__init__(node_id=node_id, max_workers=max_workers)

        # storing all available tasks and optimizers
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()

        # convert single items to lists
        if isinstance(target_tasks, str):
            target_tasks = [target_tasks]
        if isinstance(target_optimizers, str):
            target_optimizers = [target_optimizers]

        # validate target tasks and optimizers
        for task in target_tasks:
            if task not in self.available_tasks:
                raise ValueError(f"Task '{task}' not found in registered tasks")
        for optimizer in target_optimizers:
            if optimizer not in self.available_optimizers:
                raise ValueError(f"Optimizer '{optimizer}' not found in registered optimizers")

        # create task-optimizer pairs queue
        self.task_optimizer_pairs = deque(
            [(task, optimizer) for task in target_tasks for optimizer in target_optimizers]
        )
        self.total_pairs = len(self.task_optimizer_pairs)
        self.current_pair_index = 0

        # current task and optimizer (will be set when processing pairs)
        self.target_task = None
        self.target_optimizer = None

        # tuning parameters
        self.n_trials = n_trials
        self.n_samples_per_trial = n_samples_per_trial
        self.objective = objective
        self.trial_duration = trial_duration
        self.base_study_name = study_name
        self.storage = storage

        # results storage
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.all_results = {}

        # trial tracking
        self.current_trial = None
        self.current_plan_times = []
        self.samples_collected = 0

        # trajectory tracking for reward calculation
        self.trial_start_time = None
        self.trajectory_states = []
        self.trajectory_sensors = []
        self.trajectory_controls = []
        self.trajectory_times = []
        self.current_rewards = []

        # reset state shape tracking for new pair
        self.expected_state_shape = None
        self.shape_changes_logged = False

        # optuna study (will be created for each task-optimizer pair)
        self.study = None

        # console for printing
        self.console = Console()

        print(f"Starting hyperparameter tuning for {len(self.task_optimizer_pairs)} task-optimizer pairs!")
        print(f"Pairs to process: {[(task, opt) for task, opt in self.task_optimizer_pairs]}")
        print(f"Target: {n_trials} trials with {n_samples_per_trial} samples each per pair")
        if objective == "multiobjective":
            print("Mode: Multi-objective optimization (plan_time vs reward)")
        else:
            print(f"Mode: Single-objective optimization ({objective})")

        # start processing the first pair
        self.start_next_pair()

    def start_next_pair(self) -> None:
        """Start processing the next task-optimizer pair."""
        if not self.task_optimizer_pairs:
            # All pairs completed
            self.print_final_results()
            sys.exit(0)

        # get next pair
        self.target_task, self.target_optimizer = self.task_optimizer_pairs.popleft()
        self.current_pair_index += 1

        print(f"\n{'=' * 60}")
        print(
            f"Processing pair {self.current_pair_index}/{self.total_pairs}: {self.target_task} + {self.target_optimizer}"
        )
        print(f"{'=' * 60}")

        # create new study for this pair
        study_name = self.base_study_name or f"optimize_{self.target_task}_{self.target_optimizer}"
        if self.objective == "multiobjective":
            study_name += "_multiobjective"
            self.study = optuna.create_study(
                directions=["minimize", "maximize"],
                study_name=study_name,
                storage=self.storage,
                load_if_exists=True,
            )
        else:
            self.study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=self.storage,
                load_if_exists=True,
            )

        # reset trial tracking for new pair
        self.current_trial = None
        self.current_plan_times = []
        self.samples_collected = 0

        # reset trajectory tracking
        self.trial_start_time = None
        self.trajectory_states = []
        self.trajectory_sensors = []
        self.trajectory_controls = []
        self.trajectory_times = []
        self.current_rewards = []

        # reset shape change logging for this trial (but keep expected_state_shape)
        self.shape_changes_logged = False

        # send new task and optimizer to pipeline
        self.node.send_output("task", pa.array([self.target_task]))
        self.node.send_output("optimizer", pa.array([self.target_optimizer]))

        # send task reset to ensure clean state for new pair
        self.node.send_output("task_reset", pa.array([True]))

        print(f"Starting trials for {self.target_task} + {self.target_optimizer}")

        # start first trial for this pair
        self.start_next_trial()

    def start_next_trial(self) -> None:
        """Start the next tuning trial for the current task-optimizer pair."""
        if len(self.study.trials) >= self.n_trials:
            # Current pair completed, save results and move to next pair
            self.save_pair_results()
            self.start_next_pair()
            return

        # create new trial
        self.current_trial = self.study.ask()
        self.current_plan_times = []
        self.samples_collected = 0

        # reset trajectory tracking
        self.trial_start_time = None
        self.trajectory_states = []
        self.trajectory_sensors = []
        self.trajectory_controls = []
        self.trajectory_times = []
        self.current_rewards = []

        # suggest parameters based on optimizer type
        suggested_params = self.suggest_parameters(self.current_trial)

        print(f"Trial {len(self.study.trials)}: Testing parameters {suggested_params}")

        # send the optimizer config with suggested parameters
        self.send_optimizer_config(suggested_params)

    def suggest_parameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest parameters for the current trial based on optimizer type."""
        params = {}

        # base optimizer parameters
        params["num_rollouts"] = trial.suggest_int("num_rollouts", 8, 64, step=8)
        params["num_nodes"] = trial.suggest_int("num_nodes", 3, 12)
        params["use_noise_ramp"] = trial.suggest_categorical("use_noise_ramp", [True, False])
        if params["use_noise_ramp"]:
            params["noise_ramp"] = trial.suggest_float("noise_ramp", 0.5, 5.0)

        # optimizer-specific parameters
        if self.target_optimizer == "mppi":
            params["sigma"] = trial.suggest_float("sigma", 0.001, 1.0, log=True)
            params["temperature"] = trial.suggest_float("temperature", 0.001, 2.0, log=True)

        elif self.target_optimizer == "cem":
            params["sigma_min"] = trial.suggest_float("sigma_min", 0.01, 0.5, log=True)
            params["sigma_max"] = trial.suggest_float("sigma_max", 0.5, 2.0, log=True)
            params["num_elites"] = trial.suggest_int("num_elites", 1, min(8, params["num_rollouts"] // 2))

        elif self.target_optimizer == "ps":
            params["sigma"] = trial.suggest_float("sigma", 0.001, 1.0, log=True)

        return params

    def send_optimizer_config(self, params: Dict[str, Any]) -> None:
        """Send the optimizer configuration with suggested parameters."""
        # get the optimizer config class
        _, optimizer_config_cls = self.available_optimizers[self.target_optimizer]

        # create a unique override key for this trial
        override_key = f"optuna_trial_{len(self.study.trials)}"

        # register the parameter overrides
        set_config_overrides(override_key, optimizer_config_cls, params)

        # create optimizer config instance and apply overrides
        optimizer_config = optimizer_config_cls()
        optimizer_config.set_override(override_key)

        # send the updated optimizer configuration
        arr, metadata = to_arrow(optimizer_config)
        self.node.send_output("optimizer_config", arr, metadata)

        # also trigger a task reset to ensure the new parameters are applied
        self.node.send_output("task_reset", pa.array([True]))

    @on_event("INPUT", "states")
    def on_states(self, event: dict) -> None:
        """Handle simulation state events for trajectory collection."""
        if self.current_trial is None:
            return

        # extract state message
        state_msg = from_arrow(event["value"], event["metadata"], MujocoState)

        # initialize trial start time
        if self.trial_start_time is None:
            self.trial_start_time = state_msg.time

        # collect trajectory data with shape validation
        try:
            state_vector = np.concatenate([state_msg.qpos, state_msg.qvel])
            current_shape = len(state_vector)

            # handle shape initialization and changes
            if self.expected_state_shape is None:
                # First state in this task-optimizer pair
                self.expected_state_shape = current_shape
                print(f"Initialized state shape for {self.target_task} + {self.target_optimizer}: {current_shape}")
            elif current_shape != self.expected_state_shape:
                # Shape mismatch detected
                if not self.shape_changes_logged:
                    print(f"State shape change detected in {self.target_task} + {self.target_optimizer}:")
                    print(f"  Expected: {self.expected_state_shape}, Got: {current_shape}")
                    print(f"  qpos: {len(state_msg.qpos)}, qvel: {len(state_msg.qvel)}")
                    self.shape_changes_logged = True

                # If we're early in the trial (first few states), reset and adapt
                if len(self.trajectory_states) < 5:
                    print(f"  Early in trial - adapting to new shape: {current_shape}")
                    self.expected_state_shape = current_shape
                    self.trajectory_times = [state_msg.time]
                    self.trajectory_states = [state_vector]
                    self.trajectory_sensors = [state_msg.sensordata.copy()]
                    self.trajectory_controls = [state_msg.ctrl.copy()]
                    self.trial_start_time = state_msg.time
                    return
                else:
                    # Late in trial - skip this state to maintain consistency
                    return

            # Normal trajectory collection
            self.trajectory_times.append(state_msg.time)
            self.trajectory_states.append(state_vector)
            self.trajectory_sensors.append(state_msg.sensordata.copy())
            self.trajectory_controls.append(state_msg.ctrl.copy())

        except Exception as e:
            print(f"Error collecting trajectory data: {e}")
            print(f"qpos shape: {state_msg.qpos.shape}, qvel shape: {state_msg.qvel.shape}")
            return

    @on_event("INPUT", "plan_time")
    def on_plan_time(self, event: dict) -> None:
        """Handle plan time events."""
        if self.current_trial is None:
            return

        # extract plan time
        plan_time = event["value"].to_numpy(zero_copy_only=False)[0]
        self.current_plan_times.append(plan_time)
        self.samples_collected += 1

        # check if we have enough samples for this trial
        if self.samples_collected >= self.n_samples_per_trial:
            self.complete_trial()

    def complete_trial(self) -> None:
        """Complete the current trial and report the objective value."""
        if not self.current_plan_times:
            print("Warning: No plan times collected for trial. Skipping trial completion.")
            self.start_next_trial()
            return

        # compute plan time metrics
        mean_plan_time = np.mean(self.current_plan_times)
        std_plan_time = np.std(self.current_plan_times)

        # compute reward metrics if trajectory data is available
        cumulative_reward = -1e12  # default fallback value
        reward_computed_successfully = False

        if self.trajectory_states and self.objective in ["reward", "multiobjective"]:
            try:
                computed_reward = self.compute_cumulative_reward()
                # Check if reward computation was successful (not the default 0.0 from error handling)
                if computed_reward != 0.0 or len(self.trajectory_states) == 0:
                    cumulative_reward = computed_reward
                    reward_computed_successfully = True
                else:
                    print("Warning: Reward computation returned 0.0, using fallback strategy")
                    # For failed reward computation, use a penalty based on plan time
                    cumulative_reward = -mean_plan_time * 10  # penalty for failed reward computation
            except Exception as e:
                print(f"Error in complete_trial reward computation: {e}")
                cumulative_reward = -mean_plan_time * 10  # penalty for failed reward computation

        # report to optuna based on objective type
        try:
            if self.objective == "plan_time":
                objective_value = mean_plan_time
                self.study.tell(self.current_trial, objective_value)
            elif self.objective == "reward":
                if not reward_computed_successfully and cumulative_reward == -1e12:
                    print("Warning: Skipping trial due to failed reward computation")
                    self.start_next_trial()
                    return
                objective_value = -cumulative_reward  # negative because optuna minimizes
                self.study.tell(self.current_trial, objective_value)
            elif self.objective == "multiobjective":
                if not reward_computed_successfully and cumulative_reward == -1e12:
                    print("Warning: Using plan time only for multiobjective due to failed reward computation")
                    cumulative_reward = -mean_plan_time * 10
                # Multi-objective: minimize plan_time, maximize reward
                objectives = [mean_plan_time, cumulative_reward]
                self.study.tell(self.current_trial, objectives)
            else:
                raise ValueError(f"Unknown objective: {self.objective}")
        except Exception as e:
            print(f"Error reporting to Optuna: {e}")
            self.start_next_trial()
            return

        # print results
        print(f"Trial {len(self.study.trials)} completed:")
        print(f"  Plan time: {mean_plan_time:.4f}s ± {std_plan_time:.4f}s")
        if self.objective in ["reward", "multiobjective"]:
            status = "✓" if reward_computed_successfully else "✗ (fallback)"
            print(f"  Cumulative reward: {cumulative_reward:.4f} {status}")
        if self.objective == "multiobjective":
            print(f"  Objectives: [plan_time={mean_plan_time:.4f}s, reward={cumulative_reward:.4f}]")
        else:
            # For single objectives, we have objective_value defined
            if self.objective == "plan_time":
                objective_value = mean_plan_time
            elif self.objective == "reward":
                objective_value = -cumulative_reward
            print(f"  Objective value: {objective_value:.4f}")

        # start next trial
        self.start_next_trial()

    def save_pair_results(self) -> None:
        """Save results for the current task-optimizer pair."""
        pair_key = f"{self.target_task}_{self.target_optimizer}"

        # extract best results
        results = {
            "task": self.target_task,
            "optimizer": self.target_optimizer,
            "objective": self.objective,
            "n_trials": len(self.study.trials),
            "completed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        }

        if self.objective == "multiobjective":
            # Multi-objective results
            pareto_trials = self.study.best_trials
            results["pareto_front_size"] = len(pareto_trials)
            results["pareto_solutions"] = []
            for trial in pareto_trials[:10]:  # Store top 10 Pareto solutions
                if trial.values is not None:
                    plan_time, reward = trial.values
                    results["pareto_solutions"].append(
                        {"plan_time": plan_time, "reward": reward, "parameters": trial.params}
                    )
        else:
            # Single objective results
            best_trial = self.study.best_trial
            results["best_value"] = best_trial.value
            results["best_parameters"] = best_trial.params

            # parameter importance
            try:
                importance = optuna.importance.get_param_importances(self.study)
                results["parameter_importance"] = importance
            except Exception:
                results["parameter_importance"] = {}

        # save to all_results
        self.all_results[pair_key] = results

        # save individual pair results to file
        pair_file = self.results_dir / f"{pair_key}_results.json"
        with open(pair_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved for {self.target_task} + {self.target_optimizer}")
        if self.objective == "multiobjective":
            print(f"  Pareto front size: {len(pareto_trials)}")
        else:
            print(f"  Best value: {best_trial.value:.4f}")

    def compute_cumulative_reward(self) -> float:
        """Compute the cumulative reward for the current trajectory."""
        if not self.trajectory_states:
            return 0.0

        # get task instance for reward computation
        task_entry = self.available_tasks[self.target_task]
        task_cls, task_config_cls = task_entry
        task = task_cls()
        task_config = task_config_cls()

        try:
            # convert trajectory data to numpy arrays with error handling
            states = np.array(self.trajectory_states)  # shape: (T, nq+nv)
            sensors = np.array(self.trajectory_sensors)  # shape: (T, nsensordata)
            controls = np.array(self.trajectory_controls)  # shape: (T, nu)

            # validate array shapes
            if len(states.shape) != 2 or len(sensors.shape) != 2 or len(controls.shape) != 2:
                print(
                    f"Warning: Invalid trajectory array shapes. States: {states.shape}, Sensors: {sensors.shape}, Controls: {controls.shape}"
                )
                return 0.0

            # reshape for task reward function: (1, T, dim) for single rollout
            states_batch = states[None, :, :]  # (1, T, nq+nv)
            sensors_batch = sensors[None, :, :]  # (1, T, nsensordata)
            controls_batch = controls[None, :, :]  # (1, T, nu)

            # compute rewards using task's reward function
            rewards = task.reward(states_batch, sensors_batch, controls_batch, task_config)

            # return the cumulative reward (single rollout)
            return float(rewards[0]) if len(rewards) > 0 else 0.0

        except ValueError as e:
            print(f"Error computing trajectory reward: {e}")
            print(
                f"Trajectory lengths - States: {len(self.trajectory_states)}, Sensors: {len(self.trajectory_sensors)}, Controls: {len(self.trajectory_controls)}"
            )
            if self.trajectory_states:
                state_shapes = [len(s) for s in self.trajectory_states[:5]]  # Show first 5 shapes
                print(f"State vector shapes (first 5): {state_shapes}")
            return 0.0
        except Exception as e:
            print(f"Unexpected error computing reward: {e}")
            return 0.0

    def print_final_results(self) -> None:
        """Print the final tuning results for all task-optimizer pairs."""
        self.console.print("\n[bold green]Multi-Pair Hyperparameter Tuning Complete![/bold green]")

        if not self.all_results:
            print("No results available.")
            return

        # save combined results
        combined_file = self.results_dir / "combined_results.json"
        with open(combined_file, "w") as f:
            json.dump(self.all_results, f, indent=2)

        # print summary for each pair
        self.console.print(f"\n[bold]Results Summary for {len(self.all_results)} Task-Optimizer Pairs:[/bold]")
        self.console.print("=" * 80)

        for _, results in self.all_results.items():
            task = results["task"]
            optimizer = results["optimizer"]

            self.console.print(f"\n[bold cyan]{task} + {optimizer}[/bold cyan]")
            self.console.print(f"  Completed trials: {results['completed_trials']}/{results['n_trials']}")

            if self.objective == "multiobjective":
                pareto_size = results.get("pareto_front_size", 0)
                self.console.print(f"  Pareto front size: {pareto_size}")

                # show best solutions
                pareto_solutions = results.get("pareto_solutions", [])
                if pareto_solutions:
                    self.console.print("  Top Pareto solutions:")
                    for i, solution in enumerate(pareto_solutions[:3]):
                        plan_time = solution["plan_time"]
                        reward = solution["reward"]
                        self.console.print(f"    {i + 1}. plan_time={plan_time:.4f}s, reward={reward:.4f}")
            else:
                best_value = results.get("best_value", "N/A")
                if best_value != "N/A":
                    if self.objective == "plan_time":
                        self.console.print(f"  Best plan time: {best_value:.4f}s")
                    elif self.objective == "reward":
                        self.console.print(f"  Best cumulative reward: {-best_value:.4f}")

                # show top parameters
                best_params = results.get("best_parameters", {})
                if best_params:
                    self.console.print("  Best parameters:")
                    for param, value in list(best_params.items())[:5]:  # Show top 5 params
                        self.console.print(f"    {param}: {value}")

        # print comparison summary
        self.console.print(f"\n[bold]Cross-Pair Comparison ({self.objective}):[/bold]")
        self.console.print("-" * 60)

        if self.objective == "multiobjective":
            # For multi-objective, show Pareto front sizes
            pareto_summary = []
            for pair_key, results in self.all_results.items():
                pareto_size = results.get("pareto_front_size", 0)
                pareto_summary.append((pair_key, pareto_size))

            pareto_summary.sort(key=lambda x: x[1], reverse=True)
            self.console.print("Pairs ranked by Pareto front size:")
            for i, (pair_key, size) in enumerate(pareto_summary):
                task, optimizer = pair_key.split("_", 1)
                self.console.print(f"  {i + 1:2d}. {task} + {optimizer}: {size} solutions")
        else:
            # For single objective, rank pairs by best value
            best_values = []
            for pair_key, results in self.all_results.items():
                best_value = results.get("best_value")
                if best_value is not None:
                    best_values.append((pair_key, best_value))

            # sort based on objective (minimize for plan_time, maximize for reward)
            reverse_sort = self.objective == "reward"
            best_values.sort(key=lambda x: x[1], reverse=reverse_sort)

            self.console.print(f"Pairs ranked by best {self.objective}:")
            for i, (pair_key, value) in enumerate(best_values):
                task, optimizer = pair_key.split("_", 1)
                if self.objective == "plan_time":
                    self.console.print(f"  {i + 1:2d}. {task} + {optimizer}: {value:.4f}s")
                elif self.objective == "reward":
                    self.console.print(f"  {i + 1:2d}. {task} + {optimizer}: {-value:.4f}")

        self.console.print(f"\n[bold]Results saved to: {self.results_dir}[/bold]")
        self.console.print(f"  Combined results: {combined_file}")
        self.console.print(f"  Individual results: {self.results_dir}/*_results.json")

        print("\nMulti-pair hyperparameter tuning complete! You may terminate the stack.")
