# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import sys
from typing import Any, Dict, Literal

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
    """A node that tunes task/optimizer parameters using Optuna to minimize plan times."""

    def __init__(
        self,
        node_id: str = "tuner",
        max_workers: int | None = None,
        n_trials: int = 50,
        n_samples_per_trial: int = 20,
        target_task: str = "cylinder_push",
        target_optimizer: str = "mppi",
        objective: Literal["plan_time", "reward", "combined"] = "plan_time",
        reward_weight: float = 1.0,
        plan_time_weight: float = 1.0,
        trial_duration: float = 10.0,
        study_name: str | None = None,
        storage: str | None = None,
    ) -> None:
        """Initialize the optimizer node."""
        super().__init__(node_id=node_id, max_workers=max_workers)

        # storing all available tasks and optimizers
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()

        # validate target task and optimizer
        if target_task not in self.available_tasks:
            raise ValueError(f"Task '{target_task}' not found in registered tasks")
        if target_optimizer not in self.available_optimizers:
            raise ValueError(f"Optimizer '{target_optimizer}' not found in registered optimizers")

        self.target_task = target_task
        self.target_optimizer = target_optimizer
        self.n_trials = n_trials
        self.n_samples_per_trial = n_samples_per_trial
        self.objective = objective
        self.reward_weight = reward_weight
        self.plan_time_weight = plan_time_weight
        self.trial_duration = trial_duration

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

        # optuna study
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name or f"optimize_{target_task}_{target_optimizer}",
            storage=storage,
            load_if_exists=True,
        )

        # console for printing
        self.console = Console()

        print(f"Starting hyperparameter tuning for {target_task} + {target_optimizer}!")
        print(f"Target: {n_trials} trials with {n_samples_per_trial} samples each")

        # set initial task and optimizer
        self.node.send_output("task", pa.array([self.target_task]))
        self.node.send_output("optimizer", pa.array([self.target_optimizer]))

        self.start_next_trial()

    def start_next_trial(self) -> None:
        """Start the next tuning trial."""
        if len(self.study.trials) >= self.n_trials:
            self.print_results()
            sys.exit(0)

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

        # collect trajectory data
        self.trajectory_times.append(state_msg.time)
        self.trajectory_states.append(np.concatenate([state_msg.qpos, state_msg.qvel]))
        self.trajectory_sensors.append(state_msg.sensordata.copy())
        self.trajectory_controls.append(state_msg.ctrl.copy())

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
            return

        # compute plan time metrics
        mean_plan_time = np.mean(self.current_plan_times)
        std_plan_time = np.std(self.current_plan_times)

        # compute reward metrics if trajectory data is available
        cumulative_reward = -1e12
        if self.trajectory_states and self.objective in ["reward", "combined"]:
            cumulative_reward = self.compute_cumulative_reward()

        # compute objective value based on selected objective
        if self.objective == "plan_time":
            objective_value = mean_plan_time
        elif self.objective == "reward":
            objective_value = -cumulative_reward  # negative because optuna minimizes
        elif self.objective == "combined":
            # normalize both metrics and combine
            objective_value = self.plan_time_weight * mean_plan_time - self.reward_weight * cumulative_reward
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # report to optuna
        self.study.tell(self.current_trial, objective_value)

        # print results
        print(f"Trial {len(self.study.trials)} completed:")
        print(f"  Plan time: {mean_plan_time:.4f}s ± {std_plan_time:.4f}s")
        if self.objective in ["reward", "combined"]:
            print(f"  Cumulative reward: {cumulative_reward:.4f}")
        print(f"  Objective value: {objective_value:.4f}")

        # start next trial
        self.start_next_trial()

    def compute_cumulative_reward(self) -> float:
        """Compute the cumulative reward for the current trajectory."""
        if not self.trajectory_states:
            return 0.0

        # get task instance for reward computation
        task_entry = self.available_tasks[self.target_task]
        task_cls, task_config_cls = task_entry
        task = task_cls()
        task_config = task_config_cls()

        # convert trajectory data to numpy arrays
        states = np.array(self.trajectory_states)  # shape: (T, nq+nv)
        sensors = np.array(self.trajectory_sensors)  # shape: (T, nsensordata)
        controls = np.array(self.trajectory_controls)  # shape: (T, nu)

        # reshape for task reward function: (1, T, dim) for single rollout
        states_batch = states[None, :, :]  # (1, T, nq+nv)
        sensors_batch = sensors[None, :, :]  # (1, T, nsensordata)
        controls_batch = controls[None, :, :]  # (1, T, nu)

        # compute rewards using task's reward function
        rewards = task.reward(states_batch, sensors_batch, controls_batch, task_config)

        # return the cumulative reward (single rollout)
        return float(rewards[0]) if len(rewards) > 0 else 0.0

    def print_results(self) -> None:
        """Print the tuning results."""
        self.console.print("\n[bold green]Hyperparameter Tuning Complete![/bold green]")

        if not self.study.trials:
            print("No trials completed.")
            return

        # print best trial
        best_trial = self.study.best_trial
        self.console.print("[bold]Best Trial:[/bold]")
        if self.objective == "plan_time":
            self.console.print(f"  Best plan time: {best_trial.value:.4f}s")
        elif self.objective == "reward":
            self.console.print(f"  Best cumulative reward: {-best_trial.value:.4f}")
        elif self.objective == "combined":
            self.console.print(f"  Best combined objective: {best_trial.value:.4f}")
        self.console.print("  Parameters:")
        for key, value in best_trial.params.items():
            self.console.print(f"    {key}: {value}")

        # print tuning history
        self.console.print("\n[bold]Tuning History:[/bold]")
        for i, trial in enumerate(self.study.trials):
            status = "✓" if trial.state == optuna.trial.TrialState.COMPLETE else "✗"
            value_str = f"{trial.value:.4f}s" if trial.value is not None else "N/A"
            self.console.print(f"  Trial {i + 1:2d} {status} {value_str}")

        # print parameter importance (if available)
        try:
            importance = optuna.importance.get_param_importances(self.study)
            if importance:
                self.console.print("\n[bold]Parameter Importance:[/bold]")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    self.console.print(f"  {param}: {imp:.3f}")
        except Exception:
            pass  # importance analysis might fail for small studies

        print("\nHyperparameter tuning complete! You may terminate the stack.")
