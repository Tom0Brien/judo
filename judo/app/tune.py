# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import sys
from typing import Any, Dict

import numpy as np
import optuna
import pyarrow as pa
from dora_utils.dataclasses import to_arrow
from dora_utils.node import DoraNode, on_event
from rich.console import Console

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

        # trial tracking
        self.current_trial = None
        self.current_plan_times = []
        self.samples_collected = 0

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

        # compute objective (mean plan time)
        objective_value = np.mean(self.current_plan_times)
        std_objective = np.std(self.current_plan_times)

        # report to optuna
        self.study.tell(self.current_trial, objective_value)

        print(
            f"Trial {len(self.study.trials)} completed: mean plan time = {objective_value:.4f}s ± {std_objective:.4f}s"
        )

        # start next trial
        self.start_next_trial()

    def print_results(self) -> None:
        """Print the tuning results."""
        self.console.print("\n[bold green]Hyperparameter Tuning Complete![/bold green]")

        if not self.study.trials:
            print("No trials completed.")
            return

        # print best trial
        best_trial = self.study.best_trial
        self.console.print("[bold]Best Trial:[/bold]")
        self.console.print(f"  Value (mean plan time): {best_trial.value:.4f}s")
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
