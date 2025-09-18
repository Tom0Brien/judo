# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import numpy as np
import optuna
import optuna.visualization as vis
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
        
        # store plan time stats for all trials (regardless of objective)
        self.trial_plan_time_stats = []  # List of (mean, std) for each completed trial

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
        print(f"Results and visualizations will be saved to: {self.results_dir}")

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
        self.trial_plan_time_stats = []

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
        params["num_rollouts"] = 64
        params["num_nodes"] = 6
        params["use_noise_ramp"] = False

        # optimizer-specific parameters
        if self.target_optimizer == "mppi":
            params["sigma"] = trial.suggest_float("sigma", 0.001, 1.0, log=True)
            params["temperature"] = trial.suggest_float("temperature", 0.001, 2.0, log=True)

        elif self.target_optimizer == "cem":
            # Sample sigma_min first, then sigma_max to ensure sigma_min <= sigma_max
            params["sigma_min"] = trial.suggest_float("sigma_min", 0.01, 2.0, log=True)
            params["sigma_max"] = trial.suggest_float("sigma_max", params["sigma_min"], 2.0, log=True)
            params["num_elites"] = trial.suggest_int("num_elites", 1, 16)

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
        
        # store plan time stats for this trial (regardless of objective)
        self.trial_plan_time_stats.append((mean_plan_time, std_plan_time))

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

        # Find individual best solutions for each objective
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

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

            # Find individual best solutions
            if completed_trials:
                # Best reward (highest reward value)
                best_reward_trial = max(completed_trials, key=lambda t: t.values[1] if t.values else -float("inf"))
                if best_reward_trial.values:
                    results["best_reward_solution"] = {
                        "plan_time": best_reward_trial.values[0],
                        "reward": best_reward_trial.values[1],
                        "parameters": best_reward_trial.params,
                    }

                # Best plan time (lowest plan time value)
                best_plan_time_trial = min(completed_trials, key=lambda t: t.values[0] if t.values else float("inf"))
                if best_plan_time_trial.values:
                    results["best_plan_time_solution"] = {
                        "plan_time": best_plan_time_trial.values[0],
                        "reward": best_plan_time_trial.values[1],
                        "parameters": best_plan_time_trial.params,
                    }
        else:
            # Single objective results
            best_trial = self.study.best_trial
            results["best_value"] = best_trial.value
            results["best_parameters"] = best_trial.params

            # For single objectives, still track the other metric if available
            if completed_trials and self.objective == "plan_time":
                # We're optimizing plan time, but also show best reward achieved
                # Note: For plan_time objective, trial.value is plan_time
                best_plan_time_trial = best_trial  # This is already the best plan time
                results["best_plan_time_solution"] = {"plan_time": best_trial.value, "parameters": best_trial.params}

                # Try to find trial with best reward (this would require additional tracking)
                # For now, we'll note this limitation in a comment
                results["note"] = "Individual reward tracking not available for single-objective plan_time optimization"

            elif completed_trials and self.objective == "reward":
                # We're optimizing reward, but also show best plan time achieved
                # Note: For reward objective, trial.value is -reward (since optuna minimizes)
                best_reward_trial = best_trial  # This is already the best reward
                results["best_reward_solution"] = {
                    "reward": -best_trial.value,  # Convert back from negative
                    "parameters": best_trial.params,
                }

                # Find best plan time from stored stats
                if self.trial_plan_time_stats:
                    best_plan_time_idx = np.argmin([stats[0] for stats in self.trial_plan_time_stats])
                    best_plan_time_mean, best_plan_time_std = self.trial_plan_time_stats[best_plan_time_idx]
                    best_plan_time_trial = completed_trials[best_plan_time_idx]
                    
                    results["best_plan_time_solution"] = {
                        "plan_time": best_plan_time_mean,
                        "plan_time_std": best_plan_time_std,
                        "parameters": best_plan_time_trial.params,
                    }
                    
                    # Also store overall plan time statistics
                    all_plan_times = [stats[0] for stats in self.trial_plan_time_stats]
                    results["plan_time_stats"] = {
                        "mean": float(np.mean(all_plan_times)),
                        "std": float(np.std(all_plan_times)),
                        "min": float(np.min(all_plan_times)),
                        "max": float(np.max(all_plan_times)),
                    }

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

        # Print results summary
        print(f"Results saved for {self.target_task} + {self.target_optimizer}")
        if self.objective == "multiobjective":
            print(f"  Pareto front size: {len(pareto_trials)}")
            if "best_reward_solution" in results:
                best_reward = results["best_reward_solution"]
                print(f"  Best reward: {best_reward['reward']:.4f} (plan_time: {best_reward['plan_time']:.4f}s)")
            if "best_plan_time_solution" in results:
                best_plan_time = results["best_plan_time_solution"]
                print(f"  Best plan time: {best_plan_time['plan_time']:.4f}s (reward: {best_plan_time['reward']:.4f})")
        else:
            print(f"  Best value: {best_trial.value:.4f}")
            
            # Show plan time information for reward optimization
            if self.objective == "reward" and "best_plan_time_solution" in results:
                best_plan_time = results["best_plan_time_solution"]
                print(f"  Best plan time: {best_plan_time['plan_time']:.4f}s ± {best_plan_time['plan_time_std']:.4f}s")
                
                if "plan_time_stats" in results:
                    stats = results["plan_time_stats"]
                    print(f"  Plan time stats: mean={stats['mean']:.4f}s, std={stats['std']:.4f}s, min={stats['min']:.4f}s, max={stats['max']:.4f}s")

        # Generate and save visualizations
        self.save_visualizations()
        
        # Extract and save trial data for custom analysis
        self.save_trial_data()
        
        # Create custom visualizations
        self.create_custom_plots()

    def save_visualizations(self) -> None:
        """Generate and save Optuna visualizations for the current study."""
        if not self.study or len(self.study.trials) == 0:
            print("No trials available for visualization")
            return

        pair_key = f"{self.target_task}_{self.target_optimizer}"
        viz_dir = self.results_dir / f"{pair_key}_visualizations"
        viz_dir.mkdir(exist_ok=True)

        try:
            # 1. Optimization History
            if self.objective == "multiobjective":
                # For multi-objective, show both objectives
                fig = vis.plot_optimization_history(self.study, target=lambda t: t.values[0] if t.values else None, target_name="plan_time")
                fig.write_html(str(viz_dir / "optimization_history_plan_time.html"))
                
                fig = vis.plot_optimization_history(self.study, target=lambda t: t.values[1] if t.values else None, target_name="reward")
                fig.write_html(str(viz_dir / "optimization_history_reward.html"))
            else:
                fig = vis.plot_optimization_history(self.study)
                fig.write_html(str(viz_dir / "optimization_history.html"))

            # 2. Parameter Importance
            try:
                fig = vis.plot_param_importances(self.study)
                fig.write_html(str(viz_dir / "param_importances.html"))
            except Exception as e:
                print(f"Could not generate parameter importance plot: {e}")

            # 3. Parallel Coordinate Plot
            try:
                fig = vis.plot_parallel_coordinate(self.study)
                fig.write_html(str(viz_dir / "parallel_coordinate.html"))
            except Exception as e:
                print(f"Could not generate parallel coordinate plot: {e}")

            # 4. Slice Plot
            try:
                fig = vis.plot_slice(self.study)
                fig.write_html(str(viz_dir / "slice_plot.html"))
            except Exception as e:
                print(f"Could not generate slice plot: {e}")

            # 5. Contour Plot (for single objective only)
            if self.objective != "multiobjective":
                try:
                    fig = vis.plot_contour(self.study)
                    fig.write_html(str(viz_dir / "contour_plot.html"))
                except Exception as e:
                    print(f"Could not generate contour plot: {e}")

            # 6. Pareto Front (for multi-objective only)
            if self.objective == "multiobjective":
                try:
                    fig = vis.plot_pareto_front(self.study)
                    fig.write_html(str(viz_dir / "pareto_front.html"))
                except Exception as e:
                    print(f"Could not generate Pareto front plot: {e}")

            # 7. Trial Values vs Parameters
            try:
                fig = vis.plot_timeline(self.study)
                fig.write_html(str(viz_dir / "timeline.html"))
            except Exception as e:
                print(f"Could not generate timeline plot: {e}")

            print(f"  Visualizations saved to: {viz_dir}")

        except Exception as e:
            print(f"Error generating visualizations: {e}")

    def save_trial_data(self) -> None:
        """Extract and save trial data for custom analysis and visualization."""
        if not self.study or len(self.study.trials) == 0:
            print("No trials available for data extraction")
            return

        pair_key = f"{self.target_task}_{self.target_optimizer}"
        data_dir = self.results_dir / f"{pair_key}_data"
        data_dir.mkdir(exist_ok=True)

        try:
            # Extract trial data
            trials_data = []
            for trial in self.study.trials:
                trial_data = {
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "values": trial.values,  # For multi-objective
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                    "system_attrs": trial.system_attrs,
                    "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                    "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                }
                trials_data.append(trial_data)

            # Save as JSON
            import json
            with open(data_dir / "trials_data.json", "w") as f:
                json.dump(trials_data, f, indent=2)

            # Save as CSV for easy analysis
            import pandas as pd
            
            # Flatten the data for CSV
            csv_data = []
            for trial_data in trials_data:
                row = {
                    "trial_number": trial_data["trial_number"],
                    "state": trial_data["state"],
                    "value": trial_data["value"],
                    "datetime_start": trial_data["datetime_start"],
                    "datetime_complete": trial_data["datetime_complete"],
                }
                
                # Add parameters
                for param, value in trial_data["params"].items():
                    row[f"param_{param}"] = value
                
                # Add objective values (for multi-objective)
                if trial_data["values"]:
                    if len(trial_data["values"]) == 2:  # Multi-objective
                        row["objective_plan_time"] = trial_data["values"][0]
                        row["objective_reward"] = trial_data["values"][1]
                    else:  # Single objective
                        row["objective_value"] = trial_data["values"][0]
                
                csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            df.to_csv(data_dir / "trials_data.csv", index=False)

            # Save parameter importance data
            try:
                importance = optuna.importance.get_param_importances(self.study)
                with open(data_dir / "param_importance.json", "w") as f:
                    json.dump(importance, f, indent=2)
            except Exception as e:
                print(f"Could not extract parameter importance: {e}")

            # Save study metadata
            study_metadata = {
                "study_name": self.study.study_name,
                "direction": str(self.study.direction),
                "directions": [str(d) for d in self.study.directions] if hasattr(self.study, 'directions') else None,
                "n_trials": len(self.study.trials),
                "best_trial_number": self.study.best_trial.number if self.study.best_trial else None,
                "best_value": self.study.best_value,
                "best_params": self.study.best_params,
                "objective": self.objective,
                "target_task": self.target_task,
                "target_optimizer": self.target_optimizer,
            }
            
            with open(data_dir / "study_metadata.json", "w") as f:
                json.dump(study_metadata, f, indent=2)

            print(f"  Trial data saved to: {data_dir}")
            print(f"    - trials_data.json: Complete trial information")
            print(f"    - trials_data.csv: Flattened data for analysis")
            print(f"    - param_importance.json: Parameter importance scores")
            print(f"    - study_metadata.json: Study configuration and best results")

        except Exception as e:
            print(f"Error saving trial data: {e}")

    def create_custom_plots(self) -> None:
        """Create custom visualizations using the extracted data."""
        if not self.study or len(self.study.trials) == 0:
            return

        pair_key = f"{self.target_task}_{self.target_optimizer}"
        data_dir = self.results_dir / f"{pair_key}_data"
        viz_dir = self.results_dir / f"{pair_key}_visualizations"
        
        if not data_dir.exists():
            print("No trial data available for custom plots")
            return

        try:
            import pandas as pd
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Load the data
            df = pd.read_csv(data_dir / "trials_data.csv")
            
            # 1. Custom optimization history with plan time stats
            if self.objective == "reward" and "objective_value" in df.columns:
                fig = go.Figure()
                
                # Add optimization history
                fig.add_trace(go.Scatter(
                    x=df["trial_number"],
                    y=-df["objective_value"],  # Convert back from negative
                    mode='lines+markers',
                    name='Reward',
                    line=dict(color='blue'),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"Custom Optimization History - {self.target_task} + {self.target_optimizer}",
                    xaxis_title="Trial Number",
                    yaxis_title="Cumulative Reward",
                    template="plotly_white",
                    width=800,
                    height=500
                )
                
                fig.write_html(str(viz_dir / "custom_optimization_history.html"))
            
            # 2. Parameter distribution plots
            param_cols = [col for col in df.columns if col.startswith("param_")]
            if param_cols:
                n_params = len(param_cols)
                cols = min(3, n_params)
                rows = (n_params + cols - 1) // cols
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=[col.replace("param_", "") for col in param_cols]
                )
                
                for i, param_col in enumerate(param_cols):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    fig.add_trace(
                        go.Histogram(
                            x=df[param_col],
                            name=param_col.replace("param_", ""),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    title=f"Parameter Distributions - {self.target_task} + {self.target_optimizer}",
                    template="plotly_white",
                    width=1200,
                    height=400 * rows
                )
                
                fig.write_html(str(viz_dir / "custom_param_distributions.html"))
            
            # 3. Parameter vs Objective scatter plots
            if param_cols and "objective_value" in df.columns:
                n_params = len(param_cols)
                cols = min(2, n_params)
                rows = (n_params + cols - 1) // cols
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=[f"{col.replace('param_', '')} vs Reward" for col in param_cols]
                )
                
                for i, param_col in enumerate(param_cols):
                    row = i // cols + 1
                    col = i % cols + 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df[param_col],
                            y=-df["objective_value"],  # Convert back from negative
                            mode='markers',
                            name=param_col.replace("param_", ""),
                            showlegend=False,
                            marker=dict(
                                size=8,
                                color=df["trial_number"],
                                colorscale="Viridis",
                                showscale=(i == 0)
                            )
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    title=f"Parameter vs Objective - {self.target_task} + {self.target_optimizer}",
                    template="plotly_white",
                    width=1200,
                    height=400 * rows
                )
                
                fig.write_html(str(viz_dir / "custom_param_vs_objective.html"))
            
            print(f"  Custom plots saved to: {viz_dir}")
            
        except ImportError:
            print("  Custom plots require pandas and plotly. Install with: pip install pandas plotly")
        except Exception as e:
            print(f"Error creating custom plots: {e}")

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

                # show individual best solutions
                best_reward = results.get("best_reward_solution")
                if best_reward:
                    self.console.print(
                        f"  Best reward: {best_reward['reward']:.4f} (plan_time: {best_reward['plan_time']:.4f}s)"
                    )

                best_plan_time = results.get("best_plan_time_solution")
                if best_plan_time:
                    self.console.print(
                        f"  Best plan time: {best_plan_time['plan_time']:.4f}s (reward: {best_plan_time['reward']:.4f})"
                    )

                # show top pareto solutions
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

                # show individual best solutions for single objectives too
                best_reward = results.get("best_reward_solution")
                if best_reward:
                    self.console.print(f"  Best reward achieved: {best_reward['reward']:.4f}")

                best_plan_time = results.get("best_plan_time_solution")
                if best_plan_time:
                    if self.objective == "reward" and "plan_time_std" in best_plan_time:
                        self.console.print(f"  Best plan time achieved: {best_plan_time['plan_time']:.4f}s ± {best_plan_time['plan_time_std']:.4f}s")
                    else:
                        self.console.print(f"  Best plan time achieved: {best_plan_time['plan_time']:.4f}s")

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

        # Show absolute best across all pairs for individual objectives
        self.console.print("\n[bold]Absolute Best Across All Pairs:[/bold]")
        self.console.print("-" * 50)

        # Find absolute best reward across all pairs
        best_reward_global = None
        best_reward_pair = None
        best_plan_time_global = None
        best_plan_time_pair = None

        for pair_key, results in self.all_results.items():
            task, optimizer = pair_key.split("_", 1)

            # Check best reward solution
            best_reward_solution = results.get("best_reward_solution")
            if best_reward_solution and (
                best_reward_global is None or best_reward_solution["reward"] > best_reward_global
            ):
                best_reward_global = best_reward_solution["reward"]
                best_reward_pair = f"{task} + {optimizer}"

            # Check best plan time solution
            best_plan_time_solution = results.get("best_plan_time_solution")
            if best_plan_time_solution and (
                best_plan_time_global is None or best_plan_time_solution["plan_time"] < best_plan_time_global
            ):
                best_plan_time_global = best_plan_time_solution["plan_time"]
                best_plan_time_pair = f"{task} + {optimizer}"

        if best_reward_global is not None:
            self.console.print(f"Best reward: {best_reward_global:.4f} ({best_reward_pair})")
        if best_plan_time_global is not None:
            self.console.print(f"Best plan time: {best_plan_time_global:.4f}s ({best_plan_time_pair})")

        self.console.print(f"\n[bold]Results saved to: {self.results_dir}[/bold]")
        self.console.print(f"  Combined results: {combined_file}")
        self.console.print(f"  Individual results: {self.results_dir}/*_results.json")
        
        # Show visualization and data information
        viz_dirs = list(self.results_dir.glob("*_visualizations"))
        data_dirs = list(self.results_dir.glob("*_data"))
        
        if viz_dirs:
            self.console.print(f"  Visualizations: {len(viz_dirs)} directories with HTML plots")
            for viz_dir in viz_dirs:
                pair_name = viz_dir.name.replace("_visualizations", "")
                self.console.print(f"    {pair_name}: {viz_dir}")
        else:
            self.console.print("  Visualizations: None generated")
            
        if data_dirs:
            self.console.print(f"  Trial Data: {len(data_dirs)} directories with raw data")
            for data_dir in data_dirs:
                pair_name = data_dir.name.replace("_data", "")
                self.console.print(f"    {pair_name}: {data_dir}")
        else:
            self.console.print("  Trial Data: None extracted")

        print("\nMulti-pair hyperparameter tuning complete! You may terminate the stack.")
