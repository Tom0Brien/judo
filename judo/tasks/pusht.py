# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig

XML_PATH = str(MODEL_PATH / "xml/pusht.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/pusht_sim.xml")


@dataclass
class PushtConfig(TaskConfig):
    """Reward configuration for the pusht task."""

    w_position: float = 1.0  # Weight for position tracking cost
    w_orientation: float = 1.0  # Weight for orientation tracking cost
    w_proximity: float = 0.01  # Weight for pusher-to-block proximity cost


class Pusht(Task[PushtConfig]):
    """Push a T-shaped block to a desired pose.

    This task is adapted from the hydrax pusht task, where a red pusher sphere
    controls the position of a T-shaped block to match a target pose (green T).
    """

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str = SIM_XML_PATH) -> None:
        """Initialize the pusht task."""
        super().__init__(model_path, sim_model_path=sim_model_path)

        # Get sensor IDs for position and orientation tracking
        self.position_sensor_id = self.model.sensor("position").id
        self.orientation_sensor_id = self.model.sensor("orientation").id

        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: PushtConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute the reward for the pusht task.

        The reward encourages the T-shaped block to reach the target position and
        orientation while keeping the pusher close to the block.

        Args:
            states: The rolled out states. Shape=(num_rollouts, T, nq + nv).
            sensors: The rolled out sensors readings. Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: The rolled out controls. Shape=(num_rollouts, T, nu).
            config: The current task config.
            system_metadata: Additional metadata from the system.

        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """
        batch_size = states.shape[0]

        # Extract state components
        # qpos = [block_x, block_y, block_angle, pusher_x, pusher_y]
        # qvel = [block_vx, block_vy, block_vz, pusher_vx, pusher_vy]
        block_pos = states[..., :2]  # [batch, time, 2] - block x,y position
        pusher_pos = states[..., 3:5]  # [batch, time, 2] - pusher x,y position

        # Extract sensor data for position and orientation errors
        # Position sensor: 3D position error of block relative to goal
        position_sensor_start = self.get_sensor_start_index("position")
        position_err = sensors[..., position_sensor_start : position_sensor_start + 3]  # [batch, time, 3]

        # Orientation sensor: quaternion error of block relative to goal
        orientation_sensor_start = self.get_sensor_start_index("orientation")
        orientation_err = sensors[..., orientation_sensor_start : orientation_sensor_start + 4]  # [batch, time, 4]

        # Close to block error: pusher position relative to block (with y-bias)
        pusher_with_bias = pusher_pos + np.array([0.0, 0.1])  # Add y-bias to pusher
        close_to_block_err = block_pos - pusher_with_bias  # [batch, time, 2]

        # Compute cost components (matching hydrax exactly)
        position_cost = config.w_position * np.sum(np.square(position_err), axis=-1).sum(-1)  # [batch]
        orientation_cost = config.w_orientation * np.sum(np.square(orientation_err), axis=-1).sum(-1)  # [batch]
        proximity_cost = config.w_proximity * np.sum(np.square(close_to_block_err), axis=-1).sum(-1)  # [batch]

        # Total cost (positive cost, higher is worse)
        total_cost = position_cost + orientation_cost + proximity_cost

        # Verify shapes
        assert position_cost.shape == (batch_size,)
        assert orientation_cost.shape == (batch_size,)
        assert proximity_cost.shape == (batch_size,)

        # Return negative cost as reward (higher is better)
        return -total_cost

    def reset(self) -> None:
        """Reset the pusht task to initial conditions."""
        # Set initial state: qpos = [block_x, block_y, block_angle, pusher_x, pusher_y]
        self.data.qpos = np.array([0.1, 0.1, 1.3, 0.0, 0.0])
        self.data.qvel = np.zeros_like(self.data.qvel)

        mujoco.mj_forward(self.model, self.data)
