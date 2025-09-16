# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm
from judo.utils.math_utils import quat_diff, quat_diff_so3

XML_PATH = str(MODEL_PATH / "xml/g1_23dof.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/g1_23dof_sim.xml")


@dataclass
class HumanoidStandupConfig(TaskConfig):
    """Reward configuration for the humanoid standup task."""

    w_orientation: float = 10.0  # Weight for torso orientation (staying upright)
    w_height: float = 10.0  # Weight for torso height tracking
    w_nominal: float = 0.1  # Weight for nominal joint configuration
    w_control: float = 0.01  # Weight for control penalty
    target_height: float = 0.9  # Target height for the torso


class HumanoidStandup(Task[HumanoidStandupConfig]):
    """The Unitree G1 humanoid stands up and maintains an upright pose.
    
    This task encourages the humanoid to:
    1. Maintain an upright orientation (minimize torso tilt)
    2. Maintain a target height 
    3. Keep joints close to a nominal standing configuration
    4. Minimize control effort
    """

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str = SIM_XML_PATH) -> None:
        """Initialize the humanoid standup task."""
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"G1 model not found at {model_path}. "
                "Please ensure the G1 model files are available in judo's model directory."
            )

        super().__init__(model_path, sim_model_path=sim_model_path)

        # Standing configuration
        self.qstand = np.array(self.model.keyframe("stand").qpos)

        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: HumanoidStandupConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute the reward for the humanoid standup task.

        The reward encourages the humanoid to stand upright and maintain
        a stable standing pose.

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
        

        # Height tracking reward
        torso_position_sensor_start = self.get_sensor_start_index("imu_in_torso_position")
        torso_position = sensors[..., torso_position_sensor_start : torso_position_sensor_start + 3]
        torso_height = torso_position[..., 2]
        target_height = config.target_height * np.ones_like(torso_height)
        height_reward = -config.w_height * quadratic_norm(torso_height - target_height)

        # Orientation tracking reward
        orientation_sensor_start = self.get_sensor_start_index("imu_in_torso_quat")
        orientation = sensors[..., orientation_sensor_start : orientation_sensor_start + 4]
        orientation_reward = -config.w_orientation * quadratic_norm(quat_diff_so3(orientation, self.qstand[3:7])).sum(-1)

        # Nominal joint configuration reward
        joints = states[..., :self.model.nq]
        joint_reward = -config.w_nominal * quadratic_norm(joints - self.qstand[:self.model.nq]).sum(-1)
         
        assert height_reward.shape == (batch_size,)
        assert orientation_reward.shape == (batch_size,)
        assert joint_reward.shape == (batch_size,)

        # Return negative cost as reward (higher is better)
        return height_reward + orientation_reward + joint_reward

    def reset(self) -> None:
        """Reset the humanoid to a fallen position that requires standing up."""
        if hasattr(self, "qstand"):
            # Start from the standing pose but make it fall over
            self.data.qpos[:] = self.qstand.copy()
            
            # Tilt the robot to make it fall (modify base orientation)
            # Original standing quaternion is [1, 0, 0, 0] (upright)
            # Set to a tilted orientation that will make it fall
            self.data.qpos[2] = 0
            self.data.qpos[3:7] = [0.7, 0.0, -0.7, 0.0]  # Tilted orientation
            
            # Set the base position to the ground
            self.data.qpos[0] = 0.0
            self.data.qpos[1] = 0.0
            
            # Zero velocity
            self.data.qvel[:] = 0.0

            mujoco.mj_forward(self.model, self.data)
            print("Robot initialized in fallen position for standup task.")
        else:
            # Fallback to default reset if no reference loaded
            super().reset()
            print("Using default reset (standing reference not available).") 