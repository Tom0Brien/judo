# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.utils.fields import np_1d_field
from judo.utils.math_utils import quat_diff_so3

XML_PATH = str(MODEL_PATH / "xml/fr3_reach.xml")

# Home position for the FR3 arm (7 DOF + gripper)
QPOS_HOME = np.array([
    -0.196,
    -0.189,
    0.182,
    -2.1,
    0.0378,
    1.91,
    0.756,
    0,
    0,
])


@slider("w_position", 0.0, 100.0, 0.1)
@slider("w_orientation", 0.0, 100.0, 0.1) 
@slider("w_control", 0.0, 10.0, 0.001)
@slider("w_velocity", 0.0, 10.0, 0.001)
@dataclass
class FR3ReachConfig(TaskConfig):
    """Configuration for FR3 reach task."""
    
    # Target position (can be adjusted via GUI)
    target_pos: np.ndarray = np_1d_field(
        np.array([0.5, 0.0, 0.4]),
        names=["x", "y", "z"],
        mins=[0.2, -0.5, 0.3],
        maxs=[1.0, 0.5, 1.0],
        steps=[0.01, 0.01, 0.01],
        vis_name="target_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.6, 0.2, 0.6],
    )
    
    # Target orientation as quaternion (w, x, y, z)
    target_quat: np.ndarray = np_1d_field(
        np.array([0, -1, 0, 0]),
        names=["w", "x", "y", "z"],
        mins=[-1.0, -1.0, -1.0, -1.0],
        maxs=[1.0, 1.0, 1.0, 1.0],
        steps=[0.01, 0.01, 0.01, 0.01],
        vis_name="target_orientation",
    )
    
    # Reward weights
    w_position: float = 10.0     # Position tracking weight
    w_orientation: float = 1.0   # Orientation tracking weight  
    w_control: float = 0.01      # Control effort penalty
    w_velocity: float = 0.005    # Velocity penalty


class FR3Reach(Task[FR3ReachConfig]):
    """Franka FR3 reaching task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initialize the FR3 reach task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        
        # Home command for the actuators
        self.reset_command = np.array([     0.5,
            0.0,
            0.4,
            -3.14,
            0.0,
            0.0,])
        
        # Get joint indices for the arm (7 DOF + 1 gripper)
        arm_pos_adr = self.get_joint_position_start_index("fr3_joint1")
        self.arm_pos_slice = slice(arm_pos_adr, arm_pos_adr + 8)  # 7 arm + 1 gripper DOF
        
        arm_vel_adr = self.get_joint_velocity_start_index("fr3_joint1") 
        self.arm_vel_slice = slice(arm_vel_adr, arm_vel_adr + 8)
        
        # Get sensor indices
        self.gripper_pos_adr = self.get_sensor_start_index("gripper_position")
        self.gripper_pos_slice = slice(self.gripper_pos_adr, self.gripper_pos_adr + 3)
        
        self.gripper_quat_adr = self.get_sensor_start_index("gripper_orientation") 
        self.gripper_quat_slice = slice(self.gripper_quat_adr, self.gripper_quat_adr + 4)
        
        self.target_pos_adr = self.get_sensor_start_index("target_position")
        self.target_pos_slice = slice(self.target_pos_adr, self.target_pos_adr + 3)
        
        self.target_quat_adr = self.get_sensor_start_index("target_orientation")
        self.target_quat_slice = slice(self.target_quat_adr, self.target_quat_adr + 4)
        
        self.reset()



    def pre_rollout(self, curr_state: np.ndarray, config: FR3ReachConfig) -> None:
        """Update mocap target position and orientation before rollout."""
        # Update target position in mocap body
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if target_body_id >= 0:
            self.data.mocap_pos[0] = config.target_pos
            self.data.mocap_quat[0] = config.target_quat / np.linalg.norm(config.target_quat)  # normalize quaternion

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: FR3ReachConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute reward for FR3 reach task.
        
        Args:
            states: Shape (num_rollouts, T, nq + nv)
            sensors: Shape (num_rollouts, T, sensor_dim)  
            controls: Shape (num_rollouts, T, nu)
            config: Task configuration
            system_metadata: Additional metadata
            
        Returns:
            rewards: Shape (num_rollouts,)
        """
        # Extract state components
        arm_positions = states[..., self.arm_pos_slice]  # (num_rollouts, T, 8)
        arm_velocities = states[..., self.arm_vel_slice]  # (num_rollouts, T, 8)
        
        # Get gripper and target positions/orientations from sensors
        gripper_pos = sensors[..., self.gripper_pos_slice]  # (num_rollouts, T, 3)
        gripper_quat = sensors[..., self.gripper_quat_slice]  # (num_rollouts, T, 4)
        target_pos = config.target_pos
        target_quat = config.target_quat
        
        # Position error
        position_error = gripper_pos - target_pos  # (num_rollouts, T, 3)
        position_cost = np.sum(position_error ** 2, axis=-1)  # (num_rollouts, T)
        
        # Orientation error using proper quaternion difference (like leap_cube)
        quat_diff = quat_diff_so3(gripper_quat, target_quat)  # (num_rollouts, T, 3)
        orientation_cost = 0.5 * np.square(quat_diff).sum(-1)  # (num_rollouts, T)
        
        # Control effort cost
        control_cost = np.sum(controls ** 2, axis=-1)  # (num_rollouts, T)
        
        # Velocity cost  
        velocity_cost = np.sum(arm_velocities ** 2, axis=-1)  # (num_rollouts, T)
        
        # Combine costs with weights (negative because we want to minimize)
        total_cost = (
            config.w_position * position_cost +
            config.w_orientation * orientation_cost + 
            config.w_control * control_cost +
            config.w_velocity * velocity_cost
        )  # (num_rollouts, T)
        
        # Sum over time and negate for reward (higher is better)
        rewards = -total_cost.sum(axis=-1)  # (num_rollouts,)
        
        return rewards

    def reset(self) -> None:
        """Reset the robot to home position."""
        self.data.qpos[:len(QPOS_HOME)] = QPOS_HOME
        self.data.qvel[:] = 0.0
        self.data.ctrl[:len(self.reset_command)] = self.reset_command
        
        # Set initial target position  
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        if target_body_id >= 0:
            self.data.mocap_pos[0] = [0.6, 0.2, 0.6]
            self.data.mocap_quat[0] = [1.0, 0.0, 0.0, 0.0]
        
        mujoco.mj_forward(self.model, self.data) 