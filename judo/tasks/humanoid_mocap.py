# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from huggingface_hub import hf_hub_download

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig

XML_PATH = str(MODEL_PATH / "xml/g1_23dof.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/g1_23dof_sim.xml")


@dataclass
class HumanoidMocapConfig(TaskConfig):
    """Reward configuration for the humanoid mocap task."""

    w_configuration: float = 1.0  # Weight for configuration tracking
    w_foot_position: float = 5.0  # Weight for foot position tracking
    w_foot_orientation: float = 0.1  # Weight for foot orientation tracking
    w_control: float = 1.0  # Weight for control penalty
    reference_filename: str = "Lafan1/mocap/UnitreeG1/walk1_subject1.npz"  # Motion capture reference file


class HumanoidMocap(Task[HumanoidMocapConfig]):
    """The Unitree G1 humanoid tracks a reference from motion capture.

    Retargeted motion capture data comes from the LocoMuJoCo dataset:
    https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main.
    """

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str = SIM_XML_PATH) -> None:
        """Initialize the humanoid mocap task."""
        # Check if the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"G1 model not found at {model_path}. "
                "Please ensure the hydrax G1 model files are available or copy them to judo's model directory."
            )

        super().__init__(model_path, sim_model_path=sim_model_path)

        # Get sensor IDs for foot tracking
        self.left_foot_pos_sensor_id = self.model.sensor("left_foot_position").id
        self.left_foot_quat_sensor_id = self.model.sensor("left_foot_orientation").id
        self.right_foot_pos_sensor_id = self.model.sensor("right_foot_position").id
        self.right_foot_quat_sensor_id = self.model.sensor("right_foot_orientation").id

        # Get site IDs for reference foot positions
        self.left_foot_site_id = self.model.site("left_foot").id
        self.right_foot_site_id = self.model.site("right_foot").id

        # Load default reference data
        self.load_reference_data("Lafan1/mocap/UnitreeG1/walk1_subject1.npz")

        # Cost weights for configuration tracking (base pose weighted more heavily)
        self.cost_weights = np.ones(self.model.nq)
        self.cost_weights[:7] = 5.0  # Base pose is more important

        self.reset()

    def load_reference_data(self, reference_filename: str) -> None:
        """Load motion capture reference data from HuggingFace."""
        # Download and load reference data
        npz_file = np.load(
            hf_hub_download(
                repo_id="robfiras/loco-mujoco-datasets",
                filename=reference_filename,
                repo_type="dataset",
            )
        )

        self.reference = np.array(npz_file["qpos"])
        self.reference_fps = npz_file["frequency"]

        # Precompute reference foot positions and orientations
        n_frames = len(self.reference)
        self.ref_left_pos = np.zeros((n_frames, 3))
        self.ref_left_quat = np.zeros((n_frames, 4))
        self.ref_right_pos = np.zeros((n_frames, 3))
        self.ref_right_quat = np.zeros((n_frames, 4))

        # Create temporary data for forward kinematics
        temp_data = mujoco.MjData(self.model)
        for i in range(n_frames):
            temp_data.qpos[:] = self.reference[i]
            mujoco.mj_forward(self.model, temp_data)

            # Store foot positions
            self.ref_left_pos[i] = temp_data.site_xpos[self.left_foot_site_id]
            self.ref_right_pos[i] = temp_data.site_xpos[self.right_foot_site_id]

            # Store foot orientations
            mujoco.mju_mat2Quat(
                self.ref_left_quat[i],
                temp_data.site_xmat[self.left_foot_site_id].flatten(),
            )
            mujoco.mju_mat2Quat(
                self.ref_right_quat[i],
                temp_data.site_xmat[self.right_foot_site_id].flatten(),
            )

    def get_reference_configuration(self, t: float) -> np.ndarray:
        """Get the reference position (q) at time t."""
        i = int(t * self.reference_fps)
        i = np.clip(i, 0, self.reference.shape[0] - 1)
        return self.reference[i, :]

    def get_reference_foot_data(self, t: float) -> tuple[np.ndarray, ...]:
        """Get the reference foot positions and orientations at time t."""
        i = int(t * self.reference_fps)
        i = np.clip(i, 0, self.reference.shape[0] - 1)
        return (
            self.ref_left_pos[i],
            self.ref_left_quat[i],
            self.ref_right_pos[i],
            self.ref_right_quat[i],
        )

    def get_foot_position_errors(self, sensors: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Get position errors for both feet."""
        ref_left_pos, _, ref_right_pos, _ = self.get_reference_foot_data(t)

        # Get sensor data
        left_pos_start = self.get_sensor_start_index("left_foot_position")
        right_pos_start = self.get_sensor_start_index("right_foot_position")

        left_pos = sensors[left_pos_start : left_pos_start + 3]
        right_pos = sensors[right_pos_start : right_pos_start + 3]

        left_err = left_pos - ref_left_pos
        right_err = right_pos - ref_right_pos

        return left_err, right_err

    def get_foot_orientation_errors(self, sensors: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Get orientation errors for both feet."""
        _, ref_left_quat, _, ref_right_quat = self.get_reference_foot_data(t)

        # Get sensor data
        left_quat_start = self.get_sensor_start_index("left_foot_orientation")
        right_quat_start = self.get_sensor_start_index("right_foot_orientation")

        left_quat = sensors[left_quat_start : left_quat_start + 4]
        right_quat = sensors[right_quat_start : right_quat_start + 4]

        # Quaternion difference (simplified version of mjx quat_sub)
        left_err = left_quat - ref_left_quat
        right_err = right_quat - ref_right_quat

        return left_err, right_err

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: HumanoidMocapConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute the reward for the humanoid mocap task.

        The reward encourages the humanoid to track motion capture reference data
        including joint configurations and foot positions/orientations.

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
        T = states.shape[1]

        # Extract joint positions (first nq elements of state)
        q = states[..., : self.model.nq]  # [batch, time, nq]

        # Initialize cost accumulation
        total_cost = np.zeros(batch_size)

        # Process each timestep
        for t in range(T):
            time_val = t * self.dt  # Convert timestep to time

            # Configuration tracking cost
            q_ref = self.get_reference_configuration(time_val)
            q_err = self.cost_weights[None, :] * (q[:, t, :] - q_ref[None, :])  # [batch, nq]
            configuration_cost = config.w_configuration * np.sum(np.square(q_err), axis=-1)  # [batch]

            # Foot tracking costs
            foot_position_cost = np.zeros(batch_size)
            foot_orientation_cost = np.zeros(batch_size)

            for b in range(batch_size):
                left_pos_err, right_pos_err = self.get_foot_position_errors(sensors[b, t, :], time_val)
                left_ori_err, right_ori_err = self.get_foot_orientation_errors(sensors[b, t, :], time_val)

                foot_position_cost[b] = config.w_foot_position * (
                    np.sum(np.square(left_pos_err)) + np.sum(np.square(right_pos_err))
                )
                foot_orientation_cost[b] = config.w_foot_orientation * (
                    np.sum(np.square(left_ori_err)) + np.sum(np.square(right_ori_err))
                )

            # Control penalty (using reference joint positions as control target)
            u_ref = q_ref[7:]  # Joint positions (excluding base)
            control_cost = config.w_control * np.sum(np.square(controls[:, t, :] - u_ref[None, :]), axis=-1)  # [batch]

            # Accumulate costs
            timestep_cost = configuration_cost + 0 * foot_position_cost + 0 * foot_orientation_cost + control_cost
            total_cost += timestep_cost

        # Verify shape
        assert total_cost.shape == (batch_size,)

        # Return negative cost as reward (higher is better)
        return -total_cost

    def reset(self) -> None:
        """Reset the humanoid to the first frame of the reference motion."""
        if hasattr(self, "reference"):
            # Start from the first frame of the reference motion
            self.data.qpos[:] = self.reference[0]
            self.data.qvel[:] = 0.0  # Start with zero velocity

            mujoco.mj_forward(self.model, self.data)
            print("Reference loaded successfully.")
        else:
            # Fallback to default reset if no reference loaded
            super().reset()
            print("Reference not loaded, using default reset.")
