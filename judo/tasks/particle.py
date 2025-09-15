# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm

XML_PATH = str(MODEL_PATH / "xml/particle.xml")
SIM_XML_PATH = str(MODEL_PATH / "xml/particle_sim.xml")


@dataclass
class ParticleConfig(TaskConfig):
    """Reward configuration for the particle task."""

    w_position: float = 5.0  # Weight for position tracking cost
    w_velocity: float = 0.1  # Weight for velocity penalty
    w_control: float = 0.1  # Weight for control effort penalty


class Particle(Task[ParticleConfig]):
    """A velocity-controlled planar point mass chases a target position.

    This task is adapted from the hydrax particle task, where a 2D point mass
    is controlled with velocity commands to reach a target position marked by
    a green sphere.
    """

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str = SIM_XML_PATH) -> None:
        """Initialize the particle task."""
        super().__init__(model_path, sim_model_path=sim_model_path)

        # Get site and body IDs for tracking
        self.pointmass_site_id = self.model.site("pointmass").id
        self.goal_body_id = self.model.body("goal").id

        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: ParticleConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Compute the reward for the particle task.

        The reward encourages the particle to reach the target position while
        penalizing velocity and control effort.

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

        # Extract positions and velocities
        # For particle: qpos = [x, y], qvel = [vx, vy]
        positions = states[..., :2]  # [batch, time, 2]
        velocities = states[..., 2:]  # [batch, time, 2]

        # Get goal position from the mocap body (static in this case)
        # The goal is at position [0.25, 0, 0.01], so we use [0.25, 0] for 2D
        goal_pos = np.array([0.25, 0.0])

        # Position tracking cost - penalize distance to goal
        position_errors = positions - goal_pos[None, None, :]  # [batch, time, 2]
        position_cost = config.w_position * quadratic_norm(position_errors).sum(-1)  # [batch]

        # Velocity penalty - encourage the particle to come to rest
        velocity_cost = config.w_velocity * quadratic_norm(velocities).sum(-1)  # [batch]

        # Control effort penalty
        control_cost = config.w_control * quadratic_norm(controls).sum(-1)  # [batch]

        # Total cost (negative reward)
        total_cost = position_cost + velocity_cost + control_cost

        # Verify shapes
        assert position_cost.shape == (batch_size,)
        assert velocity_cost.shape == (batch_size,)
        assert control_cost.shape == (batch_size,)

        # Return negative cost as reward (higher is better)
        return -total_cost

    def reset(self) -> None:
        """Reset the particle to a deterministic starting position for fair comparison."""
        # Start at a fixed position with small deterministic offset for interesting dynamics
        # Use a consistent but non-trivial starting position
        self.data.qpos[:2] = np.array([-0.1, 0.05])  # Fixed starting position
        self.data.qvel[:2] = np.array([0.02, -0.01])  # Small fixed initial velocity

        # Set goal position (mocap body)
        goal_pos = np.array([0.25, 0.0, 0.01])
        if hasattr(self.data, "mocap_pos"):
            self.data.mocap_pos[0] = goal_pos

        mujoco.mj_forward(self.model, self.data)

    def pre_sim_step(self) -> None:
        """Update goal position before each simulation step."""
        # For now, keep the goal fixed, but this could be extended to move the goal
        goal_pos = np.array([0.25, 0.0, 0.01])
        if hasattr(self.data, "mocap_pos"):
            self.data.mocap_pos[0] = goal_pos
