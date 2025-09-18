# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml/t_push.xml")


@slider("w_pusher_proximity", 0.0, 5.0, 0.1)
@slider("w_t_block_orientation", 0.0, 5.0, 0.1)
@dataclass
class TPushConfig(TaskConfig):
    """Reward configuration for the T-shaped push task."""

    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_t_block_position: float = 0.1
    w_t_block_orientation: float = 0.8
    pusher_goal_offset: float = 0.25
    goal_pos: np.ndarray = np_1d_field(
        np.array([0.0, 0.0]),
        names=["x", "y"],
        mins=[-1.0, -1.0],
        maxs=[1.0, 1.0],
        steps=[0.01, 0.01],
        vis_name="goal_position",
        xyz_vis_indices=[0, 1, None],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )
    goal_orientation: float = -3.14  # Target orientation in radians


class TPush(Task[TPushConfig]):
    """Defines the T-shaped push balancing task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the T-shaped push task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: TPushConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the T-shaped push reward from MJPC with orientation cost.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The T-shaped push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the T-block.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the T-block to the goal.
            * `orientation_reward`, penalizing the orientation error of the T-block.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        # Extract state components: [pusher_x, pusher_y, t_block_x, t_block_y, t_block_angle, pusher_vx, pusher_vy, t_block_vx, t_block_vy, t_block_angular_vel]
        pusher_pos = states[..., 0:2]
        t_block_pos = states[..., 2:4]
        t_block_angle = states[..., 4]
        pusher_vel = states[..., 5:7]
        t_block_goal = config.goal_pos[0:2]

        t_block_to_goal = t_block_goal - t_block_pos
        t_block_to_goal_norm = np.linalg.norm(t_block_to_goal, axis=-1, keepdims=True)
        t_block_to_goal_direction = t_block_to_goal / t_block_to_goal_norm

        pusher_goal = t_block_pos - config.pusher_goal_offset * t_block_to_goal_direction

        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)
        pusher_reward = -config.w_pusher_proximity * pusher_proximity.sum(-1)

        velocity_reward = -config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)

        goal_proximity = quadratic_norm(t_block_pos - t_block_goal)
        goal_reward = -config.w_t_block_position * goal_proximity.sum(-1)

        # Orientation cost: penalize deviation from target orientation
        angle_diff = t_block_angle - config.goal_orientation
        # Normalize angle difference to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        orientation_reward = -config.w_t_block_orientation * quadratic_norm(angle_diff)

        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert orientation_reward.shape == (batch_size,)

        return pusher_reward + velocity_reward + goal_reward + orientation_reward

    def reset(self) -> None:
        """Resets the model to a default state."""
        self.data.qpos = np.array([3, 1, 2, 0.0, 0.0])  # [pusher_x, pusher_y, t_block_x, t_block_y, t_block_angle]
        self.data.qvel = np.zeros(5)  # [pusher_vx, pusher_vy, t_block_vx, t_block_vy, t_block_angular_vel]
        mujoco.mj_forward(self.model, self.data) 