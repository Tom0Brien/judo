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

XML_PATH = str(MODEL_PATH / "xml/box_push.xml")


@slider("w_pusher_proximity", 0.0, 5.0, 0.1)
@slider("w_cart_orientation", 0.0, 5.0, 0.1)
@dataclass
class BoxPushConfig(TaskConfig):
    """Reward configuration for the box push task."""

    w_pusher_proximity: float = 0.5
    w_pusher_velocity: float = 0.0
    w_cart_position: float = 0.1
    w_cart_orientation: float = 0.8
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


class BoxPush(Task[BoxPushConfig]):
    """Defines the box push balancing task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the box push task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: BoxPushConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the box push reward from MJPC with orientation cost.

        Maps a list of states, list of controls, to a batch of rewards (summed over time) for each rollout.

        The box push reward has four terms:
            * `pusher_reward`, penalizing the distance between the pusher and the cart.
            * `velocity_reward` penalizing squared linear velocity of the pusher.
            * `goal_reward`, penalizing the distance from the cart to the goal.
            * `orientation_reward`, penalizing the orientation error of the cart.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        """
        batch_size = states.shape[0]

        # Extract state components: [pusher_x, pusher_y, cart_x, cart_y, cart_angle, pusher_vx, pusher_vy, cart_vx, cart_vy, cart_angular_vel]
        pusher_pos = states[..., 0:2]
        cart_pos = states[..., 2:4]
        cart_angle = states[..., 4]
        pusher_vel = states[..., 5:7]
        cart_goal = config.goal_pos[0:2]

        cart_to_goal = cart_goal - cart_pos
        cart_to_goal_norm = np.linalg.norm(cart_to_goal, axis=-1, keepdims=True)
        cart_to_goal_direction = cart_to_goal / cart_to_goal_norm

        pusher_goal = cart_pos - config.pusher_goal_offset * cart_to_goal_direction

        pusher_proximity = quadratic_norm(pusher_pos - pusher_goal)
        pusher_reward = -config.w_pusher_proximity * pusher_proximity.sum(-1)

        velocity_reward = -config.w_pusher_velocity * quadratic_norm(pusher_vel).sum(-1)

        goal_proximity = quadratic_norm(cart_pos - cart_goal)
        goal_reward = -config.w_cart_position * goal_proximity.sum(-1)

        # Orientation cost: penalize deviation from target orientation
        angle_diff = cart_angle - config.goal_orientation
        # Normalize angle difference to [-pi, pi]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        orientation_reward = -config.w_cart_orientation * quadratic_norm(angle_diff)

        assert pusher_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
        assert goal_reward.shape == (batch_size,)
        assert orientation_reward.shape == (batch_size,)

        return pusher_reward + velocity_reward + goal_reward + orientation_reward

    def reset(self) -> None:
        """Resets the model to a default state."""
        self.data.qpos = np.array([3, 1, 2, 0.0, 0.0])  # [pusher_x, pusher_y, cart_x, cart_y, cart_angle]
        self.data.qvel = np.zeros(5)  # [pusher_vx, pusher_vy, cart_vx, cart_vy, cart_angular_vel]
        mujoco.mj_forward(self.model, self.data) 