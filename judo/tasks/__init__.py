# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from typing import Dict, Tuple, Type

from judo.tasks.base import Task, TaskConfig
from judo.tasks.box_push import BoxPush, BoxPushConfig
from judo.tasks.caltech_leap_cube import CaltechLeapCube, CaltechLeapCubeConfig
from judo.tasks.cartpole import Cartpole, CartpoleConfig
from judo.tasks.cylinder_push import CylinderPush, CylinderPushConfig
from judo.tasks.fr3_pick import FR3Pick, FR3PickConfig
from judo.tasks.fr3_reach import FR3Reach, FR3ReachConfig
from judo.tasks.humanoid_mocap import HumanoidMocap, HumanoidMocapConfig
from judo.tasks.humanoid_standup import HumanoidStandup, HumanoidStandupConfig
from judo.tasks.leap_cube import LeapCube, LeapCubeConfig
from judo.tasks.leap_cube_down import LeapCubeDown, LeapCubeDownConfig
from judo.tasks.particle import Particle, ParticleConfig
from judo.tasks.pusht import Pusht, PushtConfig

_registered_tasks: Dict[str, Tuple[Type[Task], Type[TaskConfig]]] = {
    "box_push": (BoxPush, BoxPushConfig),
    "cylinder_push": (CylinderPush, CylinderPushConfig),
    "cartpole": (Cartpole, CartpoleConfig),
    "fr3_pick": (FR3Pick, FR3PickConfig),
    "fr3_reach": (FR3Reach, FR3ReachConfig),
    "leap_cube": (LeapCube, LeapCubeConfig),
    "leap_cube_down": (LeapCubeDown, LeapCubeDownConfig),
    "caltech_leap_cube": (CaltechLeapCube, CaltechLeapCubeConfig),
    "particle": (Particle, ParticleConfig),
    "pusht": (Pusht, PushtConfig),
    "humanoid_mocap": (HumanoidMocap, HumanoidMocapConfig),
    "humanoid_standup": (HumanoidStandup, HumanoidStandupConfig),
}


def get_registered_tasks() -> Dict[str, Tuple[Type[Task], Type[TaskConfig]]]:
    """Returns a dictionary of registered tasks."""
    return _registered_tasks


def register_task(name: str, task_type: Type[Task], task_config_type: Type[TaskConfig]) -> None:
    """Registers a new task."""
    _registered_tasks[name] = (task_type, task_config_type)


__all__ = [
    "get_registered_tasks",
    "register_task",
    "Task",
    "TaskConfig",
    "BoxPush",
    "BoxPushConfig",
    "CaltechLeapCube",
    "CaltechLeapCubeConfig",
    "Cartpole",
    "CartpoleConfig",
    "CylinderPush",
    "CylinderPushConfig",
    "FR3Pick",
    "FR3PickConfig",
    "FR3Reach",
    "FR3ReachConfig",
    "LeapCube",
    "LeapCubeConfig",
    "LeapCubeDown",
    "LeapCubeDownConfig",
    "Particle",
    "ParticleConfig",
    "Pusht",
    "PushtConfig",
    "HumanoidMocap",
    "HumanoidMocapConfig",
    "HumanoidStandup",
    "HumanoidStandupConfig",
]
