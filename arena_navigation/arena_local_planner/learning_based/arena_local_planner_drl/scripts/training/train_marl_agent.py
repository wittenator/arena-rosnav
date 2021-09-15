import sys
from functools import partial
from typing import Callable, List

import numpy as np
import rospy
import rospkg
import os
from multiprocessing import cpu_count, set_start_method

from stable_baselines3 import PPO
from supersuit.vector import MakeCPUAsyncConstructor
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

set_start_method("fork")
rospy.set_param("/MARL", True)

from rl_agent.training_agent_wrapper import TrainingDRLAgent
from scripts.deployment.drl_agent_node import DeploymentDRLAgent
from rl_agent.envs.pettingzoo_env import FlatlandPettingZooEnv, env_fn

from nav_msgs.srv import GetMap

DEFAULT_HYPERPARAMETER = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    "hyperparameters",
    "default.json",
)
DEFAULT_ACTION_SPACE = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    "default_settings.yaml",
)


def instantiate_drl_agents(
    num_robots: int = 1,
    ns: str = None,
    robot_name_prefix: str = "robot",
    hyperparameter_path: str = DEFAULT_HYPERPARAMETER,
    action_space_path: str = DEFAULT_ACTION_SPACE,
) -> list:
    return [
        TrainingDRLAgent(
            ns=ns,
            robot_name=robot_name_prefix + str(i + 1),
            hyperparameter_path=hyperparameter_path,
            action_space_path=action_space_path,
        )
        for i in range(num_robots)
    ]


def main():
    rospy.set_param("/MARL", True)
    rospy.init_node(f"USER_NODE", anonymous=True)
    env = vec_env_create(env_fn, instantiate_drl_agents, 2, 1, 2)
    model = PPO('MlpPolicy', env, verbose=3, n_steps=16)
    model.learn(total_timesteps=2000000)

def vec_env_create(env_fn: Callable, agent_list_fn: Callable, num_robots: int, num_cpus: int, num_vec_envs: int) -> SB3VecEnvWrapper:
    env_list_fns = [partial(env_fn, ns=f"sim_{i}", agent_list=agent_list_fn(num_robots, ns=f"sim_{i}")) for i in range(1, num_vec_envs+1)]
    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor(num_cpus)(env_list_fns, None, None)
    return SB3VecEnvWrapper(vec_env)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
