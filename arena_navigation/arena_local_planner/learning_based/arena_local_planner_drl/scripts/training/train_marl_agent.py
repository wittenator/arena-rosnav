import sys
from datetime import time
from functools import partial
from typing import Callable, List
from multiprocessing import cpu_count

import numpy as np
import rospy
import rospkg
import os
from multiprocessing import cpu_count, set_start_method

from stable_baselines3 import PPO
from supersuit.vector import MakeCPUAsyncConstructor, ConcatVecEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

from rl_agent.utils.supersuit_utils import vec_env_create, MarkovVectorEnv_patched
from tools.argsparser import parse_marl_training_args
from tools.train_agent_utils import (
    get_agent_name,
    get_paths,
    choose_agent_model,
    initialize_hyperparameters,
)

import os, sys, rospy, time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    MarlEvalCallback,
)
from stable_baselines3.common.policies import BasePolicy

from rl_agent.model.agent_factory import AgentFactory
from rl_agent.model.base_agent import BaseAgent
from rl_agent.model.custom_policy import *
from rl_agent.model.custom_sb3_policy import *
from tools.argsparser import parse_training_args
from tools.custom_mlp_utils import *
from tools.train_agent_utils import *

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


def main(args):
    # generate agent name and model specific paths
    AGENT_NAME = get_agent_name(args)
    PATHS = get_paths(AGENT_NAME, args)

    # initialize hyperparameters (save to/ load from json)
    params = initialize_hyperparameters(
        PATHS=PATHS,
        load_target=args.load,
        config_name=args.config,
        n_envs=args.n_envs,
    )

    env = vec_env_create(
        env_fn,
        instantiate_drl_agents,
        num_robots=args.robots,
        num_cpus=cpu_count() - 1,
        num_vec_envs=args.n_envs,
    )
    model = choose_agent_model(AGENT_NAME, PATHS, args, env, params)

    # set num of timesteps to be generated
    n_timesteps = 40000000 if args.n is None else args.n

    start = time.time()
    try:
        model.learn(
            total_timesteps=n_timesteps,
            reset_num_timesteps=True,
            callback=get_evalcallback(
                env=env,
                num_robots=args.robots,
            ),
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")
    # finally:
    # update the timesteps the model has trained in total
    # update_total_timesteps_json(n_timesteps, PATHS)

    model.env.close()
    print(f"Time passed: {time.time() - start}s")
    print("Training script will be terminated")
    sys.exit()


def get_evalcallback(num_robots: int, env: VecEnv) -> MarlEvalCallback:
    eval_env = MarkovVectorEnv_patched(FlatlandPettingZooEnv(
        num_agents=num_robots,
        ns="eval_sim",
        agent_list_fn=instantiate_drl_agents,
        max_num_moves_per_eps=2000,
    ), black_death=True)

    return MarlEvalCallback(
        train_env=None,
        eval_env=eval_env,
        num_robots=num_robots,
        n_eval_episodes=10,
        eval_freq=1,
        deterministic=True,
    )


if __name__ == "__main__":
    set_start_method("fork")
    args, _ = parse_marl_training_args()
    # rospy.init_node("train_env", disable_signals=False, anonymous=True)
    main(args)
