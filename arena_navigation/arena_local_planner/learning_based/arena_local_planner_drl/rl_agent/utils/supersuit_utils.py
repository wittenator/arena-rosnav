from functools import partial
from typing import Callable

import rospy
from supersuit.vector import ConcatVecEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

def vec_env_create(
    env_fn: Callable,
    agent_list_fn: Callable,
    num_robots: int,
    num_cpus: int,
    num_vec_envs: int,
) -> SB3VecEnvWrapper:
    env_list_fns = [
        partial(
            env_fn,
            ns=f"sim_{i}",
            num_agents=num_robots,
            agent_list_fn=agent_list_fn,
        )
        for i in range(1, num_vec_envs + 1)
    ]
    # TODO: That could be a problem
    env = env_list_fns[0]()
    action_space = env.observation_space
    observation_space = env.observation_space
    metadata = env.metadata

    num_cpus = min(num_cpus, num_vec_envs)
    rospy.init_node("train_env", disable_signals=False, anonymous=True)
    vec_env = ConcatVecEnv(
        env_list_fns, observation_space, action_space
    )
    return SB3VecEnvWrapper(vec_env)