import os
from functools import partial
from multiprocessing import shared_memory
from typing import Callable
import multiprocessing as mp

import cloudpickle
import gym
import numpy as np
import rospy
from supersuit.vector import MakeCPUAsyncConstructor, ConcatVecEnv, ProcConcatVec
from supersuit.vector.multiproc_vec import async_loop
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from supersuit.vector.utils.shared_array import SharedArray
from supersuit.vector.utils.space_wrapper import SpaceWrapper

class SharedArray_patched(SharedArray):
    def __setstate__(self, state):
        (self.shared_arr, self.dtype, self.shape) = state
        self._set_np_arr()

class ProcConcatVec_patched(ProcConcatVec):
    def __init__(self, vec_env_constrs, observation_space, action_space, tot_num_envs, metadata):
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs = tot_num_envs
        self.metadata = metadata

        self.shared_obs = SharedArray_patched((num_envs,) + self.observation_space.shape, dtype=self.observation_space.dtype)
        act_space_wrap = SpaceWrapper(self.action_space)
        self.shared_act = SharedArray_patched((num_envs,) + act_space_wrap.shape, dtype=act_space_wrap.dtype)
        self.shared_rews = SharedArray_patched((num_envs,), dtype=np.float32)
        self.shared_dones = SharedArray_patched((num_envs,), dtype=np.uint8)

        pipes = []
        procs = []
        for constr in vec_env_constrs:
            inpt, outpt = mp.Pipe()
            constr = constr
            proc = mp.Process(
                target=async_loop, args=(constr, inpt, outpt, self.shared_obs, self.shared_act, self.shared_rews, self.shared_dones)
            )
            proc.start()
            outpt.close()
            pipes.append(inpt)
            procs.append(proc)

        self.pipes = pipes
        self.procs = procs

        num_envs = 0
        env_nums = self._receive_info()
        idx_starts = []
        for pipe, cnum_env in zip(self.pipes, env_nums):
            cur_env_idx = num_envs
            num_envs += cnum_env
            pipe.send(cur_env_idx)
            idx_starts.append(cur_env_idx)

        assert num_envs == tot_num_envs
        self.idx_starts = idx_starts

class call_wrap_patched:
    def __init__(self, fn, data):
        print(os.getpid())
        cloudpickle.dumps(data[0])
        self.fn = fn
        self.data = data

    def __call__(self, *args):
        rospy.init_node("train_env", disable_signals=False, anonymous=True)
        print(os.getpid())
        return self.fn(self.data)

def MakeCPUAsyncConstructor_patched(max_num_cpus):
    if max_num_cpus == 0 or max_num_cpus == 1:
        rospy.init_node("train_env", disable_signals=False, anonymous=True)
        return ConcatVecEnv
    else:

        def constructor(env_fn_list, obs_space, act_space):
            example_env = env_fn_list[0]()
            envs_per_env = getattr(example_env, "num_envs", 1)

            num_fns = len(env_fn_list)
            envs_per_cpu = (num_fns + max_num_cpus - 1) // max_num_cpus
            alloced_num_cpus = (num_fns + envs_per_cpu - 1) // envs_per_cpu

            env_cpu_div = []
            num_envs_alloced = 0
            while num_envs_alloced < num_fns:
                start_idx = num_envs_alloced
                end_idx = min(num_fns, start_idx + envs_per_cpu)
                env_cpu_div.append(env_fn_list[start_idx:end_idx])
                num_envs_alloced = end_idx

            assert alloced_num_cpus == len(env_cpu_div)

            cat_env_fns = [call_wrap_patched(ConcatVecEnv, env_fns) for env_fns in env_cpu_div]
            return ProcConcatVec_patched(cat_env_fns, obs_space, act_space, num_fns * envs_per_env, example_env.metadata)

        return constructor

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

    num_cpus = min(num_cpus, num_vec_envs)
    vec_env = MakeCPUAsyncConstructor_patched(num_cpus)(
        env_list_fns, observation_space, action_space
    )
    return SB3VecEnvWrapper(vec_env)