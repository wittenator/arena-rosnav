from time import sleep
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import rospy

from pettingzoo import *
from pettingzoo.utils import wrappers, from_parallel, to_parallel
import supersuit as ss

from rl_agent.training_agent_wrapper import TrainingDRLAgent
from task_generator.marl_tasks import get_MARL_task

from flatland_msgs.srv import StepWorld, StepWorldRequest


def env_fn(**kwargs: dict) -> ParallelEnv:
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = FlatlandPettingZooEnv(**kwargs)
    #env = ss.pad_action_space_v0(env)
    #env = ss.pad_observations_v0(env)
    #env = ss.black_death_v2(env)
    env = ss.pettingzoo_env_to_vec_env_v0(env)
    return env


class FlatlandPettingZooEnv(ParallelEnv):
    """
    The Parallel environment steps every live agent at once. If you are unsure if you
    have implemented a ParallelEnv correctly, try running the `parallel_api_test` in
    the Developer documentation on the website.
    """

    def __init__(
        self,
        ns: str = None,
        agent_list: List[TrainingDRLAgent] = [],
        task_mode: str = "random",
        max_num_moves_per_eps: int = 1000,
    ) -> None:
        """[summary]

        Description:
            The init method takes in environment arguments and
            should define the following attributes:
            - possible_agents
            - action_spaces
            - observation_spaces

            These attributes should not be changed after initialization.

        Args:
            ns (str, optional): [description]. Defaults to None.
            agent_list (List[TrainingDRLAgent], optional): [description]. Defaults to [].
            task_mode (str, optional): [description]. Defaults to "random".
            max_num_moves_per_eps (int, optional): [description]. Defaults to 1000.
        """
        self._ns = "" if ns is None or ns == "" else ns + "/"
        rospy.init_node(f"train_env_{ns}", disable_signals=False)
        self._is_train_mode = rospy.get_param("/train_mode")
        self.metadata = {"render.modes": ["human"], "name": "rps_v2"}

        self.agents = []
        self.possible_agents = [a._robot_sim_ns for a in agent_list]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_object_mapping = dict(zip(self.possible_agents, agent_list))
        self._robot_sim_ns = [a._robot_sim_ns for a in agent_list]

        self._validate_agent_list()

        # action space
        self.action_spaces = {
            agent: agent_list[i].action_space
            for i, agent in enumerate(self.possible_agents)
        }

        # observation space
        self.observation_spaces = {
            agent: agent_list[i].observation_space
            for i, agent in enumerate(self.possible_agents)
        }

        # task manager
        self.task_manager = get_MARL_task(
            ns=ns,
            mode=task_mode,
            robot_names=self._robot_sim_ns,
        )

        # service clients
        if self._is_train_mode:
            self._service_name_step = f"{self._ns}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

        self._max_num_moves = max_num_moves_per_eps

    def observation_space(self, agent):
        return self.agent_object_mapping[agent].observation_space

    def action_space(self, agent):
        return self.agent_object_mapping[agent].action_space

    def _validate_agent_list(self) -> None:
        # check if all agents named differently (target different namespaces)
        assert len(self.possible_agents) == len(
            set(self.possible_agents)
        ), "Robot names and thus their namespaces, have to be unique!"

    def reset(self) -> Dict[str, np.ndarray]:
        """Resets the environment and returns a dictionary of observations (keyed by the agent name)

        Returns:
            Dict[str, np.ndarray]: [description]
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        

        for agent in self.agents:
            self.agent_object_mapping[agent].reward_calculator.reset()

        self.task_manager.reset()

        if self._is_train_mode:
            self._sim_step_client()
        observations = {
            agent: self.agent_object_mapping[agent].get_observations()[0]
            for agent in self.agents
        }

        return observations

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """[summary]

        Description:
            Receives a dictionary of actions keyed by the agent name.
            Returns the observation dictionary, reward dictionary, done dictionary,
            and info dictionary, where each dictionary is keyed by the agent.

        Args:
            actions (Dict[str, np.ndarray]): [description]

        Returns:
            Tuple[ Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]], ]:
                [description]
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # actions
        for agent in self.possible_agents:
            if agent in actions.keys():
                self.agent_object_mapping[agent].publish_action(actions[agent])
            else:
                noop = np.zeros(shape=self.observation_space(agent).shape)
                self.agent_object_mapping[agent].publish_action(noop)

        # fast-forward simulation
        self.call_service_takeSimStep()
        self.num_moves += 1

        merged_obs, rewards, reward_infos = {}, {}, {}

        for agent in actions.keys():
            # observations
            merged, _dict = self.agent_object_mapping[agent].get_observations()
            merged_obs[agent] = merged

            # rewards and infos
            reward, reward_info = self.agent_object_mapping[agent].get_reward(
                action=actions[agent], obs_dict=_dict
            )
            rewards[agent], reward_infos[agent] = reward, reward_info

        # dones & infos
        dones, infos = self._get_dones(reward_infos), self._get_infos(
            reward_infos
        )

        self.agents = [agent for agent in self.agents if not dones[agent]]

        return merged_obs, rewards, dones, infos

    @property
    def max_num_agents(self):
        return len(self.agents)

    def call_service_takeSimStep(self, t: float = None):
        """Fast-forwards the simulation time.

        Args:
            t (float, optional):
                Time in seconds. When t is None, time is forwarded by 'step_size' s.
                Defaults to None.
        """
        request = StepWorldRequest() if t is None else StepWorldRequest(t)

        try:
            response = self._sim_step_client(request)
            rospy.logdebug("step service=", response)
        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)

    def _get_dones(
        self, reward_infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, bool]:
        """[summary]

        Args:
            reward_infos (Dict[str, Dict[str, Any]]): [description]

        Returns:
            Dict[str, bool]: [description]
        """
        return (
            {agent: reward_infos[agent]["is_done"] for agent in self.agents}
            if self.num_moves < self._max_num_moves
            else {agent: True for agent in self.agents}
        )

    def _get_infos(
        self, reward_infos: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """[summary]

        Args:
            reward_infos (Dict[str, Dict[str, Any]]): [description]

        Returns:
            Dict[str, Dict[str, Any]]: [description]
        """
        infos = {agent: {} for agent in self.agents}
        for agent in self.agents:
            if reward_infos[agent]["is_done"]:
                infos[agent] = {
                    "done_reason": reward_infos[agent]["done_reason"],
                    "is_success": reward_infos[agent]["is_success"],
                }
            elif self.num_moves >= self._max_num_moves:
                infos[agent] = {
                    "done_reason": 0,
                    "is_success": 0,
                }
        return infos