from typing import List, Tuple, Dict

import numpy as np
import rospy
import rospkg

from pettingzoo import *
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

from rl_agent.training_agent_wrapper import TrainingDRLAgent
from task_generator.marl_tasks import get_MARL_task

from flatland_msgs.srv import StepWorld, StepWorldRequest


def env():
    """
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    env = FlatlandPettingZooEnv()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
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
    ) -> None:
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        """
        self._ns = "" if ns is None or ns == "" else ns + "/"
        self._is_train_mode = rospy.get_param("/train_mode")

        self.possible_agents = [a._robot_sim_ns for a in agent_list]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.agent_object_mapping = dict(zip(self.possible_agents, agent_list))
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

        self._robot_sim_ns = [a._robot_sim_ns for a in agent_list]

        # task manager
        self.task_manager = get_MARL_task(
            ns=self._ns,
            mode=task_mode,
            robot_names=self._robot_sim_ns,
        )

        # service clients
        if self._is_train_mode:
            self._service_name_step = f"{self._ns}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )

    def _validate_agent_list(self) -> None:
        # check if all agents named differently (target different namespaces)
        assert len(self.possible_agents) == len(set(self.possible_agents))

    def reset(self) -> Dict[str, np.ndarray]:
        """
        resets the environment and returns a dictionary of observations (keyed by the agent name)
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0

        (
            self.agent_object_mapping[agent].reward_calculator.reset()
            for agent in self.agents
        )

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
        Dict[str, dict],
    ]:
        """
        receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, done dictionary,
        and info dictionary, where each dictionary is keyed by the agent.
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # actions
        for agent, action in actions.items():
            self.agent_object_mapping[agent].publish_action(action)

        # fast-forward simulation
        self.call_service_takeSimStep()

        # observations
        observations: Dict[
            str, Tuple[np.ndarray, dict]
        ] = {  # [0] - merged obs, [1] - obs dict
            agent: self.agent_object_mapping[agent].get_observations()
            for agent in self.agents
        }

        mergeds_obs: Dict[str, np.ndarray] = {
            agent: observations[agent][0] for agent in self.agents
        }

        # rewards and infos
        rewards_and_info: Dict[str, float] = {
            agent: self.agent_object_mapping[agent].get_reward(
                action=actions[agent], obs_dict=observations[agent][1]
            )
            for agent in self.agents
        }

        rewards, infos = {}, {}
        for agent, reward_and_info in rewards_and_info.items():
            rewards[agent] = reward_and_info[0]
            infos[agent] = reward_and_info[1]

        # dones
        # TODO: when do we consider the episode as done?
        dones = {}

        return mergeds_obs, rewards, dones, infos

    def render(self, mode="human"):
        """
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        raise NotImplementedError

    def state(self):
        """
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        """
        raise NotImplementedError(
            "state() method has not been implemented in the environment {}.".format(
                self.metadata.get("name", self.__class__.__name__)
            )
        )

    @property
    def max_num_agents(self):
        return len(self.agents)

    def call_service_takeSimStep(self, t: float = None):
        request = StepWorldRequest() if t is None else StepWorldRequest(t)

        try:
            response = self._sim_step_client(request)
            rospy.logdebug("step service=", response)
        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)
