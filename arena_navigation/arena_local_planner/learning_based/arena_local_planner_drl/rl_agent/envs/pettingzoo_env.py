from typing import List

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

        # task manager + robot managers
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

    def reset(self):
        """
        resets the environment and returns a dictionary of observations (keyed by the agent name)
        """
        raise NotImplementedError

    def seed(self, seed=None):
        """
        Reseeds the environment (making it deterministic).
        `reset()` must be called after `seed()`, and before `step()`.
        """
        pass

    def step(self, actions):
        """
        receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, done dictionary,
        and info dictionary, where each dictionary is keyed by the agent.
        """
        raise NotImplementedError

    def render(self, mode="human"):
        """
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside
        of classic, and `'ansi'` which returns the strings printed
        (specific to classic environments).
        """
        raise NotImplementedError

    def close(self):
        """
        Closes the rendering window.
        """
        pass

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
    def num_agents(self):
        return len(self.agents)

    @property
    def max_num_agents(self):
        pass

    def __str__(self):
        """
        returns a name which looks like: "space_invaders_v1" by default
        """
        if hasattr(self, "metadata"):
            return self.metadata.get("name", self.__class__.__name__)
        else:
            return self.__class__.__name__

    @property
    def unwrapped(self):
        return self

    def call_service_takeSimStep(self, t: float = None):
        request = StepWorldRequest() if t is None else StepWorldRequest(t)

        try:
            response = self._sim_step_client(request)
            rospy.logdebug("step service=", response)
        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)
