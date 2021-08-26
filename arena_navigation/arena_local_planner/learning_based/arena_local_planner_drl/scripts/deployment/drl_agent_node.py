#!/usr/bin/env python
from typing import Tuple

import json
import numpy as np
import os
import pickle
import rospy
import rospkg
import sys
import yaml

from gym import spaces
from stable_baselines3 import PPO

from flatland_msgs.srv import StepWorld, StepWorldRequest
from geometry_msgs.msg import Twist
from rospy.exceptions import ROSException
from std_msgs.msg import Bool

from rl_agent.base_agent import BaseDRLAgent
from rl_agent.utils.observation_collector import ObservationCollector
from rl_agent.utils.reward import RewardCalculator


""" TEMPORARY GLOBAL CONSTANTS """
NS_PREFIX = ""
MODELS_DIR = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"), "agents"
)
ROOT_ROBOT_PATH = os.path.join(
    rospkg.RosPack().get_path("simulator_setup"), "robot"
)
DEFAULT_ROBOT_SETTING = os.path.join(
    ROOT_ROBOT_PATH,
    "myrobot.model.yaml",
)
DEFAULT_ACTION_SPACE = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    "default_settings.yaml",
)
LASER_NUM_BEAMS, LASER_MAX_RANGE = 360, 3.5
GOAL_RADIUS = 0.33


class DeploymentDRLAgent(BaseDRLAgent):
    def __init__(
        self,
        agent_name: str,
        ns: str = None,
        robot_name: str = None,
        action_space_path: str = DEFAULT_ACTION_SPACE,
        *args,
        **kwargs,
    ) -> None:
        """Initialization procedure for the DRL agent node.

        Args:
            agent_name (str):
                Agent name (directory has to be of the same name)
            robot_name (str, optional):
                Robot specific ROS namespace extension. Defaults to None.
            ns (str, optional):
                Simulation specific ROS namespace. Defaults to None.
            action_space_path (str, optional):
                Path to yaml file containing action space settings.
                Defaults to DEFAULT_ACTION_SPACE.
        """
        self._is_train_mode = rospy.get_param("/train_mode")

        if not self._is_train_mode:
            rospy.init_node(f"DRL_local_planner", anonymous=True)

        self.name = agent_name
        self.setup_agent()

        self._ns = "" if ns is None or ns == "" else "/" + ns + "/"
        self._ns_robot = (
            self._ns if robot_name is None else self._ns + robot_name + "/"
        )

        robot_setting_path = os.path.join(
            ROOT_ROBOT_PATH, self.robot_config_name + ".model.yaml"
        )
        self.read_setting_files(robot_setting_path, action_space_path)
        self.setup_action_space()
        self.setup_reward_calculator()

        self.observation_collector = ObservationCollector(
            ns, self._num_laser_beams, self._laser_range
        )

        # for time controlling in train mode
        self._action_frequency = 1 / rospy.get_param("/robot_action_rate")

        if self._is_train_mode:
            # w/o action publisher node
            self._action_pub = rospy.Publisher(
                f"{self._ns_robot}cmd_vel", Twist, queue_size=1
            )
            # step world to fast forward simulation time
            self._service_name_step = f"{self._ns}step_world"
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld
            )
        else:
            # w/ action publisher node
            # (controls action rate being published on '../cmd_vel')
            self._action_pub = rospy.Publisher(
                f"{self._ns_robot}cmd_vel_pub", Twist, queue_size=1
            )

    def setup_agent(self) -> None:
        """Loads the trained policy and when required the VecNormalize object."""
        model_file = os.path.join(MODELS_DIR, self.name, "best_model.zip")
        vecnorm_file = os.path.join(MODELS_DIR, self.name, "vec_normalize.pkl")
        model_params_file = os.path.join(
            MODELS_DIR, self.name, "hyperparameters.json"
        )

        assert os.path.isfile(
            model_file
        ), f"Compressed model cannot be found at {model_file}!"
        assert os.path.isfile(
            vecnorm_file
        ), f"VecNormalize file cannot be found at {vecnorm_file}!"
        assert os.path.isfile(
            model_params_file
        ), f"Hyperparameter file cannot be found at {vecnorm_file}!"

        with open(vecnorm_file, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        with open(model_params_file, "r") as file:
            hyperparams = json.load(file)

        self._agent = PPO.load(model_file).policy
        self._obs_norm_func = vec_normalize.normalize_obs
        self._agent_params = hyperparams

        self._get_robot_name_from_params()

    def run(self) -> None:
        """Loop for running the agent until ROS is shutdown.
        
        Note:
            Calls the 'step_world'-service for fast-forwarding the \
            simulation time in training mode. The simulation is forwarded \
            by action_frequency seconds. Otherwise, communicates with \
            the ActionPublisher node in order to comply with the specified \
            action publishing rate.
        """
        while not rospy.is_shutdown():
            if self._is_train_mode:
                self.call_service_takeSimStep(self._action_frequency)
            else:
                self._wait_for_next_action_cycle()
            obs = self.get_observations()[0]
            action = self.get_action(obs)
            self.publish_action(action)

    def _get_disc_action(self, action: int) -> np.ndarray:
        """Returns defined velocity commands for parsed action index.\
            (Discrete action space)

        Args:
            action (int): Index of the desired action.

        Returns:
            np.ndarray: Velocity commands corresponding to the index.
        """
        return super()._get_disc_action(action)

    def _wait_for_next_action_cycle(self) -> None:
        """Stops the loop until a trigger message is sent by the ActionPublisher

        Note:
            Only use this method in combination with the ActionPublisher node!
            That node is only booted when training mode is off.
        """
        try:
            rospy.wait_for_message(f"{self._ns_robot}next_cycle", Bool)
        except ROSException:
            pass

    def call_service_takeSimStep(self, t: float = None) -> None:
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


def main(agent_name: str) -> None:
    AGENT = DeploymentDRLAgent(agent_name=agent_name, ns=NS_PREFIX)

    try:
        AGENT.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    AGENT_NAME = sys.argv[1]
    main(agent_name=AGENT_NAME)
