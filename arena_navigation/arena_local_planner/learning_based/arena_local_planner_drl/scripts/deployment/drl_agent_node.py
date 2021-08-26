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


class DeploymentDRLAgent:
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

    def _get_robot_name_from_params(self):
        """Retrives the agent-specific robot name from the dictionary loaded\
            from respective 'hyperparameter.json'.    
        """
        assert self._agent_params and self._agent_params["robot"]
        self.robot_config_name = self._agent_params["robot"]

    def read_setting_files(
        self, robot_setting_yaml: str, action_space_yaml: str
    ) -> None:
        """Retrieves the robot radius (in 'self._robot_radius'), \
            laser scan range (in 'self._laser_range') and \
            the action space from respective yaml file.

        Args:
            robot_setting_yaml (str): 
                Yaml file containing the robot specific settings. 
            action_space_yaml (str): 
                Yaml file containing the action space configuration. 
        """
        with open(robot_setting_yaml, "r") as fd:
            robot_data = yaml.safe_load(fd)
            # get robot radius
            for body in robot_data["bodies"]:
                if body["name"] == "base_footprint":
                    for footprint in body["footprints"]:
                        if footprint["type"] == "circle":
                            self._robot_radius = (
                                footprint.setdefault("radius", 0.3) * 1.05
                            )
                        if footprint["radius"]:
                            self._robot_radius = footprint["radius"] * 1.05

            # get laser related information
            for plugin in robot_data["plugins"]:
                if plugin["type"] == "Laser":
                    laser_angle_min = plugin["angle"]["min"]
                    laser_angle_max = plugin["angle"]["max"]
                    laser_angle_increment = plugin["angle"]["increment"]
                    self._num_laser_beams = int(
                        round(
                            (laser_angle_max - laser_angle_min)
                            / laser_angle_increment
                        )
                        + 1
                    )
                    self._laser_range = plugin["range"]

        with open(action_space_yaml, "r") as fd:
            setting_data = yaml.safe_load(fd)

            self._discrete_actions = setting_data["robot"]["discrete_actions"]
            self._cont_actions = {
                "linear_range": setting_data["robot"]["continuous_actions"][
                    "linear_range"
                ],
                "angular_range": setting_data["robot"]["continuous_actions"][
                    "angular_range"
                ],
            }

    def setup_action_space(self) -> None:
        """Sets up the action space. (spaces.Box)"""
        assert self._discrete_actions or self._cont_actions
        assert (
            self._agent_params and "discrete_action_space" in self._agent_params
        )

        self._action_space = (
            spaces.Discrete(len(self._discrete_actions))
            if self._agent_params["discrete_action_space"]
            else spaces.Box(
                low=np.array(
                    [
                        self._cont_actions["linear_range"][0],
                        self._cont_actions["angular_range"][0],
                    ]
                ),
                high=np.array(
                    [
                        self._cont_actions["linear_range"][1],
                        self._cont_actions["angular_range"][1],
                    ]
                ),
                dtype=np.float,
            )
        )

    def setup_reward_calculator(self) -> None:
        """Sets up the reward calculator."""
        assert self._agent_params and "reward_fnc" in self._agent_params
        self.reward_calculator = RewardCalculator(
            robot_radius=self._robot_radius,
            safe_dist=1.6 * self._robot_radius,
            goal_radius=GOAL_RADIUS,
            rule=self._agent_params["reward_fnc"],
            extended_eval=False,
        )

    @property
    def action_space(self) -> spaces.Box:
        """Returns the DRL agent's action space.

        Returns:
            spaces.Box: Agent's action space
        """
        return self._action_space

    @property
    def observation_space(self) -> spaces.Box:
        """Returns the DRL agent's observation space.

        Returns:
            spaces.Box: Agent's observation space
        """
        return self.observation_collector.observation_space

    def get_observations(self) -> Tuple[np.ndarray, dict]:
        """Retrieves the latest synchronized observation.

        Returns:
            Tuple[np.ndarray, dict]: 
                Tuple, where first entry depicts the observation data concatenated \
                into one array. Second entry represents the observation dictionary.
        """
        merged_obs, obs_dict = self.observation_collector.get_observations()
        if self._agent_params["normalize"]:
            merged_obs = self._obs_norm_func(merged_obs)
        return merged_obs, obs_dict

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Infers an action based on the given observation.

        Args:
            obs (np.ndarray): Merged observation array.

        Returns:
            np.ndarray:
                Action in [linear velocity, angular velocity]
        """
        action = self._agent.predict(obs, deterministic=True)[0]
        if self._agent_params["discrete_action_space"]:
            action = self._get_disc_action(action)
        else:
            # clip action
            action = np.maximum(
                np.minimum(self._action_space.high, action),
                self._action_space.low,
            )
        return action

    def publish_action(self, action: np.ndarray) -> None:
        """Publishes an action on 'self._action_pub' (ROS topic).

        Args:
            action (np.ndarray):
                Action in [linear velocity, angular velocity]
        """
        action_msg = Twist()
        action_msg.linear.x = action[0]
        action_msg.angular.z = action[1]
        self._action_pub.publish(action_msg)

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
        return np.array(
            [
                self._discrete_actions[action]["linear"],
                self._discrete_actions[action]["angular"],
            ]
        )

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
