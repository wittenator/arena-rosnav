import numpy as np
import rospy
import rospkg
import os

rospy.set_param("/MARL", True)

from rl_agent.training_agent_wrapper import TrainingDRLAgent
from scripts.deployment.drl_agent_node import DeploymentDRLAgent
from rl_agent.envs.pettingzoo_env import FlatlandPettingZooEnv

from nav_msgs.srv import GetMap

DEFAULT_HYPERPARAMETER = os.path.join(
    rospkg.RosPack().get_path("arena_local_planner_drl"),
    "configs",
    "hyperparameters",
    "default.json",
)


def main():
    rospy.set_param("/MARL", True)
    rospy.init_node(f"USER_NODE", anonymous=True)

    agent1 = TrainingDRLAgent(
        ns="sim_1",
        robot_name="test1",
        hyperparameter_path=DEFAULT_HYPERPARAMETER,
    )
    agent2 = TrainingDRLAgent(
        ns="sim_1",
        robot_name="test2",
        hyperparameter_path=DEFAULT_HYPERPARAMETER,
    )
    agent3 = TrainingDRLAgent(
        ns="sim_1",
        robot_name="test3",
        hyperparameter_path=DEFAULT_HYPERPARAMETER,
    )
    agent4 = TrainingDRLAgent(
        ns="sim_1",
        robot_name="test4",
        hyperparameter_path=DEFAULT_HYPERPARAMETER,
    )

    agent_list = [agent1, agent2, agent3, agent4]

    env = FlatlandPettingZooEnv(ns="sim_1", agent_list=agent_list)
    obs = env.reset()

    AGENT = DeploymentDRLAgent(
        agent_name="rule_04", ns="sim_1", robot_name="test1"
    )

    agent_names = env.agents
    for _ in range(100000000):
        actions = {agent: AGENT.get_action(obs[agent]) for agent in agent_names}
        obs, rewards, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
