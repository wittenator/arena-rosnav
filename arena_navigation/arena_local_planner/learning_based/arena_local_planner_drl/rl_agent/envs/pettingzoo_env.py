from pettingzoo import *
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AECIterable

from task_generator.tasks import *

def env():
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = FlatlandPettingZooEnv()
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class FlatlandPettingZooEnv(AECEnv):
    '''
    The AECEnv steps agents one at a time. If you are unsure if you
    have implemented a AECEnv correctly, try running the `api_test` documented in
    the Developer documentation on the website.
    '''
    def __init__(self) -> None:
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.possible_agents = ["robot_" + str(r) for r in range(2)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # ROBOT CONFIGS
        # action space

        # observation space

        # task manager + robot managers

    def step(self, action) -> None:
        '''
        receives a dictionary of actions keyed by the agent name.
        Returns the observation dictionary, reward dictionary, done dictionary, and info dictionary,
        where each dictionary is keyed by the agent.
        '''
        raise NotImplementedError

    def reset(self) -> None:
        '''
        resets the environment and returns a dictionary of observations (keyed by the agent name)
        '''
        raise NotImplementedError

    def seed(self, seed=None) -> None:
        '''
        Reseeds the environment (making the resulting environment deterministic).
        `reset()` must be called after `seed()`, and before `step()`.
        '''
        pass

    def observe(self, agent) -> None:
        '''
        Returns the observation an agent currently can make. `last()` calls this function.
        '''
        raise NotImplementedError

    def render(self, mode='human') -> None:
        '''
        Displays a rendered frame from the environment, if supported.
        Alternate render modes in the default environments are `'rgb_array'`
        which returns a numpy array and is supported by all environments outside of classic,
        and `'ansi'` which returns the strings printed (specific to classic environments).
        '''
        raise NotImplementedError

    def state(self) -> None:
        '''
        State returns a global view of the environment appropriate for
        centralized training decentralized execution methods like QMIX
        '''
        raise NotImplementedError('state() method has not been implemented in the environment {}.'.format(self.metadata.get('name', self.__class__.__name__)))

    def close(self) -> None:
        '''
        Closes the rendering window, subprocesses, network connections, or any other resources
        that should be released.
        '''
        pass

    def _dones_step_first(self):
        '''
        Makes .agent_selection point to first done agent. Stores old value of agent_selection
        so that _was_done_step can restore the variable after the done agent steps.
        '''
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        return self.agent_selection

    def agent_iter(self, max_iter=2**63):
        '''
        yields the current agent (self.agent_selection) when used in a loop where you step() each iteration.
        '''
        return AECIterable(self, max_iter)

    def last(self, observe=True):
        '''
        returns observation, cumulative reward, done, info   for the current agent (specified by self.agent_selection)
        '''
        agent = self.agent_selection
        observation = self.observe(agent) if observe else None
        return observation, self._cumulative_rewards[agent], self.dones[agent], self.infos[agent]

    def _was_done_step(self, action):
        '''
        Helper function that performs step() for done agents.

        Does the following:

        1. Removes done agent from .agents, .dones, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is done, loads that one, otherwise load next live agent
        3. Clear the rewards dict

        Highly recomended to use at the beginning of step as follows:

        def step(self, action):
            if self.dones[self.agent_selection]:
                self._was_done_step()
                return
            # main contents of step
        '''
        if action is not None:
            raise ValueError("when an agent is done, the only valid action is None")

        # removes done agent
        agent = self.agent_selection
        assert self.dones[agent], "an agent that was not done as attemted to be removed"
        del self.dones[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next done agent or loads next live agent (Stored in _skip_agent_selection)
        _dones_order = [agent for agent in self.agents if self.dones[agent]]
        if _dones_order:
            if getattr(self, '_skip_agent_selection', None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _dones_order[0]
        else:
            if getattr(self, '_skip_agent_selection', None) is not None:
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()





