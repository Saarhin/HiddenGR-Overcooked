from abc import abstractmethod
import pickle
from typing import Any, Collection, List, NoReturn, overload
from gym.core import Env
import numpy as np
import random
import datetime

from random import Random

from pddlgym.core import InvalidAction
from pddlgym.structs import Literal

from ml.common import GOAL_REWARD
from ml.common_functions import check_for_partial_goals
from ml.utilsRL import solve_fset

from tqdm import tqdm
from math import log2, exp
from queue import PriorityQueue


# This are for typing (we may want to move this elsewhere)
State = Any


def softmax(values: List[float]) -> List[float]:
    """Computes softmax probabilities for an array of values
    TODO We should probably use numpy arrays here
    Args:
        values (np.array): Input values for which to compute softmax

    Returns:
        np.array: softmax probabilities
    """
    return [(exp(q)) / sum([exp(_q) for _q in values]) for q in values]


class RLAgent:
    """
    This is a base class used as parent class for any
    RL agent. This is currently not much in use, but is
    recommended as development goes on.
    """

    def __init__(self,
                 env: Env,
                 problem: int = None,
                 episodes: int = 100,
                 decaying_eps: bool = True,
                 eps: float = 0.9,
                 alpha: float = 0.01,
                 decay: float = 0.00005,
                 gamma: float = 0.99,
                 action_list: Collection[Literal] = None,
                 rand: Random = Random(),
                 is_optimistic_initialization: bool = True):
        self.env = env
        self.problem = problem
        self.episodes = episodes
        self.decaying_eps = decaying_eps
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.gamma = gamma
        self.action_list = action_list
        self._random = rand
        if problem:
            self.env.fix_problem_index(problem)
        self.is_optimistic_initialization = is_optimistic_initialization

    @abstractmethod
    def agent_start(self, state: State) -> Any:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, state: State) -> Any:
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        pass

    @abstractmethod
    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    @abstractmethod
    def policy(self, state: State) -> Any:
        """The action for the specified state under the currently learned policy
           (unlike agent_step, this does not update the policy using state as a sample.
           Args:
                state (Any): the state observation from the environment
           Returns:
                The action prescribed for that state
        """
        pass

    @abstractmethod
    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        pass

    @abstractmethod
    def learn(self, init_threshold: int = 20):
        pass

    def __getitem__(self, state: State) -> Any:
        """[summary]

        Args:
            state (Any): The state for which we want to get the policy

        Raises:
            InvalidAction: [description]

        Returns:
            Any: [description]
        """""
        return self.softmax_policy(state)


def print_q_values(q_values: Collection[int], actions: Collection[Literal]):
    values = ""
    for i, q in enumerate(q_values):
        values += f"{actions[i]}: {q}\n"
    return values


class TabularQLearner(RLAgent):
    """
    A modified Tabular Q-Learning agent that works with SimAgents.
    Incorporates agent-specific information in state representation.
    """

    def __init__(self,
                 env: Env,
                 init_obs: Any,
                 problem: int = None,
                 episodes: int = 500,
                 decaying_eps: bool = True,
                 eps: float = 1.0,
                 alpha: float = 0.5,
                 decay: float = 0.000002,
                 gamma: float = 0.9,
                 action_list: Collection[Literal] = None,
                 rand: Random = Random(),
                 check_partial_goals: bool = True,
                 valid_only: bool = False,
                 is_optimistic_initialization: bool = True,
                 target_agent_name: str = None,
                 **kwargs):
        super().__init__(env, problem=problem, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha,
                         decay=decay, gamma=gamma, action_list=action_list, rand=rand,
                         is_optimistic_initialization=is_optimistic_initialization)
        
        # Store the target agent name for agent-specific learning
        self.target_agent_name = target_agent_name
        self.valid_only = valid_only
        
        # Define action space for Overcooked environment
        if not action_list:
            # Default action space for Overcooked
            self.action_list = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        else:
            self.action_list = action_list
            
        self.actions = len(self.action_list)
        self.check_partial_goals = check_partial_goals
        self.goal_literals_achieved = set()

        self.q_table = {}

        # hyperparameters
        self.episodes = episodes
        self.gamma = gamma
        self.decay = decay
        self.c_eps = eps
        self.base_eps = eps
        self.patience = 400000
        if decaying_eps:
            def epsilon():
                self.c_eps = max((self.episodes - self.step) / self.episodes, 0.01)
                return self.c_eps

            self.eps = epsilon
        else:
            self.eps = lambda: eps
        self.decaying_eps = decaying_eps
        self.alpha = alpha
        self.last_state = None
        self.last_action = None

    def states_in_q(self) -> List:
        """Returns the states stored in the q_values table

        Returns:
            List: The states for which we have a mapping in the q-table
        """
        return self.q_table.keys()

    def policy(self, state: State) -> Any:
        """Returns the greedy deterministic policy for the specified state

        Args:
            state (State): the state for which we want the action

        Raises:
            InvalidAction: Not sure about this one

        Returns:
            Any: The greedy action learned for state
        """
        return self.best_action(state)

    def epsilon_greedy_policy(self, state: State) -> Any:
        eps = self.eps()
        if self._random.random() <= eps:
            action = self._random.randint(0, self.actions - 1)
        else:
            action = self.policy(state)
        return action

    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        if state not in self.q_table:
            self.add_new_state(state)
            # If we query a state we have not visited, return a uniform distribution
            # return softmax([0]*self.actions)
        return softmax(self.q_table[state])

    def save_q_table(self, path: str):
        # sadly, this does not work, because the state we are using
        # is a frozenset of literals, which are not serializable.
        # a way to fix this is to use array states built using
        # common_functions.build_state

        with open(path, 'w') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path: str):
        with open(path, 'r') as f:
            table = pickle.load(path)
            self.q_table = table

    def add_new_state(self, state: State):
        # self.q_table[state] = [1. for _ in range(self.actions)]
        self.q_table[state] = [0.] * self.actions

    def get_all_q_values(self, state: State) -> List[float]:
        if state in self.q_table:
            return self.q_table[state]
        else:
            return [0.] * self.actions

    def best_action(self, state: State) -> int:
        if state not in self.q_table:
            self.add_new_state(state)
            # self.q_table[state] = [0 for _ in range(self.actions)]
        return np.argmax(self.q_table[state])

    def get_max_q(self, state: State) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return np.max(self.q_table[state])

    def set_q_value(self, state: State, action: Any, q_value: float):
        if state not in self.q_table:
            self.add_new_state(state)
        self.q_table[state][action] = q_value

    def get_q_value(self, state: State, action: Any) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return self.q_table[state][action]

    def agent_start(self, state: State) -> int:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        self.last_state = state
        self.last_action = self.policy(state)
        return self.last_action

    def agent_step(self, reward: float, state: State) -> int:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma * max_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        action = self.epsilon_greedy_policy(state)
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = - old_q

        new_q = old_q + self.alpha * (reward + td_error)
        self.set_q_value(self.last_state, self.last_action, new_q)
       
    def get_agent_specific_state(self, env_state):
        """
        Create an agent-specific state representation to encourage role specialization.
        
        Args:
            env_state: The environment state object
            
        Returns:
            A state representation that includes agent-specific information
        """
        # If no target agent is specified, use the global state
        if not self.target_agent_name:
            return env_state.get_repr()
            
        # Find the target agent
        try:
            target_agent = next((a for a in env_state.sim_agents if a.name == self.target_agent_name), None)
            
            if target_agent:
                # Create agent-specific state tuple
                agent_state = (
                    self.target_agent_name,  # Include agent identity
                    target_agent.location,
                    tuple((obj_name, obj[0].location) if obj else (obj_name, None) 
                          for obj_name, obj in env_state.world.objects.items()),
                    target_agent.get_holding()
                )
               
                return agent_state
        except Exception as e:
            print(f"Error creating agent-specific state: {e}")
            
        # Fallback to global state
        return env_state.get_repr()
    
    def print_q_table(self, max_entries: int = 10) -> None:
        """
        Print the current Q-table in a readable format, limiting to max_entries
        to avoid overwhelming output.

        Args:
            max_entries: Maximum number of state entries to print
        """
        print("\n=== Q-Table Contents ===")
        print(f"Total states in Q-table: {len(self.q_table)}")

        if not self.q_table:
            print("Q-table is empty.")
            return

        # Get a sample of states to print
        states_to_print = list(self.q_table.keys())[:max_entries]

        for i, state in enumerate(states_to_print):
            print(f"\nState {i+1}: {state}")
            q_values = self.q_table[state]

            for action_idx, q_value in enumerate(q_values):
                action = self.action_list[action_idx] if action_idx < len(self.action_list) else f"Action {action_idx}"
                print(f"  {action}: {q_value:.2f}")

        if len(self.q_table) > max_entries:
            print(f"\n... and {len(self.q_table) - max_entries} more states")

        print("========================\n")

        
    def constant_optimistic_initialization(self, init: Any) -> None:
        """
        Initialize the Q-table with a constant optimistic value of 100 for all actions.
        This simple approach encourages exploration by making all actions seem promising initially.

        Args:
            init: The initial state from the environment
        """
        print(f'Initializing Q-table with constant optimistic value of 100 for agent {self.target_agent_name or "unknown"}')

        # Reset the environment to get the initial state
        obs = self.env.reset()

        # Get the initial state representation
        state = self.get_agent_specific_state(obs)

        # Initialize the initial state with optimistic values
        for action_idx in range(self.actions):
            self.set_q_value(state, action_idx, 100.0)

        print(f'Initialized initial state with optimistic values')

        # Print the initialized Q-table
        self.print_q_table(max_entries=1)

#     def learn(self, init_threshold: int = 20):
#         """
#         Learn the Q-function using environment interactions.
#         Modified to use agent-specific state representations.
#         """
#         tsteps = 50
#         done_times = 0
#         patience = 0
#         converged_at = None
#         max_r = float("-inf")

#         # Get the environment object
#         init = self.env.reset()
        
#         self.constant_optimistic_initialization(init)

#         print(f'LEARNING FOR GOAL: {getattr(init, "goal", "Unknown")}')
#         print(f'Using {self.__class__.__name__} for agent {self.target_agent_name or "unknown"}')

#         # Define action vectors mapping
#         action_vectors = {
#             'Stay': (0, 0),
#             'Up': (0, -1),
#             'Down': (0, 1),
#             'Left': (-1, 0),
#             'Right': (1, 0),
#             'Interact': (0, 0)
#         }

#         tq = tqdm(range(self.episodes),
#                   postfix=f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")

#         for n in tq:
#             self.step = n
#             episode_r = 0

#             # Reset the environment for each episode
#             env_state = self.env.reset()
            
#             # Use agent-specific state representation
#             state = self.get_agent_specific_state(env_state)

#             action = self.agent_start(state)
#             done = False
#             tstep = 0

#             while tstep < tsteps and not done:
#                 try:
#                     # Get the action name or tuple from the action list
#                     action_value = self.action_list[action]

#                     # Create a dictionary mapping agent names to their actions
#                     action_dict = {}
                    
#                     # For training, only the target agent follows policy; others take random actions
#                     for agent in env_state.sim_agents:
#                         if self.target_agent_name and agent.name == self.target_agent_name:
#                             # This agent follows the policy
#                             if action_value in action_vectors:
#                                 action_dict[agent.name] = action_vectors[action_value]
#                             elif isinstance(action_value, tuple):
#                                 action_dict[agent.name] = action_value
#                             else:
#                                 action_dict[agent.name] = action_vectors['Stay']
#                         else:
#                             # Other agents take random actions during training
#                             random_action = random.choice(list(action_vectors.values()))
#                             action_dict[agent.name] = random_action

#                     # Pass the actions to the environment
#                     obs, reward, done, _ = self.env.step(action_dict)

#                     # Get the new agent-specific state representation
#                     next_state = self.get_agent_specific_state(obs)

#                     if done:
#                         reward = 100.
#                 except Exception as e:
#                     print(f"Error during step: {e}")
#                     next_state = state
#                     reward = -1.
#                     done = False

#                 if done:
#                     done_times += 1

#                 action = self.agent_step(reward, next_state)
#                 tstep += 1
#                 episode_r += reward

#             if done:  # One last update at the terminal state
#                 self.agent_end(reward)

#             if episode_r > max_r:
#                 max_r = episode_r
#                 tq.set_postfix_str(
#                     f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
                
#             if (n + 1) % 100 == 0:
#                 tq.set_postfix_str(
#                     f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
                
#             if (n + 1) % 1000 == 0:
#                 tq.set_postfix_str(
#                     f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
                
#                 if done_times <= 10:
#                     patience += 1
#                     if patience >= self.patience:
#                         print(f"Did not find goal after {n} episodes. Retrying.")
#                         raise InvalidAction("Did not learn")
#                 else:
#                     patience = 0
                    
#                 if done_times == 1000 and converged_at is None:
#                     converged_at = n
#                     print(f"***Policy converged to goal at {converged_at}***")
                
#                 done_times = 0
                
#             self.goal_literals_achieved.clear()
            
    def learn(self, init_threshold: int = 20):
        """
        Modified learning process that encourages collaborative behavior.
        Uses a phased approach with all agents following policies from the start.
        """
        tsteps = 50
        done_times = 0
        patience = 0
        converged_at = None
        max_r = float("-inf")

        # Get the environment object
        init = self.env.reset()

        self.constant_optimistic_initialization(init)

        print(f'LEARNING FOR GOAL: {getattr(init, "goal", "Unknown")}')
        print(f'Using collaborative {self.__class__.__name__} for agent {self.target_agent_name or "unknown"}')

        # Define action vectors mapping
        action_vectors = {
            'Stay': (0, 0),
            'Up': (0, -1),
            'Down': (0, 1),
            'Left': (-1, 0),
            'Right': (1, 0),
            'Interact': (0, 0)
        }

        # Modified training phases - all agents follow policies with decreasing exploration:
        # 1. High exploration phase (30% of episodes) - 70% exploration
        # 2. Medium exploration phase (40% of episodes) - 50% exploration
        # 3. Low exploration phase (30% of episodes) - 30% exploration
        phase1_end = int(self.episodes * 0.3)
        phase2_end = int(self.episodes * 0.7)

        tq = tqdm(range(self.episodes),
                  postfix=f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")

        for n in tq:
            self.step = n
            episode_r = 0

            # Determine current training phase and exploration rate
            if n < phase1_end:
                current_phase = "high_exploration"
                exploration_rate = 0.7  # 70% exploration in early phase
            elif n < phase2_end:
                current_phase = "medium_exploration"
                exploration_rate = 0.5  # 50% exploration in middle phase
            else:
                current_phase = "low_exploration"
                exploration_rate = 0.3  # 30% exploration in final phase

            # Reset the environment for each episode
            env_state = self.env.reset()

            # Use agent-specific state representation
            state = self.get_agent_specific_state(env_state)

            action = self.agent_start(state)
            done = False
            tstep = 0

            # Get all agent names from the environment
            all_agent_names = [agent.name for agent in env_state.sim_agents]

            while tstep < tsteps and not done:
                try:
                    # Get the action name or tuple from the action list
                    action_value = self.action_list[action]

                    # Create a dictionary mapping agent names to their actions
                    action_dict = {}

                    # Determine actions for all agents
                    for agent in env_state.sim_agents:
                        if agent.name == self.target_agent_name:
                            # This agent follows its learning policy
                            if action_value in action_vectors:
                                action_dict[agent.name] = action_vectors[action_value]
                            elif isinstance(action_value, tuple):
                                action_dict[agent.name] = action_value
                            else:
                                action_dict[agent.name] = action_vectors['Stay']
                        else:
                            # Other agents now follow policies with exploration in all phases
                            other_policy = self.get_other_agent_policy(agent.name)
                            if other_policy and random.random() > exploration_rate:
                                # Use other agent's policy
                                try:
                                    other_state = self.create_state_for_other_agent(agent, env_state)
                                    other_action_idx = other_policy.policy(other_state)
                                    other_action = other_policy.action_list[other_action_idx]
                                    action_dict[agent.name] = other_action
                                except Exception as e:
                                    # Fallback to random if policy lookup fails
                                    print(f"Policy lookup failed for {agent.name}: {e}")
                                    action_dict[agent.name] = random.choice(list(action_vectors.values()))
                            else:
                                # Random exploration based on current phase rate
                                action_dict[agent.name] = random.choice(list(action_vectors.values()))

                    # Pass the actions to the environment
                    obs, reward, done, _ = self.env.step(action_dict)

                    # Get the new agent-specific state representation
                    next_state = self.get_agent_specific_state(obs)

                    # Add collaborative reward component
                    collaborative_reward = self.calculate_collaborative_reward(obs, action_dict)
                    total_reward = reward + collaborative_reward

                    if done:
                        total_reward = 100.0 + collaborative_reward  # Bonus for completing the task
                except Exception as e:
                    print(f"Error during step: {e}")
                    next_state = state
                    total_reward = -1.0
                    done = False

                if done:
                    done_times += 1

                action = self.agent_step(total_reward, next_state)
                tstep += 1
                episode_r += total_reward

            if done:  # One last update at the terminal state
                self.agent_end(total_reward)

            if episode_r > max_r:
                max_r = episode_r
                tq.set_postfix_str(
                    f"Phase: {current_phase}. States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")

            if (n + 1) % 100 == 0:
                tq.set_postfix_str(
                    f"Phase: {current_phase}. States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")

            if (n + 1) % 1000 == 0:
                tq.set_postfix_str(
                    f"Phase: {current_phase}. States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")

                if done_times <= 10:
                    patience += 1
                    if patience >= self.patience:
                        print(f"Did not find goal after {n} episodes. Retrying.")
                        raise InvalidAction("Did not learn")
                else:
                    patience = 0

                if done_times == 1000 and converged_at is None:
                    converged_at = n
                    print(f"***Policy converged to goal at {converged_at}***")

                done_times = 0

            self.goal_literals_achieved.clear()

    def get_other_agent_policy(self, agent_name):
        """
        Get the policy for another agent.

        Args:
            agent_name: Name of the other agent

        Returns:
            The policy object for the other agent, or None if not found
        """
        # This is a stub method that needs to be implemented based on how 
        # you store policies for different agents in your system.
        # Typically this would look up the policy from a shared registry:

        # Example implementation:
        # return policy_registry.get_policy(agent_name, self.problem)

        # For testing, return None
        return None

    def create_state_for_other_agent(self, agent, env_state):
        """
        Create a state representation for another agent.

        Args:
            agent: The agent object
            env_state: The environment state

        Returns:
            A state representation appropriate for the other agent's policy
        """
        # Similar to get_agent_specific_state but for a different agent
        try:
            agent_state = (
                agent.name,  # Include agent identity
                agent.location,
                tuple((obj_name, obj[0].location) if obj else (obj_name, None) 
                      for obj_name, obj in env_state.world.objects.items()),
                agent.get_holding()
            )
            return agent_state
        except Exception as e:
            print(f"Error creating state for other agent: {e}")
            # Fallback to global state
            return env_state.get_repr()

    def calculate_collaborative_reward(self, obs, action_dict):
        """
        Calculate additional reward based on collaborative behavior for pre-prepared meal delivery.
        Food is already chopped and plated - agents just need to pick up and deliver.
        Now with debug print statements to track reward components.

        Args:
            obs: Current observation (environment state)
            action_dict: Dictionary of actions taken by all agents

        Returns:
            A reward bonus for collaborative behavior
        """
        collaborative_reward = 0.0
        reward_components = {}  # Dictionary to track individual reward components

        try:
            # Get all agents
            agents = obs.sim_agents
            if len(agents) < 2:
                print("No collaboration reward: fewer than 2 agents")
                return 0.0  # No collaboration possible with a single agent

            # Find our target agent
            target_agent = next((a for a in agents if a.name == self.target_agent_name), None)
            if not target_agent:
                print(f"No collaboration reward: target agent '{self.target_agent_name}' not found")
                return 0.0

            # 1. Reward for role division (pickup vs. delivery)
            food_pickup_locations = self._get_food_pickup_locations(obs)
            delivery_locations = self._get_delivery_locations(obs)

            # Track which agents are near pickup points and which are near delivery points
            pickup_agents = []
            delivery_agents = []

            for agent in agents:
                # Check if agent is near food pickup
                near_pickup = any(self._manhattan_distance(agent.location, loc) <= 2 
                                  for loc in food_pickup_locations)

                # Check if agent is near delivery
                near_delivery = any(self._manhattan_distance(agent.location, loc) <= 2 
                                   for loc in delivery_locations)

                if near_pickup:
                    pickup_agents.append(agent)
                if near_delivery:
                    delivery_agents.append(agent)

            # Reward if agents are divided between pickup and delivery
            role_division_reward = 0.0
            if pickup_agents and delivery_agents and len(set(pickup_agents) & set(delivery_agents)) == 0:
                # Agents are specializing in different areas
                role_division_reward = 0.5
                collaborative_reward += role_division_reward
                print(f"✓ REWARD: +{role_division_reward} for role division (pickup vs delivery)")
                print(f"  - Pickup agents: {[a.name for a in pickup_agents]}")
                print(f"  - Delivery agents: {[a.name for a in delivery_agents]}")
            else:
                print(f"✗ NO REWARD for role division - agents not properly divided")
                if not pickup_agents:
                    print("  - No agents near pickup locations")
                if not delivery_agents:
                    print("  - No agents near delivery locations")
                if len(set(pickup_agents) & set(delivery_agents)) > 0:
                    print(f"  - Some agents are in both pickup and delivery areas: {[a.name for a in set(pickup_agents) & set(delivery_agents)]}")

            reward_components['role_division'] = role_division_reward

            # 2. Reward for efficient task handling
            # Check if agents are carrying food items
            agents_with_food = []
            agents_without_food = []

            for agent in agents:
                held_item = agent.get_holding()
                if held_item and self._is_food_item(held_item):
                    agents_with_food.append(agent)
                else:
                    agents_without_food.append(agent)

            # Reward if some agents are carrying food while others are free to pick up more
            task_handling_reward = 0.0
            if agents_with_food and agents_without_food:
                task_handling_reward = 0.3
                collaborative_reward += task_handling_reward
                print(f"✓ REWARD: +{task_handling_reward} for efficient task handling")
                print(f"  - Agents with food: {[a.name for a in agents_with_food]}")
                print(f"  - Agents without food: {[a.name for a in agents_without_food]}")
            else:
                print(f"✗ NO REWARD for task handling - need some agents with food and some without")
                if not agents_with_food:
                    print("  - No agents carrying food")
                if not agents_without_food:
                    print("  - All agents are carrying food")

            reward_components['task_handling'] = task_handling_reward

            # 3. Reward for progress toward delivery
            # Check if agents with food are moving toward delivery points
            delivery_progress_reward = 0.0

            for agent in agents_with_food:
                # Find closest delivery location
                closest_delivery = None
                min_distance = float('inf')

                for loc in delivery_locations:
                    dist = self._manhattan_distance(agent.location, loc)
                    if dist < min_distance:
                        min_distance = dist
                        closest_delivery = loc

                # Reward for being close to delivery while holding food
                if closest_delivery:
                    # Scale reward based on proximity (closer = better)
                    proximity_reward = max(0, 1 - (min_distance / 10))  # Scale from 0 to 1
                    agent_reward = 0.5 * proximity_reward  # Increased from 0.2 to 0.5
                    delivery_progress_reward += agent_reward
                    collaborative_reward += agent_reward
                    print(f"✓ REWARD: +{agent_reward:.2f} for {agent.name} progress toward delivery")
                    print(f"  - Distance to delivery: {min_distance}, proximity factor: {proximity_reward:.2f}")
                else:
                    print(f"✗ NO REWARD for {agent.name} delivery progress - no delivery location found")

            reward_components['delivery_progress'] = delivery_progress_reward

            # 4. Reward for strategic positioning
            # If an agent without food is near pickup while an agent with food is near delivery
            strategic_position_reward = 0.0
            if any(any(self._manhattan_distance(agent.location, loc) <= 2 for loc in food_pickup_locations)
                   for agent in agents_without_food) and \
               any(any(self._manhattan_distance(agent.location, loc) <= 2 for loc in delivery_locations)
                   for agent in agents_with_food):
                strategic_position_reward = 0.4
                collaborative_reward += strategic_position_reward
                print(f"✓ REWARD: +{strategic_position_reward} for strategic team positioning")
                print("  - Agents without food are near pickup while agents with food are near delivery")
            else:
                print(f"✗ NO REWARD for strategic positioning")
                empty_near_pickup = any(any(self._manhattan_distance(agent.location, loc) <= 2 
                                          for loc in food_pickup_locations)
                                      for agent in agents_without_food)
                full_near_delivery = any(any(self._manhattan_distance(agent.location, loc) <= 2 
                                           for loc in delivery_locations)
                                       for agent in agents_with_food)

                if not empty_near_pickup:
                    print("  - No empty agents near pickup")
                if not full_near_delivery:
                    print("  - No agents with food near delivery")

            reward_components['strategic_position'] = strategic_position_reward

            # 5. Immediate handoff reward
            handoff_reward = 0.0
            for agent in agents:
                if agent.name == self.target_agent_name:
                    continue

                # Check if items have been "handed off" between agents
                handoff_detected = self._detect_handoff(obs, agent, target_agent)
                if handoff_detected:
                    handoff_reward += 0.5
                    collaborative_reward += 0.5
                    print(f"✓ REWARD: +0.5 for handoff between {agent.name} and {target_agent.name}")
                    break

            if handoff_reward == 0:
                print(f"✗ NO REWARD for handoffs - no valid handoffs detected")

            reward_components['handoff'] = handoff_reward

            # Print summary
            print(f"TOTAL COLLABORATIVE REWARD: {collaborative_reward:.2f}")
            print(f"Reward components: {reward_components}")

        except Exception as e:
            print(f"Error calculating collaborative reward: {e}")

        return collaborative_reward

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_food_pickup_locations(self, obs):
        """
        Extract locations where prepared food can be picked up
        """
        try:
            pickup_locs = []
            # Find objects representing prepared food locations
            for obj_name, obj in obs.world.objects.items():
                if (("sushi" in obj_name.lower() or "meal" in obj_name.lower() or 
                    "plated" in obj_name.lower() or "food" in obj_name.lower()) and 
                    obj and obj[0].location):
                    pickup_locs.append(obj[0].location)

            print('Pickup Locs:', pickup_locs)
            
            # If we found pickup locations, return them
            if pickup_locs:
                return pickup_locs

            # Fallback: look for counters or tables that might have food
            for obj_name, obj in obs.world.objects.items():
                if (("counter" in obj_name.lower() or "table" in obj_name.lower()) and 
                    obj and obj[0].location):
                    pickup_locs.append(obj[0].location)

            return pickup_locs
        except:
            # Fallback to some default locations if environment structure is different
            return [(2, 2), (3, 2), (4, 2)]  # Example default locations

    def _get_delivery_locations(self, obs):
        """Extract delivery locations from the environment state"""
        try:
            delivery_locs = []
            for obj_name, obj in obs.world.objects.items():
                if (("delivery" in obj_name.lower() or "serve" in obj_name.lower() or
                     "service" in obj_name.lower()) and obj and obj[0].location):
                    delivery_locs.append(obj[0].location)

            print('Delivery Locs:', delivery_locs)
            
            if delivery_locs:
                return delivery_locs
            else:
                # Fallback to some default locations if environment structure is different
                return [(8, 8), (9, 8)]  # Example default locations
        except:
            # Fallback to some default locations if environment structure is different
            return [(8, 8), (9, 8)]  # Example default locations

    def _is_food_item(self, item):
        """Check if the held item is a food item"""
        try:
            # Try to determine if the item is food from its name/type
            item_str = str(item).lower()
            return ("sushi" in item_str or "meal" in item_str or 
                    "plated" in item_str or "food" in item_str or 
                    "water" in item_str)
        except:
            # If we can't determine the type, assume it might be food
            return True

    def _detect_handoff(self, obs, agent1, agent2):
        """
        Detect if an item has been handed off between two agents
        For coordinate-based action environment
        """
        try:
            # Check if agents are adjacent
            distance = self._manhattan_distance(agent1.location, agent2.location)
            if distance != 1:  # Not adjacent
                return False

            # Check if one agent has food and the other doesn't
            agent1_has_food = agent1.get_holding() and self._is_food_item(agent1.get_holding())
            agent2_has_food = agent2.get_holding() and self._is_food_item(agent2.get_holding())

            # Simplest handoff detection: agents are adjacent and one has food while the other doesn't
            if (agent1_has_food and not agent2_has_food) or (not agent1_has_food and agent2_has_food):
                return True

            return False
        except:
            return False