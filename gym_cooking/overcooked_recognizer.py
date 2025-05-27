"""
Modified implementation of Overcooked environment goal recognition.
Removed dependency on RealAgents and Bayesian delegation.
"""

import numpy as np
import random
import json
import os
from collections import defaultdict, namedtuple
import gym

# Define AgentRepr at the module level to help with policy loading
AgentRepr = namedtuple("AgentRepr", "name location holding")

# Existing imports
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import SimAgent, COLORS
from utils.core import *

# Import the RL methods (using our modified version)
# This assumes the updated TabularQLearner.py has been implemented
from ml.rl import TabularQLearner
from ml.metrics import kl_divergence_norm_softmax, soft_divergence_point, trajectory_q_value


def load_recipes(level):
    """Load recipes from the level file"""
    recipes = []
    with open(f'utils/levels/{level}.txt', 'r') as f:
        phase = 1
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1
            elif phase == 2:
                recipes.append(globals()[line]())
    return recipes

class OvercookedRecognizer:
    """Goal recognizer for Overcooked environment using modified TabularQLearner"""
    
    def __init__(self, evaluation=kl_divergence_norm_softmax, method=TabularQLearner):
        """
        Initialize the Overcooked goal recognizer.
        
        Args:
            evaluation: Function to evaluate the divergence between trajectories and policies
            method: RL algorithm class to use for policy training
        """
        self.evaluation = evaluation
        self.method = method
        # Change: Store policies per agent role and recipe
        self.policies = {}  # Format: {agent_name: [policy_recipe1, policy_recipe2, ...]}
        self.recipe_list = []
        self.actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
        
    def __str__(self):
        return f"OvercookedRecognizer(evaluation={self.evaluation.__name__}, method={self.method.__name__})"
    
    def train_policies(self, env, recipe_list, arglist):
        """
        Train policies for each possible goal (recipe) and each agent role.
        
        Args:
            env: The Overcooked environment
            recipe_list: List of possible recipe goals
            arglist: Command line arguments
        """
        self.recipe_list = recipe_list
        
        # Determine the number of agents from the environment or arglist
        num_agents = arglist.num_agents
        
        # For each agent role
        for agent_idx in range(num_agents):
            agent_name = f"agent-{agent_idx+1}"
            # print(f"Training policies for {agent_name}")
            
            # Initialize policy list for this agent
            self.policies[agent_name] = []
            
            # For each recipe, train a specific policy for this agent
            for recipe_idx, recipe in enumerate(recipe_list):
                # print(f"Training policy for {agent_name} on recipe {recipe_idx}: {recipe}")
                
                # Reset environment
                obs = env.reset()
                
                # Create a TabularQLearner for this agent and recipe
                policy = self.method(
                    env=env, 
                    init_obs=obs, 
                    problem=recipe_idx,
                    action_list=self.actions,
                    episodes=1000,  # Can adjust as needed
                    target_agent_name=agent_name  # Add agent name for reference
                )
                
                # Train policy
                try:
                    # print(f"Training {agent_name} policy for recipe {recipe_idx}")
                    policy.learn(init_threshold=20)
                    # print(f"Successfully trained policy with {len(policy.q_table)} states in Q-table")
                    self.policies[agent_name].append(policy)
                    
                    # Save policy to disk
                    self._save_policy(policy, recipe_idx, recipe, agent_name)
                    
                except Exception as e:
                    # print(f"Error training policy for {agent_name} on recipe {recipe_idx}: {e}")
                    # Create a dummy policy
                    dummy_policy = self.method(env=env, init_obs=obs, problem=recipe_idx, action_list=self.actions)
                    self.policies[agent_name].append(dummy_policy)
    
    def _save_policy(self, policy, recipe_idx, recipe, agent_name):
        """Save the trained policy to disk"""
        os.makedirs("trained_policies", exist_ok=True)
        
        # Convert state keys to strings for JSON serialization
        serializable_q_table = {}
        for state, actions in policy.q_table.items():
            state_key = str(state)
            serializable_q_table[state_key] = actions
        
        policy_data = {
            "agent_name": agent_name,
            "recipe_id": recipe_idx,
            "recipe_name": str(recipe),
            "q_table": serializable_q_table
        }
        
        with open(f"trained_policies/sim_{agent_name}_recipe_{recipe_idx}.json", 'w') as f:
            json.dump(policy_data, f)
    
    def load_policies(self, env, recipe_list):
        """Load pre-trained policies for each agent from disk"""
        self.recipe_list = recipe_list
        
        # Reset the environment to initialize it
        obs = env.reset()
        
        # Determine the number of agents from the environment
        num_agents = len(obs.sim_agents)
        policies_loaded = 0
        
        for agent_idx in range(num_agents):
            agent_name = f"agent-{agent_idx+1}"
            self.policies[agent_name] = []
            
            for recipe_idx, recipe in enumerate(recipe_list):
                try:
                    policy_path = f"trained_policies/sim_{agent_name}_recipe_{recipe_idx}.json"
                    # print(f"Loading policy from {policy_path}")
                    
                    with open(policy_path, 'r') as f:
                        policy_data = json.load(f)
                    
                    # Create a new policy
                    policy = self.method(
                        env=env, 
                        init_obs=obs, 
                        problem=recipe_idx, 
                        action_list=self.actions,
                        target_agent_name=agent_name
                    )
                    
                    # Load Q-table from file
                    loaded_states = 0
                    for state_key, actions in policy_data["q_table"].items():
                        try:
                            # Convert string key back to tuple
                            locals_dict = {'AgentRepr': AgentRepr}
                            state = eval(state_key, globals(), locals_dict)
                            policy.q_table[state] = actions
                            loaded_states += 1
                        except Exception as e:
                            # print(f"Error parsing state {state_key[:50]}...: {str(e)[:100]}")
                            # Fall back to using the string key
                            policy.q_table[state_key] = actions
                    
                    self.policies[agent_name].append(policy)
                    policies_loaded += 1
                    # print(f"Loaded policy for {agent_name} on recipe {recipe_idx} with {loaded_states} states")
                    
                except Exception as e:
                    # print(f"Could not load policy for {agent_name} on recipe {recipe_idx}: {e}")
                    # Create a dummy policy as fallback
                    policy = self.method(
                        env=env, 
                        init_obs=obs, 
                        problem=recipe_idx, 
                        action_list=self.actions,
                        target_agent_name=agent_name
                    )
                    self.policies[agent_name].append(policy)
        
        return policies_loaded > 0
    
    def recognize_goal(self, trajectory, epsilon=0.1):
        """
        Recognize the goal from an observed trajectory.
        
        Args:
            trajectory: List of (state, action) pairs
            epsilon: Epsilon value for softmax
            
        Returns:
            Tuple of (predicted_goal_idx, all_goals_ranked)
        """
        if not self.policies:
            raise ValueError("No policies have been trained. Call train_policies first.")
        
        if not trajectory:
            # print("WARNING: Empty trajectory provided to recognize_goal")
            # Return a random prediction for empty trajectory
            return 0, [(i, 1.0 - (i*0.1)) for i in range(len(self.policies))]
        
        divergences = []
        
        for i, policy in enumerate(self.policies):
            try:
                # Calculate divergence between trajectory and policy
                divergence = self._calculate_divergence(trajectory, policy, epsilon)
                divergences.append((i, divergence))
            except Exception as e:
                # print(f"Error calculating divergence for policy {i}: {e}")
                # Add a default high divergence
                divergences.append((i, 1000.0))
        
        # Sort by lowest divergence
        divergences.sort(key=lambda x: x[1])
        
        # Return the goal with lowest divergence and the full ranking
        predicted_goal_idx = divergences[0][0]
        return predicted_goal_idx, divergences
    
    def _calculate_divergence(self, trajectory, policy, epsilon):
        """Calculate the divergence between a trajectory and a policy"""
        try:
            # Use the evaluation function provided in the constructor
            return self.evaluation(trajectory, policy.q_table, self.actions, epsilon=epsilon)
        except Exception as e:
            # print(f"Error in divergence calculation: {e}")
            # Return a high divergence as fallback
            return 1000.0
    
    def observe_trajectory(self, trajectory, obs_level):
        """
        Return a partially observed trajectory based on the observation level.
        
        Args:
            trajectory: Full trajectory
            obs_level: Observation level from 0.0 to 1.0
            
        Returns:
            Partially observed trajectory
        """
        if not trajectory:
            # print("WARNING: Empty trajectory provided to observe_trajectory")
            return []
            
        if obs_level >= 1.0:
            return trajectory
        
        # Calculate how many steps to observe
        num_steps = max(1, int(len(trajectory) * obs_level))
        
        # Randomly select steps to observe
        if num_steps >= len(trajectory):
            return trajectory
            
        observed_indices = sorted(random.sample(range(len(trajectory)), num_steps))
        
        # Return the observed steps
        return [trajectory[i] for i in observed_indices]


