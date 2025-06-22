"""
Experiment framework for Goal Recognition in Overcooked environment.
Modified to work with SimAgents directly, removing Bayesian delegation.
"""

import numpy as np
import random
import os
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt

import gym
from overcooked_recognizer import OvercookedRecognizer, load_recipes
from ml.metrics import kl_divergence_norm_softmax, soft_divergence_point, trajectory_q_value

def extract_trajectory(env, policies=None, target_agent_name=None, max_steps=100):
    """
    Extract state-action trajectory for a specific agent while ALL agents use their own policies.
    
    Args:
        env: Overcooked environment
        policies: Dictionary of {agent_name: [policy_recipe1, policy_recipe2, ...]}
        target_agent_name: Name of the agent to track (if None, uses first agent)
        max_steps: Maximum number of steps to run
        
    Returns:
        List of (state, action) pairs for the target agent
    """
    #print(f"Starting trajectory extraction for agent {target_agent_name or 'first agent'}")
    
    # Reset environment
    obs = env.reset()
    
    # Find the target agent
    sim_agents = obs.sim_agents
    if target_agent_name:
        target_agent = next((a for a in sim_agents if a.name == target_agent_name), None)
        if target_agent is None:
            #print(f"ERROR: Could not find agent '{target_agent_name}'. Using first agent.")
            target_agent = sim_agents[0] if sim_agents else None
    else:
        target_agent = sim_agents[0] if sim_agents else None
    
    if not target_agent:
        #print("ERROR: No agents found in environment")
        return []
    
    # Start extracting trajectory
    trajectory = []
    done = False
    steps = 0
    
    # Get the current recipe/goal
    current_recipe_idx = getattr(env, 'current_recipe_idx', 0)
    
    # Available actions
    actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
    
    try:
        while not done and steps < max_steps:
            try:
                # Get the global state representation (same as training)
                env_state_repr = obs.get_repr()
                
                # Create action dictionary - each agent will use its own policy
                action_dict = {}
                
                for agent in sim_agents:
                    # Default to random action
                    action_idx = random.randint(0, len(actions) - 1)
                    
                    # Try to use policy if available for this agent
                    if policies and agent.name in policies and len(policies[agent.name]) > current_recipe_idx:
                        # Get the appropriate policy for this agent and recipe
                        policy = policies[agent.name][current_recipe_idx]
                        
                        # Try to get action from policy
                        try:
                            if env_state_repr in policy.q_table:
                                action_idx = policy.policy(env_state_repr)
                            else:
                                # Create an agent-specific state representation
                                agent_state = (
                                    agent.name,  # Include agent identity in state
                                    agent.location,
                                    tuple((obj_name, obj[0].location) if obj else (obj_name, None) 
                                          for obj_name, obj in env.world.objects.items()),
                                    agent.get_holding()
                                )
                                
                                if agent_state in policy.q_table:
                                    action_idx = policy.policy(agent_state)
                                else:
                                    # Try string-based lookup as final fallback
                                    state_key = str(agent_state)
                                    if state_key in policy.q_table:
                                        action_idx = policy.policy(state_key)
                        except Exception as e:
                            print(f"Error in policy lookup for {agent.name}: {e}")
                            # Already defaulted to random action above
                    
                    action = actions[action_idx]
                    action_dict[agent.name] = action
                    
                    # For the target agent, record the state-action pair
                    if agent.name == target_agent.name:
                        trajectory.append((env_state_repr, action_idx))
                
                # Take action in environment
                obs, reward, done, _ = env.step(action_dict)
                
                # Update sim_agents reference
                sim_agents = obs.sim_agents
                target_agent = next((a for a in sim_agents if a.name == target_agent.name), None)
                
                steps += 1
                
            except Exception as e:
                print(f"Error in trajectory extraction at step {steps}: {e}")
                break
                
    except Exception as e:
        print(f"Fatal error in trajectory extraction: {e}")
    
    print(f"Extracted trajectory with {len(trajectory)} steps")
    return trajectory

class OvercookedGRExperiment:
    """Experiment framework for Goal Recognition in Overcooked with agent-specific policies"""
    
    def __init__(self, arglist, obs_levels=[0.1, 0.3, 0.5, 0.7, 1.0]):
        """
        Initialize the experiment.
        
        Args:
            arglist: Command line arguments
            obs_levels: Observation levels to test
        """
        self.arglist = arglist
        self.obs_levels = obs_levels
        self.env = None
        self.recognizer = None
        self.recipes = []
        self.results = {str(obs): [] for obs in obs_levels}
        self.results['full'] = []
        
        # Create output directory
        os.makedirs("results", exist_ok=True)
        
        # Print record status for debugging
        if self.arglist.record:
            print("Recording enabled in experiment initialization")
    
    def setup(self):
        """Set up the experiment environment and recognizer"""
        #print("Setting up experiment environment...")
        
        # Initialize environment - ensure record flag is passed correctly
        self.env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=self.arglist, disable_env_checker=True)
        
        # Verify record setting is properly applied
        if self.arglist.record:
            print("Record flag confirmed in environment setup")
            
        # Load recipes from level file
        self.recipes = load_recipes(self.arglist.level)
        
        # Create recognizer with specified evaluation method
        eval_method = self._get_evaluation_method(self.arglist.evaluation_method)
        self.recognizer = OvercookedRecognizer(evaluation=eval_method)
        
        # Try to load pre-trained policies
        if not self.recognizer.load_policies(self.env, self.recipes):
            #print("Pre-trained policies not found. Training new policies...")
            self.recognizer.train_policies(self.env, self.recipes, self.arglist)
        
        #print(f"Setup complete. Found {len(self.recipes)} recipes.")
    
    def run(self, num_trials=10):
        """
        Run the experiment with agents using role-specific policies.
        
        Args:
            num_trials: Number of trials to run for each recipe
        """
        #print(f"Running {num_trials} trials for each recipe with agent-specific policies...")
        
        # Execute trials
        for trial in range(num_trials):
            #print(f"Trial {trial+1}/{num_trials}")
            
            # For each recipe
            for recipe_idx, recipe in enumerate(self.recipes):
                #print(f"Testing recipe {recipe_idx+1}/{len(self.recipes)}: {recipe}")
                
                # Reset environment
                obs = self.env.reset()
                
                # Set the current recipe index in the environment for reference
                setattr(self.env, 'current_recipe_idx', recipe_idx)
                
                # Verify record flag if enabled
                if self.arglist.record:
                    if hasattr(self.env, 'game'):
                        if hasattr(self.env.game, 'record'):
                            self.env.game.record = True
                            #print(f"Game record flag set to {self.env.game.record}")
                
                # Generate full trajectories for ALL agents using the relevant policies
                full_trajectories = {}
                for agent_idx in range(self.arglist.num_agents):
                    target_agent_name = f"agent-{agent_idx+1}"
                    trajectory = extract_trajectory(
                        self.env, 
                        policies=self.recognizer.policies,  # Now a dict of agent-specific policies
                        target_agent_name=target_agent_name,
                        max_steps=self.arglist.max_num_timesteps
                    )
                    
                    if len(trajectory) == 0:
                        #print(f"Warning: Empty trajectory generated for {target_agent_name}. Skipping.")
                        break
                    
                    full_trajectories[target_agent_name] = trajectory
                
                # Skip this recipe if any agent has an empty trajectory
                if len(full_trajectories) != self.arglist.num_agents:
                    # print("Skipping recipe due to incomplete trajectories")
                    continue
                
                # Test different observation levels
                for obs_level in self.obs_levels:
                    # Partially observe trajectories for all agents
                    observed_trajs = {
                        agent_name: self.recognizer.observe_trajectory(traj, obs_level)
                        for agent_name, traj in full_trajectories.items()
                    }
                    
                    # Track results for this observation level
                    level_results = []
                    
                    # Recognize goal for each agent's trajectory
                    for agent_name, observed_traj in observed_trajs.items():
                        try:
                            predicted_goal, rankings = self.recognizer.recognize_goal(observed_traj)
                            
                            # Assume the true goal is based on the current recipe
                            is_correct = predicted_goal == recipe_idx
                            level_results.append(float(is_correct))
                            
                            # print(f"{agent_name}, Obs level {obs_level}: Predicted {predicted_goal}, Actual {recipe_idx}, Correct: {is_correct}")
                        except Exception as e:
                            print(f"Error in recognition for {agent_name} at obs level {obs_level}: {e}")
                    
                    # Store results if we have complete data
                    if len(level_results) == self.arglist.num_agents:
                        self.results[str(obs_level)].extend(level_results)
                        self.results['full'].extend(level_results)
                
                # Save intermediate results after each recipe
                self._save_results(f"agent_specific_intermediate_trial_{trial}_recipe_{recipe_idx}")
        
        # Save final results
        self._save_results("agent_specific_final")
        
        # Analyze and visualize results
        self.analyze_results()
        
        return self.results
    
    def visualize_agent_behavior(self, recipe_idx=0, max_steps=100, render_delay=0.5):
        """
        Visualize agents working together using their role-specific trained policies.
        
        Args:
            recipe_idx: Index of the recipe to use
            max_steps: Maximum number of steps to run
            render_delay: Delay between renders in seconds
        """
        import time
        
        # print(f"Visualizing agent behavior for recipe {recipe_idx}")
        
        # Reset environment
        obs = self.env.reset()
        
        # Get the agents
        sim_agents = obs.sim_agents
        
        # Enable recording if available
        if hasattr(self.env, 'game'):
            if hasattr(self.env.game, 'record'):
                self.env.game.record = True
                # print("Recording enabled for visualization")
        
        # Run the visualization
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Create action dictionary - each agent will use its own policy
            action_dict = {}
            
            for agent in sim_agents:
                # Use the agent-specific policy for the current recipe
                if agent.name in self.recognizer.policies and len(self.recognizer.policies[agent.name]) > recipe_idx:
                    policy = self.recognizer.policies[agent.name][recipe_idx]
                    
                    # Get the global state representation
                    env_state_repr = obs.get_repr()
                    
                    # Try to get action from policy
                    try:
                        # Try different state representations in order
                        if env_state_repr in policy.q_table:
                            action_idx = policy.policy(env_state_repr)
                        else:
                            # Try agent-specific state
                            agent_state = (
                                agent.name,
                                agent.location,
                                tuple((obj_name, obj[0].location) if obj else (obj_name, None) 
                                      for obj_name, obj in self.env.world.objects.items()),
                                agent.get_holding()
                            )
                            
                            if agent_state in policy.q_table:
                                action_idx = policy.policy(agent_state)
                            else:
                                # Default to random action
                                action_idx = random.randint(0, 4)
                    except Exception as e:
                        # print(f"Error getting policy action for {agent.name}: {e}")
                        action_idx = random.randint(0, 4)
                else:
                    # No policy available, use random action
                    action_idx = random.randint(0, 4)
                
                actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
                action = actions[action_idx]
                action_dict[agent.name] = action
                
                # print(f"{agent.name} at {agent.location} holding {agent.get_holding()} takes action {action}")
            
            # Take action in environment
            obs, reward, done, _ = self.env.step(action_dict)
            
            # Update sim_agents reference
            sim_agents = obs.sim_agents
            
            # Render environment
            self.env.render()
            
            # Wait for a short time
            time.sleep(render_delay)
            
            steps += 1
            
            # Print step information
            # print(f"Step {steps}: Reward {reward}, Done {done}")
            
            if done:
                print("Goal achieved!")
        
        return done
    
    def _get_evaluation_method(self, method_name):
        """Get the evaluation method based on the method name"""
        methods = {
            'kl': kl_divergence_norm_softmax,
            'dp': soft_divergence_point,
            'q_value': trajectory_q_value
        }
        return methods.get(method_name, kl_divergence_norm_softmax)
    
    def _save_results(self, suffix):
        """Save results to a file"""
        filename = f"results/agent_specific_gr_results_{self.arglist.level}_{suffix}.json"
        with open(filename, 'w') as f:
            import json
            json.dump(self.results, f, indent=2)
    
    def analyze_results(self):
        """Analyze and visualize the results"""
        # print("\nOvercooked Goal Recognition Results (Agent-Specific Policies):")
        
        # Calculate accuracy for each observation level
        accuracies = {}
        for obs in self.obs_levels:
            if len(self.results[str(obs)]) > 0:
                accuracies[obs] = sum(self.results[str(obs)]) / len(self.results[str(obs)])
                print(f"Observation Level: {obs}, Accuracy: {accuracies[obs]:.4f}")
            else:
                print(f"Observation Level: {obs}, No data")
        
        # Calculate overall accuracy
        if len(self.results['full']) > 0:
            avg_full = sum(self.results['full']) / len(self.results['full'])
            # print(f"Average accuracy across all observation levels: {avg_full:.4f}")
        
        # Plot results
        self._plot_results(accuracies)
    
    def _plot_results(self, accuracies):
        """Plot the accuracy results"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        
        # Sort observation levels
        obs_levels = sorted(accuracies.keys())
        acc_values = [accuracies[obs] for obs in obs_levels]
        
        # Create the plot
        plt.plot(obs_levels, acc_values, marker='o', linestyle='-', linewidth=2)
        plt.xlabel('Observation Level')
        plt.ylabel('Recognition Accuracy')
        plt.title('Agent-Specific Goal Recognition Accuracy vs. Observation Level')
        plt.grid(True)
        plt.ylim(0, 1.05)
        
        # Save the plot
        plt.savefig(f"results/agent_specific_gr_accuracy_{self.arglist.level}.png")
        plt.close()