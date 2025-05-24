# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard
from utils.utils import agent_settings

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple

import random

import torch
from gym import spaces
import re

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']


class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes, target):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes
        self.target_item = target
        print(self.recipes)

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        else:
            self.priors = 'spatial'

        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs)

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)
        self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
                agent_name=self.name)
        self.plan(copy.copy(obs))
        return self.action

    def get_subtasks(self, world):
        """Return different subtask permutations for recipes."""
        self.sw = STRIPSWorld(world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks, target=self.target_item)
        all_subtasks = [subtask for path in subtasks for subtask in path]

        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        return all_subtasks

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
        self.delegator = BayesianDelegator(
                agent_name=self.name,
                all_agent_names=env.get_agent_names(),
                model_type=self.model_type,
                planner=self.planner,
                none_action_prob=self.none_action_prob)

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world):
        """Refresh subtasks---relevant for Bayesian Delegation."""
        # Check whether subtask is complete.
        self.subtask_complete = False
        if self.subtask is None or len(self.subtask_agent_names) == 0:
            print("{} has no subtask".format(color(self.name, self.color)))
            return
        self.subtask_complete = self.is_subtask_complete(world)
        print("{} done with {} according to planner: {}\nplanner has subtask {} with subtask object {}".format(
            color(self.name, self.color),
            self.subtask, self.is_subtask_complete(world),
            self.planner.subtask, self.planner.goal_obj))

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
        print('{} incomplete subtasks:'.format(
            color(self.name, self.color)),
            ', '.join(str(t) for t in self.incomplete_subtasks))

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if ((self.subtask is not None and self.subtask not in self.incomplete_subtasks)
                or (self.delegator.should_reset_priors(obs=copy.copy(env),
                            incomplete_subtasks=self.incomplete_subtasks))):
            self.reset_subtasks()
            self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
            else:
                self.delegator.bayes_update(
                        obs_tm1=copy.copy(env.obs_tm1),
                        actions_tm1=env.agent_actions,
                        beta=self.beta)

    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        print('right before planning, {} had old subtask {}, new subtask {}, subtask complete {}'.format(self.name, self.subtask, self.new_subtask, self.subtask_complete))

        # Check whether this subtask is done.
        if self.new_subtask is not None:
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.
        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0-self.none_action_prob)/(len(actions)-1))
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            if self.model_type == 'greedy' or initializing_priors:
                other_agent_planners = {}
            else:
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                backup_subtask = self.new_subtask if self.new_subtask is not None else self.subtask
                other_agent_planners = self.delegator.get_other_agent_planners(
                        obs=copy.copy(env), backup_subtask=backup_subtask)

            print("[ {} Planning ] Task: {}, Task Agents: {}".format(
                self.name, self.new_subtask, self.new_subtask_agent_names))

            action = self.planner.get_next_action(
                    env=env, subtask=self.new_subtask,
                    subtask_agent_names=self.new_subtask_agent_names,
                    other_agent_planners=other_agent_planners)

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0]
            else:
                self.action = action[self.new_subtask_agent_names.index(self.name)] if self.planner.is_joint else action

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_subtask_agent_names = []

        print('{} proposed action: {}\n'.format(self.name, self.action))

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask=self.new_subtask)
        self.subtask_action_object = nav_utils.get_subtask_action_obj(subtask=self.new_subtask)

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
        if isinstance(self.new_subtask, Deliver):
            self.cur_obj_count = len(list(
                filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)),
                env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                    len(list(filter(lambda o: o in
                set(env.world.get_all_object_locs(obj=self.subtask_action_object)),
                w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        # Otherwise, for other subtasks, check based on # of objects.
        else:
            # Current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count

class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        print("{} currently at {}, action {}, holding {}".format(
                color(self.name, self.color),
                self.location,
                self.action,
                self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
         
            
## "HUMAN"
class HybridAgent:
    """
    A hybrid agent that starts with simple fetching behavior but converts to a RealAgent
    when the FetchingAgent in the environment moves.
    """
    def initialize_mock_delegator(self):
        """Initialize a mock delegator for metrics tracking in SIMPLE mode."""
        class MockDelegator:
            def __init__(self, agent_name):
                self.probs = self
                self.agent_name = agent_name
                
            def get_list(self):
                return []  # No probabilities to report
                
            def select_subtask(self, agent_name):
                return None, []
                
            def should_reset_priors(self, obs, incomplete_subtasks):
                return False
                
            def set_priors(self, obs, incomplete_subtasks, priors_type):
                pass
                
            def bayes_update(self, obs_tm1, actions_tm1, beta):
                pass
                
            def get_other_agent_planners(self, obs, backup_subtask):
                return {}
                
        self.delegator = MockDelegator(self.name)
        
    def sync_attributes_from_real_agent(self):
        """Copy metrics tracking attributes from real_agent to this agent."""
        if self.real_agent is None:
            return
            
        # Copy all relevant attributes for metrics tracking
        self.subtask = self.real_agent.subtask
        self.new_subtask = self.real_agent.new_subtask
        self.subtask_agent_names = self.real_agent.subtask_agent_names
        self.new_subtask_agent_names = self.real_agent.new_subtask_agent_names
        self.incomplete_subtasks = self.real_agent.incomplete_subtasks
        self.subtask_complete = getattr(self.real_agent, 'subtask_complete', False)
        self.is_subtask_complete = self.real_agent.is_subtask_complete
        self.delegator = self.real_agent.delegator

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes
        self.location = None
        self.holding = None
        self.action = (0, 0)
        
        # Initial state as a simplified agent
        self.mode = "SIMPLE"
        self.target_item = None
        self.real_agent = None
        self.fetching_agent_prev_location = None
        
        # Metrics tracking attributes
        self.subtask = None
        self.new_subtask = None
        self.subtask_agent_names = []
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.subtask_complete = False
        self.is_subtask_complete = lambda w: False
        self.delegator = None
        
        # Initialize target item based on recipes
        self.determine_target_item()
        
        print(f"{self.name} initialized in SIMPLE mode, targeting: {self.target_item}")

    def __str__(self):
        return color(self.name[-1], self.color)

    def determine_target_item(self):
        """Determine initial target item (sushi or water) based on recipe."""
        try:
            # Check if recipes contain specific ingredients
            recipe_str = str(self.recipes).lower()
            if "sushi" in recipe_str:
                self.target_item = "Sushi"
            elif "water" in recipe_str:
                self.target_item = "Water"
            elif "egg" in recipe_str:
                self.target_item = "Egg"  
            elif "bread" in recipe_str:
                self.target_item = "Bread"
            else:
                # Default to Sushi if unable to determine
                self.target_item = "Sushi"
                
            print(f"Recipe analysis: {recipe_str} -> Target: {self.target_item}")
        except Exception as e:
            print(f"Error determining target item: {e}")
            self.target_item = "Sushi"  # Default fallback

    def select_action(self, obs):
        """Return best next action for this agent based on current mode."""
        # Get agent representation from simulation
        sim_agent = next((a for a in obs.sim_agents if a.name == self.name), None)
       
        if not sim_agent:
            return (0, 0)

        # Check if holding state changed (for transition detection)
        previous_holding = self.holding

        # Update agent state
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        print(previous_holding)
        print(self.holding)

        # Initialize delegator if needed for metrics tracking
        if self.delegator is None:
            self.initialize_mock_delegator()

        # Find the other agent (assume it's the fetching agent)
        fetching_agent = next((a for a in obs.sim_agents if a.name != self.name), None)

        
        

        # Check if we should switch to RealAgent mode based on multiple conditions
        if self.mode == "SIMPLE":
            # Condition 1: The agent picked up an object
            if previous_holding is None and self.holding is not None:
                print(f"{self.name} picked up {self.get_holding()}, switching to REAL_AGENT mode")
                self.mode = "REAL_AGENT"
                # Initialize the RealAgent
                self.initialize_real_agent(obs, target = self.target_item)

            # Condition 2: The other agent moved (original condition)
            elif fetching_agent is not None:
                # Store the other agent's location if this is the first observation
                if self.fetching_agent_prev_location is None:
                    self.fetching_agent_prev_location = fetching_agent.location
                    print(f"Recorded other agent initial location: {self.fetching_agent_prev_location}")
                # Check if the other agent has moved
                elif fetching_agent.location != self.fetching_agent_prev_location or fetching_agent.holding is not None:
                    print(f"Other agent moved from {self.fetching_agent_prev_location} to {fetching_agent.location}")
                    print(f"{self.name} switching to REAL_AGENT mode")
                    self.mode = "REAL_AGENT"
                    # Initialize the RealAgent
                    self.initialize_real_agent(obs, target = self.target_item)

        # Choose action based on current mode
        if self.mode == "SIMPLE":
            return self.simple_mode_action(obs)
        else:  # REAL_AGENT mode
            if self.real_agent is not None:
                # Copy metrics tracking attributes from real_agent to this agent
                self.sync_attributes_from_real_agent()
                return self.real_agent.select_action(obs)
            else:
                print("Warning: real_agent not initialized properly")
                return (0, 0)
    
    def initialize_real_agent(self, obs, target):
        """Initialize a full RealAgent instance with proper setup."""
        try:
            # Create a RealAgent with the same parameters
            self.real_agent = RealAgent(
                arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes,
                target=target
            )

            # Copy current state to the RealAgent
            self.real_agent.location = self.location
            self.real_agent.holding = self.holding
            self.real_agent.action = self.action

            # Critical step: Initialize the subtasks and delegator
            # This is normally done in select_action when obs.t == 0
            self.real_agent.setup_subtasks(env=obs)

            # Update subtasks to ensure delegator is fully initialized
            self.real_agent.update_subtasks(env=obs)

            print(f"Initialized RealAgent for {self.name} with proper subtask setup")
        except Exception as e:
            print(f"Error initializing RealAgent: {e}")
            import traceback
            traceback.print_exc()
            self.mode = "SIMPLE"  # Fallback to SIMPLE mode if initialization fails
    
    def simple_mode_action(self, obs):
        """Determine action in SIMPLE mode - go for target item."""
        # If already holding the target, just stay in place
        if self.holding is not None:
            print(f"{self.name} already holding {self.get_holding()}, waiting for mode switch")
            return (0, 0)

        # Find locations of the target item
        target_locations = []
        try:
            # Look for target item in environment
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue

                if self.target_item.lower() in obj_name.lower() and obj_list:
                    for obj in obj_list:
                        if hasattr(obj, 'location') and obj.location:
                            target_locations.append(obj.location)
                            print(f"Found {self.target_item} at {obj.location}")
        except Exception as e:
            print(f"Error finding target locations: {e}")

        # Get walkable grid for pathfinding
        walkable_grid = self.get_walkable_grid(obs)

        # Make sure target locations are walkable for pathfinding
        for loc in target_locations:
            walkable_grid[loc] = True

        # Find best target and path
        best_target = None
        best_approach = None
        best_path_length = float('inf')

        for loc in target_locations:
            # If we can interact directly with the object
            if self.is_adjacent(self.location, loc):
                best_target = loc
                best_approach = self.location
                break

            # Find shortest path to any cell adjacent to the target
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adjacent = (loc[0] + dx, loc[1] + dy)
                if adjacent in walkable_grid and walkable_grid[adjacent]:
                    path_length = self.shortest_path_length(walkable_grid, self.location, adjacent)
                    if path_length < best_path_length and path_length != float('inf'):
                        best_path_length = path_length
                        best_target = loc
                        best_approach = adjacent

        # If no path found to any target, use basic navigation as fallback
        if best_target is None:
            print("No path found to any target, using basic navigation as fallback")
            closest_loc = min(target_locations, 
                             key=lambda loc: self.manhattan_distance(self.location, loc))
            #return self.navigate_to(closest_loc, obs)

        # If we're adjacent to the target, interact with it
        if self.is_adjacent(self.location, best_target):
            return self.get_direction_to(self.location, best_target)

        # Otherwise, use pathfinding to navigate
        next_step = self.get_next_step_in_path(walkable_grid, self.location, best_approach)
        if next_step:
            return self.get_direction_to(self.location, next_step)

        # Fallback to basic navigation if pathfinding fails
        #return self.navigate_to(best_target, obs)
    
    def get_holding(self):
        """Return the name of the held object."""
        if self.holding is None:
            return 'None'
        return self.holding.full_name
    
    def get_walkable_grid(self, obs):
        """Create a grid representing walkable areas, avoiding counters, obstacles, and other agents.
        Returns:
            dict: A dictionary mapping (x, y) coordinates to True if walkable, False if not
        """
        walkable_grid = {}
        # Get the grid dimensions from the environment if available
        grid_width, grid_height = 10, 10  # Default dimensions
        if hasattr(obs, 'world') and hasattr(obs.world, 'width') and hasattr(obs.world, 'height'):
            grid_width, grid_height = obs.world.width, obs.world.height
        # Initialize all positions as walkable
        for x in range(grid_width):
            for y in range(grid_height):
                walkable_grid[(x, y)] = True
                
        # Mark delivery stations as non-walkable
        try:
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue
                # Look for delivery stations
                if "delivery" in obj_name.lower() and obj_list:
                    for delivery_station in obj_list:
                        if hasattr(delivery_station, 'location') and delivery_station.location:
                            walkable_grid[delivery_station.location] = False
                            #print(f"Marked delivery station at {delivery_station.location} as non-walkable")
        except Exception as e:
            print(f"Error marking delivery stations: {e}")
        
        # Mark counter positions as non-walkable
        try:
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue
                # Look for counters and other obstacles
                if ("counter" in obj_name.lower() or "table" in obj_name.lower() or 
                    "wall" in obj_name.lower() or "obstacle" in obj_name.lower()):
                    for obj in obj_list:
                        if hasattr(obj, 'location') and obj.location:
                            # Mark as non-walkable by default
                            walkable_grid[obj.location] = False

                            # Check if this is a food location and matches our target
                            for f_name, f_list in obs.world.objects.items():
                                if any(item.lower() in f_name.lower() for item in ["sushi", "water", "egg", "bread"]) and f_list:
                                    for food in f_list:
                                        if hasattr(food, 'location') and food.location == obj.location:
                                            # Only make walkable if it's our target item
                                            if hasattr(self, 'target_item') and self.target_item.lower() in f_name.lower():
                                                walkable_grid[obj.location] = True
                                                #print(f"Keeping {obj.location} walkable because it has our target: {self.target_item}")
                                            #else:
                                                #print(f"Marked {obj.location} as non-walkable because it has food we don't want")
        except Exception as e:
            print(f"Error creating walkable grid: {e}")
        # Mark other agents' positions as non-walkable to avoid collisions
        for agent in obs.sim_agents:
            if agent.name != self.name and hasattr(agent, 'location'):
                walkable_grid[agent.location] = False
                #print(f"Marked agent {agent.name} at {agent.location} as non-walkable")

        # Log walkable/non-walkable grid for debugging
        # print("Walkable grid status for human:")
        for y in range(grid_height):
            row = ""
            for x in range(grid_width):
                if (x, y) in walkable_grid:
                    row += "O" if walkable_grid[(x, y)] else "X"
                else:
                    row += "?"
            # print(row)
        return walkable_grid
        
    def shortest_path_length(self, walkable_grid, start, goal):
        """Find the length of the shortest path from start to goal using BFS."""
        from collections import deque

        # If start or goal are not walkable, return infinity
        if not walkable_grid.get(start, False) or not walkable_grid.get(goal, False):
            return float('inf')

        # BFS for shortest path
        queue = deque([(start, 0)])  # (position, distance)
        visited = {start}

        # Four possible movement directions: up, right, down, left
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            pos, distance = queue.popleft()

            if pos == goal:
                return distance

            # Try all four directions
            for dx, dy in directions:
                next_pos = (pos[0] + dx, pos[1] + dy)

                # Check if the new position is valid and walkable
                if (next_pos not in visited and 
                    walkable_grid.get(next_pos, False)):

                    visited.add(next_pos)
                    queue.append((next_pos, distance + 1))

        # If no path is found
        return float('inf')
       
    def get_next_step_in_path(self, walkable_grid, start, goal):
        """Find the next step in the shortest path from start to goal with zig-zag movement preference."""
        from collections import deque
        import random

        # If we're already at the goal, return None
        if start == goal:
            return None

        # If start or goal are not walkable, return None
        if not walkable_grid.get(start, False) or not walkable_grid.get(goal, False):
            return None

        # Four possible movement directions: up, right, down, left
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Find all possible next steps and their path lengths to the goal
        candidate_steps = []
        for dx, dy in directions:
            next_pos = (start[0] + dx, start[1] + dy)

            # Check if the new position is valid and walkable
            if walkable_grid.get(next_pos, False):
                # Use shortest_path_length to calculate the path length to goal
                path_length = self.shortest_path_length(walkable_grid, next_pos, goal)

                # Only consider positions that have a valid path to goal
                if path_length != float('inf'):
                    # Determine if this is a horizontal or vertical move
                    move_type = 'horizontal' if dx != 0 else 'vertical'
                    candidate_steps.append((next_pos, path_length, move_type))
    
        # If we have candidate steps with valid paths
        if candidate_steps:
            # Find the minimum path length among candidates
            min_path_length = min(length for _, length, _ in candidate_steps)

            # Filter to only include steps that have this minimum path length
            best_next_steps = [(pos, move_type) for pos, length, move_type in candidate_steps if length == min_path_length]
            
            # print(best_next_steps)

            # If multiple best next steps exist (same shortest path length), use zig-zag preference
            if len(best_next_steps) > 1:
                # Initialize last_move_type if it doesn't exist
                if not hasattr(self, 'last_move_type'):
                    self.last_move_type = 'none'

                # Prefer the opposite direction of the last move for zig-zag pattern
                preferred_move_type = 'horizontal' if self.last_move_type == 'vertical' else 'vertical'

                # Filter steps that match our preferred move type
                preferred_steps = [pos for pos, move_type in best_next_steps if move_type == preferred_move_type]

                if preferred_steps:
                    # Update the last move type for next time
                    self.last_move_type = preferred_move_type
                    return random.choice(preferred_steps)

            # If no zig-zag preference applied or only one best step, choose randomly from best
            chosen_step, move_type = random.choice(best_next_steps)
            # Update the last move type for next time
            self.last_move_type = move_type
            return chosen_step

    def navigate_to(self, target_location, obs):
        """
        Simple navigation to move toward a target location.
        Tries to alternate between horizontal and vertical movement
        and checks if the next position is walkable.
        """
        dx = target_location[0] - self.location[0]
        dy = target_location[1] - self.location[1]

        # If we're already at the target, don't move
        if dx == 0 and dy == 0:
            return (0, 0)

        # Initialize last_nav_type if it doesn't exist yet
        if not hasattr(self, 'last_nav_type'):
            self.last_nav_type = 'none'

        walkable_grid = self.get_walkable_grid(obs)

        # Determine possible moves (horizontal and vertical)
        h_move = (1 if dx > 0 else -1, 0) if dx != 0 else None
        v_move = (0, 1 if dy > 0 else -1) if dy != 0 else None

        # Determine which move would be preferred for alternating
        preferred_type = 'horizontal' if self.last_nav_type != 'horizontal' and h_move else 'vertical'
        preferred_move = h_move if preferred_type == 'horizontal' else v_move
        alternate_move = v_move if preferred_type == 'horizontal' else h_move

        # Helper function to check if a move is walkable
        def is_walkable(move):
            if not move or not walkable_grid:
                return True  # Assume walkable if we can't check

            next_pos = (self.location[0] + move[0], self.location[1] + move[1])
            return walkable_grid.get(next_pos, False)

        # Try preferred move first (for alternating behavior)
        if preferred_move and is_walkable(preferred_move):
            self.last_nav_type = 'horizontal' if preferred_move[0] != 0 else 'vertical'
            return preferred_move

        # Then try alternate move
        if alternate_move and is_walkable(alternate_move):
            self.last_nav_type = 'horizontal' if alternate_move[0] != 0 else 'vertical'
            return alternate_move

        # If no walkable direction is found, stay in place
        return (0, 0)
    
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent."""
        return self.manhattan_distance(pos1, pos2) == 1
    
    def get_direction_to(self, from_pos, to_pos):
        """Get the direction to move from from_pos to to_pos."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > abs(dy):
            return (1 if dx > 0 else -1, 0) if dx != 0 else (0, 0)
        else:
            return (0, 1 if dy > 0 else -1) if dy != 0 else (0, 0)
    
    # Required methods for compatibility with environment metrics
    def refresh_subtasks(self, world):
        """Compatibility method for metrics tracking."""
        if self.mode == "REAL_AGENT" and self.real_agent is not None:
            self.real_agent.refresh_subtasks(world)
    
    def get_action_location(self):
        """Return location if agent takes its action."""
        import numpy as np
        if self.mode == "REAL_AGENT" and self.real_agent is not None:
            return self.real_agent.get_action_location()
        return tuple(np.asarray(self.location) + np.asarray(self.action))
    
    def all_done(self):
        """Return whether this agent is done with all tasks."""
        if self.mode == "REAL_AGENT" and self.real_agent is not None:
            return self.real_agent.all_done()
        return False
            

## "FETCHER"
class FetchingAgent:
    """Agent specialized in fetching objects for a "Human".
    
    This agent identifies what object (sushi, water, egg, or bread) the other agent is trying to get
    and fetches it for them to optimize team performance. After delivering an object,
    it returns to its original location.
    """
    
    def __init__(self, name, color, arglist=None):
        self.name = name
        self.color = color
        self.target_object = None  # The object to fetch (sushi, water, egg, or bread)
        self.fetchable_objects = ["Sushi", "Water", "Egg", "Bread"]  # Objects this agent can fetch
        self.fetchable_locations = []  # Locations of fetchable objects
        self.other_agent_prev_location = None  # To track other agent movement
        self.other_agent_movement_history = []  # To determine movement patterns
        self.state = "OBSERVE"  # States: OBSERVE, FETCH, DELIVER, RETURN, DONE
        self.location = None
        self.original_location = None  # Store the starting location to return to
        self.holding = None
        self.action = (0, 0)  # Store the most recent action for metrics tracking
        self.delivery_complete = False  # Flag to track if delivery has been made
        self.returning_home = False  # Flag to track if agent is returning home
        self.previous_distances = None  # Initialize distances tracker

        # Store arglist for planner initialization
        self.arglist = arglist

        # Attributes needed for metrics tracking
        self.subtask = None
        self.subtask_agent_names = []
        self.incomplete_subtasks = []
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.subtask_complete = False

        # Define a no-op is_subtask_complete function for compatibility
        self.is_subtask_complete = lambda w: False

        # Mock delegator object for metrics compatibility
        class MockDelegator:
            def __init__(self, agent_name):
                self.probs = self
                self.agent_name = agent_name

            def get_list(self):
                return []  # No probabilities to report

            def select_subtask(self, agent_name):
                return None, []

            def should_reset_priors(self, obs, incomplete_subtasks):
                return False

            def set_priors(self, obs, incomplete_subtasks, priors_type):
                # print(f"{self.agent_name} setting priors")
                pass

            def bayes_update(self, obs_tm1, actions_tm1, beta):
                # print(f"set prior {beta}")
                pass

        self.delegator = MockDelegator(self.name)
        
    def __str__(self):
        return color(self.name[-1], self.color)
        
    def get_holding(self):
        """Return the name of the held object for metrics tracking."""
        if self.holding is None:
            return 'None'
        return self.holding.full_name
    
    def select_action(self, obs):
        """Determine the next action based on the current state and observation."""
        # Update agent's state from the simulation
        sim_agent = next((a for a in obs.sim_agents if a.name == self.name), None)
        if not sim_agent:
            return (0, 0)

        self.location = sim_agent.location

        # Find the other agent (the RealAgent)
        other_agent = next((a for a in obs.sim_agents if a.name != self.name), None)

        # Record original location on first call
        if self.original_location is None:
            self.original_location = self.location
            # print(f"{self.name} recorded original location: {self.original_location}")

            # Initialize distances to objects on first call
            self.initialize_distances(obs, other_agent)

        # Check if the agent was holding something before but now isn't
        holding_before = self.holding
        self.holding = sim_agent.holding

        # If agent was holding something and now isn't, it means delivery has occurred
        if holding_before is not None and self.holding is None:
            # print(f"{self.name} has delivered the object, now returning to original location: {self.original_location}")
            self.returning_home = True
            self.state = "RETURN"

        # Check if we're done (back at original location after delivery)
        if self.returning_home and self.location == self.original_location:
            # print(f"{self.name} has returned to original location and is now standing by.")
            self.delivery_complete = True
            self.returning_home = False
            self.state = "DONE"
            return (0, 0)

        # If fully done, just stand in place
        if self.delivery_complete and self.state == "DONE":
            return (0, 0)

        # If other agent is already holding something while we're still in OBSERVE or FETCH state,
        # abort our current task and return to our original location. The logic here is that we don't
        # need to observe the agent or fetch the object because we can assume the other agent already
        # got the object. 
        if other_agent.holding is not None and self.state in ['OBSERVE', 'FETCH']:
            # print(f"Other agent already holding {other_agent.holding.full_name}, no need to fetch")
            # If we're already holding something, drop it
            if self.holding is not None:
                print(f"{self.name} will drop {self.get_holding()} and return to original location")

            # Skip to RETURN state regardless of current state
            if self.state != "RETURN" and self.state != "DONE":
                self.returning_home = True
                self.state = "RETURN"
                # print(f"{self.name} switching to RETURN state")

            if self.state == "RETURN":
                return self.return_to_origin_state(obs)
            return (0, 0)

        # Update fetchable object locations
        self.update_fetchable_locations(obs)

        # Initialize tracking of other agent if needed
        if self.other_agent_prev_location is None:
            self.other_agent_prev_location = other_agent.location

        # State machine for fetching behavior
        if self.state == "OBSERVE":
            action = self.observe_state(obs, other_agent)
        elif self.state == "FETCH":
            action = self.fetch_state(obs, other_agent)
        elif self.state == "DELIVER":
            action = self.deliver_state(obs, other_agent)
        elif self.state == "RETURN":
            action = self.return_to_origin_state(obs)
        else:
            action = (0, 0)

        # Validate the action format
        if not isinstance(action, tuple) or len(action) != 2:
            # print(f"Invalid action format returned: {action}, defaulting to (0, 0)")
            action = (0, 0)

        # Ensure action components are integers
        action = (int(action[0]), int(action[1]))

        # Ensure action is one of the valid cardinal movements or staying still
        valid_actions = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        if action not in valid_actions:
            # print(f"Invalid action {action}, defaulting to (0, 0)")
            action = (0, 0)

        # Store the action for metrics tracking
        self.action = action

        # Print current state and action for debugging
        # print(f"{self.name} is in state {self.state}, taking action {self.action}")

        return self.action
    
    def initialize_distances(self, obs, other_agent):
        """Initialize distances to all fetchable objects at the beginning."""

        # print(f"Initializing distances from other agent at {other_agent.location}")

        # Use the calculate_object_distances function to initialize distances
        self.previous_distances = self.calculate_object_distances(obs, other_agent.location)
        
        # Also store these as initial distances for total change calculation
        if self.previous_distances:
            self.initial_distances = self.previous_distances.copy()

        # Log the initial distances
        if self.previous_distances:
            for obj, dist in self.previous_distances.items():
                print(f"Initial distance to {obj}: {dist}")
        else:
            print("Failed to initialize distances")
            
    def calculate_object_distances(self, obs, agent_location):
        """Calculate distances from an agent location to each fetchable object.

        Args:
            obs: The observation from the environment
            agent_location: The location to calculate distances from

        Returns:
            dict: Mapping from object names to their shortest path distances
        """
        # Get locations of all fetchable objects
        object_locations = {obj: [] for obj in self.fetchable_objects}

        try:
            # Find locations for each fetchable object
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue

                for fetchable_obj in self.fetchable_objects:
                    if fetchable_obj.lower() in obj_name.lower() and obj_list and hasattr(obj_list[0], 'location') and obj_list[0].location:
                        object_locations[fetchable_obj].append(obj_list[0].location)
                        # print(f"Found {fetchable_obj} at: {obj_list[0].location}")

            # Log the found locations
            for obj, locs in object_locations.items():
                print(f"Found {obj} at: {locs}")

            # Get the walkable grid from the environment, with counters marked as obstacles
            walkable_grid = self.get_walkable_grid(obs)

            # Make sure target locations are marked as walkable for pathfinding
            for locs in object_locations.values():
                for loc in locs:
                    walkable_grid[loc] = True

            # Calculate current shortest path distances to closest location for each object
            current_distances = {}

            for obj, locs in object_locations.items():
                shortest_dist = float('inf')
                for loc in locs:
                    # Find distances to cells adjacent to the target for interaction
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        adjacent = (loc[0] + dx, loc[1] + dy)
                        if adjacent in walkable_grid and walkable_grid[adjacent]:
                            path_length = self.shortest_path_length(walkable_grid, agent_location, adjacent)
                            if path_length < shortest_dist:
                                shortest_dist = path_length

                current_distances[obj] = shortest_dist
                #print(f"Current shortest path distance to nearest {obj}: {shortest_dist}")

            # Check if any paths are possible
            possible_objects = [obj for obj, dist in current_distances.items() if dist != float('inf')]

            # If no valid paths are found, fallback to Manhattan distance
            if not possible_objects:
                print("No valid path to any object using pathfinding, falling back to Manhattan distance")

                # Recalculate using Manhattan distance instead
                current_distances = {}

                for obj, locs in object_locations.items():
                    if not locs:
                        continue

                    # Find the closest location for this object using Manhattan distance
                    closest_dist = float('inf')
                    for loc in locs:
                        dist = self.manhattan_distance(agent_location, loc)
                        if dist < closest_dist:
                            closest_dist = dist

                    current_distances[obj] = closest_dist
                    # print(f"Manhattan distance to nearest {obj}: {closest_dist}")

            return current_distances

        except Exception as e:
            print(f"Error calculating object distances: {e}")
            return None

    def return_to_origin_state(self, obs):
        """Return to the original starting location."""
        if self.location == self.original_location:
            print(f"{self.name} has returned to original location: {self.original_location}")
            self.state = "DONE"
            self.delivery_complete = True
            self.returning_home = False
            return (0, 0)
        
        walkable_grid = self.get_walkable_grid(obs)
        
        next_step = self.get_next_step_in_path(walkable_grid, self.location, self.original_location)
        if next_step:
            return self.get_direction_to(self.location, next_step)

        #return self.navigate_to(self.original_location, obs)
    
        
    def update_fetchable_locations(self, obs):
        """Update the locations of fetchable objects in the environment."""
        self.fetchable_locations = []
        
        try:
            # Find objects representing prepared food locations (sushi/water/egg/bread)
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue
                    
                if (any(item.lower() in obj_name.lower() for item in ["sushi", "water", "egg", "bread"]) and 
                    obj_list and hasattr(obj_list[0], 'location') and obj_list[0].location):
                    self.fetchable_locations.append(obj_list[0].location)
                    print(f"Found {obj_name} at location: {obj_list[0].location}")
            
            # If we found pickup locations, we're done
            if self.fetchable_locations:
                return
                
            # Fallback: look for counters that might have food
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue
                    
                if (("counter" in obj_name.lower() or "table" in obj_name.lower()) and 
                    obj_list and hasattr(obj_list[0], 'location') and obj_list[0].location):
                    self.fetchable_locations.append(obj_list[0].location)
                    print(f"Fallback: Found {obj_name} at location: {obj_list[0].location}")
                    
        except Exception as e:
            print(f"Error finding fetchable objects: {e}")
            # Default locations if all else fails
            self.fetchable_locations = [(2, 2), (3, 2), (4, 2)]
            
        if self.fetchable_locations:
            print(f"Found fetchable objects at: {self.fetchable_locations}")
        else:
            print("No fetchable objects found!")
        
    
    def observe_state(self, obs, other_agent):
        """Observe other agent's behavior to determine what to fetch."""
        # Track movement
        if self.other_agent_prev_location != other_agent.location:
            # Agent has moved, record direction
            direction = (
                other_agent.location[0] - self.other_agent_prev_location[0],
                other_agent.location[1] - self.other_agent_prev_location[1]
            )
            self.other_agent_movement_history.append(direction)
            self.other_agent_prev_location = other_agent.location

            # Check if there's a significant difference in distance to targetable objects
            is_target_determined = self.determine_target_object(obs, other_agent)

            if is_target_determined and self.target_object:
                self.state = "FETCH"
                print(f"{self.name} has determined target object: {self.target_object}")
                # Immediately start fetching
                return self.fetch_state(obs, other_agent)

        # While observing, move to a central position or stay put
        return self.get_observation_position(obs)
    
    # TOTAL CHANGE IN SHORTEST WALKABLE PATH
    def determine_target_object(self, obs, other_agent):
        """Determine which object the other agent is trying to get based on total change in shortest path
        from initial position to current position.

        Args:
            obs: The observation from the environment
            other_agent: The agent we're observing

        Returns:
            bool: True if the agent should switch to fetching, False if it should continue observing
        """
        # Calculate current distances from other agent to objects
        current_distances = self.calculate_object_distances(obs, other_agent.location)

        # If we couldn't calculate distances, continue observing
        if not current_distances:
            print("Could not calculate current distances, continuing to observe")
            return False

        # Calculate the TOTAL change in distance from initial state to current state for each object
        total_distance_changes = {}
        for obj in current_distances:
            if obj in self.initial_distances:
                # Negative change means the agent has moved closer to the object since the start
                change = current_distances[obj] - self.initial_distances[obj]
                total_distance_changes[obj] = change
                print(f"Total distance change for {obj}: {change} (initial: {self.initial_distances[obj]}, current: {current_distances[obj]})")

        # Find the object with the most negative total change in distance (most approached since start)
        most_approached = None
        min_total_change = float('inf')

        for obj, change in total_distance_changes.items():
            if change < min_total_change:
                min_total_change = change
                most_approached = obj

        # Find the second most approached object
        second_min_total_change = float('inf')
        for obj, change in total_distance_changes.items():
            if change < second_min_total_change and obj != most_approached:
                second_min_total_change = change

        # Calculate the difference in total distance change
        total_change_difference = second_min_total_change - min_total_change
        print(f"Total change difference between most approached ({most_approached}: {min_total_change}) and second most ({second_min_total_change}): {total_change_difference}")

        # Only transition to FETCH if one object is being approached significantly more than others
        # Significant difference threshold could be 1 or more steps
        if total_change_difference >= 1:
            self.target_object = most_approached
            print(f"Agent is approaching {self.target_object} relatively more than others since the start, switching to FETCH state")
            # Store current distances for next comparison (still needed for incremental updates)
            self.previous_distances = current_distances
            return True
        else:
            # If no clear approach pattern, continue observing
            print("No clear approach pattern detected, continuing to observe")
            # Store current distances for next comparison
            self.previous_distances = current_distances
            return False
    
    
#     # MOST RECENT CHANGE IN SHORTEST WALKABLE PATH
#     def determine_target_object(self, obs, other_agent):
#         """Determine which object the other agent is trying to get based on change in shortest path.

#         Args:
#             obs: The observation from the environment
#             other_agent: The agent we're observing

#         Returns:
#             bool: True if the agent should switch to fetching, False if it should continue observing
#         """
#         # Calculate current distances from other agent to objects
#         current_distances = self.calculate_object_distances(obs, other_agent.location)

#         # If we couldn't calculate distances, continue observing
#         if not current_distances:
#             print("Could not calculate current distances, continuing to observe")
#             return False

#         # Calculate the change in distance for each object
#         distance_changes = {}
#         for obj in current_distances:
#             if obj in self.previous_distances:
#                 change = current_distances[obj] - self.previous_distances[obj]
#                 distance_changes[obj] = change
#                 print(f"Distance change for {obj}: {change} (previous: {self.previous_distances[obj]}, current: {current_distances[obj]})")

#         # Find the object with the most negative change in distance (most approached)
#         most_approached = None
#         min_change = float('inf')

#         for obj, change in distance_changes.items():
#             if change < min_change:
#                 min_change = change
#                 most_approached = obj

#         # Find the second most approached object
#         second_min_change = float('inf')
#         for obj, change in distance_changes.items():
#             if change < second_min_change and obj != most_approached:
#                 second_min_change = change

#         # Calculate the difference in distance change
#         change_difference = second_min_change - min_change
#         print(f"Change difference between most approached ({most_approached}: {min_change}) and second most ({second_min_change}): {change_difference}")

#         # Only transition to FETCH if one object is being approached significantly more than others
#         # Significant difference threshold could be 1 or more steps
#         if change_difference >= 1:
#             self.target_object = most_approached
#             print(f"Agent is approaching {self.target_object} relatively more than others, switching to FETCH state")
#             # Store current distances for next comparison
#             self.previous_distances = current_distances
#             return True
#         else:
#             # If no clear approach pattern, continue observing
#             print("No clear approach pattern detected, continuing to observe")
#             # Store current distances for next comparison
#             self.previous_distances = current_distances
#             return False
    
#     # JUST BASED ON SHORTEST WALKABLE PATH
#     def determine_target_object(self, obs, other_agent):
#         """Determine which object the other agent is trying to get based on shortest valid path.

#         Returns:
#             bool: True if the agent should switch to fetching, False if it should continue observing
#         """
#         # Get locations of all fetchable objects
#         object_locations = {obj: [] for obj in self.fetchable_objects}

#         try:
#             # Find locations for each fetchable object
#             for obj_name, obj_list in obs.world.objects.items():
#                 if not obj_list:
#                     continue

#                 for fetchable_obj in self.fetchable_objects:
#                     if fetchable_obj.lower() in obj_name.lower() and obj_list and hasattr(obj_list[0], 'location') and obj_list[0].location:
#                         object_locations[fetchable_obj].append(obj_list[0].location)
#                         print(f"Found {fetchable_obj} at: {obj_list[0].location}")

#             # Log the found locations
#             for obj, locs in object_locations.items():
#                 print(f"Found {obj} at: {locs}")

#             # Get the walkable grid from the environment, with counters marked as obstacles
#             walkable_grid = self.get_walkable_grid(obs)

#             # Make sure target locations are marked as walkable for pathfinding
#             for locs in object_locations.values():
#                 for loc in locs:
#                     walkable_grid[loc] = True

#             # Calculate shortest path distances to closest location for each object
#             shortest_distances = {}

#             for obj, locs in object_locations.items():
#                 shortest_dist = float('inf')
#                 for loc in locs:
#                     # Find distances to cells adjacent to the target for interaction
#                     for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
#                         adjacent = (loc[0] + dx, loc[1] + dy)
#                         if adjacent in walkable_grid and walkable_grid[adjacent]:
#                             path_length = self.shortest_path_length(walkable_grid, other_agent.location, adjacent)
#                             if path_length < shortest_dist:
#                                 shortest_dist = path_length

#                 shortest_distances[obj] = shortest_dist
#                 print(f"Shortest path distance to nearest {obj}: {shortest_dist}")

#             # Check if any paths are possible
#             possible_objects = [obj for obj, dist in shortest_distances.items() if dist != float('inf')]

#             # If no valid paths are found, fallback to Manhattan distance
#             if not possible_objects:
#                 print("No valid path to any object using pathfinding, falling back to Manhattan distance")

#                 # Recalculate using Manhattan distance instead
#                 shortest_distances = {}

#                 for obj, locs in object_locations.items():
#                     if not locs:
#                         continue

#                     # Find the closest location for this object using Manhattan distance
#                     closest_dist = float('inf')
#                     for loc in locs:
#                         dist = self.manhattan_distance(other_agent.location, loc)
#                         if dist < closest_dist:
#                             closest_dist = dist

#                     shortest_distances[obj] = closest_dist
#                     print(f"Manhattan distance to nearest {obj}: {closest_dist}")

#                 # Populate possible_objects based on Manhattan distance
#                 possible_objects = [obj for obj, dist in shortest_distances.items() if dist < float('inf')]
#                 print(f"Using Manhattan distance, found possible objects: {possible_objects}")

#             if len(possible_objects) == 1:
#                 print(f"Only found path to {possible_objects[0]}")
#                 self.target_object = possible_objects[0]
#                 return True

#             # Find the object with the shortest path
#             min_dist = float('inf')
#             closest_obj = None

#             for obj, dist in shortest_distances.items():
#                 if dist < min_dist:
#                     min_dist = dist
#                     closest_obj = obj

#             # Find the second shortest path
#             second_min_dist = float('inf')
#             for obj, dist in shortest_distances.items():
#                 if dist < second_min_dist and obj != closest_obj:
#                     second_min_dist = dist

#             # Calculate the difference in path length
#             distance_difference = second_min_dist - min_dist
#             print(f"Path distance difference between closest ({closest_obj}: {min_dist}) and second closest ({second_min_dist}): {distance_difference}")

#             # Only transition to FETCH if one object has a significantly shorter path
#             # Significant difference threshold is 1 or more steps
#             if distance_difference >= 1:
#                 self.target_object = closest_obj
#                 print(f"Agent's path is {distance_difference} steps shorter to {self.target_object}, switching to FETCH state")
#                 return True
#             else:
#                 # If the difference is not significant, continue observing
#                 print("Paths to objects are approximately equal, continuing to observe")
#                 return False

#         except Exception as e:
#             print(f"Error determining target object: {e}")
#             # In case of error, continue observing instead of making a decision
#             print("Error occurred, continuing to observe")
#             return False

    def get_walkable_grid(self, obs):
        """Create a grid representing walkable areas, avoiding counters, obstacles, and other agents.
        Returns:
            dict: A dictionary mapping (x, y) coordinates to True if walkable, False if not
        """
        walkable_grid = {}
        # Get the grid dimensions from the environment if available
        grid_width, grid_height = 10, 10  # Default dimensions
        if hasattr(obs, 'world') and hasattr(obs.world, 'width') and hasattr(obs.world, 'height'):
            grid_width, grid_height = obs.world.width, obs.world.height
        # Initialize all positions as walkable
        for x in range(grid_width):
            for y in range(grid_height):
                walkable_grid[(x, y)] = True
                
        # Mark delivery stations as non-walkable
        try:
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue
                # Look for delivery stations
                if "delivery" in obj_name.lower() and obj_list:
                    for delivery_station in obj_list:
                        if hasattr(delivery_station, 'location') and delivery_station.location:
                            walkable_grid[delivery_station.location] = False
                            #print(f"Marked delivery station at {delivery_station.location} as non-walkable")
        except Exception as e:
            print(f"Error marking delivery stations: {e}")
            
        # Mark counter positions as non-walkable
        try:
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue
                # Look for counters and other obstacles
                if ("counter" in obj_name.lower() or "table" in obj_name.lower() or 
                    "wall" in obj_name.lower() or "obstacle" in obj_name.lower()):
                    for obj in obj_list:
                        if hasattr(obj, 'location') and obj.location:
                            # Mark as non-walkable by default
                            walkable_grid[obj.location] = False

                            # Check if this is a food location and matches our target
                            for f_name, f_list in obs.world.objects.items():
                                if any(item.lower() in f_name.lower() for item in ["sushi", "water", "egg", "bread"]) and f_list:
                                    for food in f_list:
                                        if hasattr(food, 'location') and food.location == obj.location:
                                            # Only make walkable if it's our target item
                                            if self.target_object and self.target_object.lower() in f_name.lower():
                                                walkable_grid[obj.location] = True
                                                #print(f"Keeping {obj.location} walkable because it has our target: {self.target_item}")
                                            #else:
                                                #print(f"Marked {obj.location} as non-walkable because it has food we don't want")
        except Exception as e:
            print(f"Error creating walkable grid: {e}")
        # Mark other agents' positions as non-walkable to avoid collisions
        if hasattr(self, 'state') and self.state not in ['DELIVER', 'OBSERVE']:
            for agent in obs.sim_agents:
                if agent.name != self.name and hasattr(agent, 'location'):
                    walkable_grid[agent.location] = False
                    #print(f"Marked agent {agent.name} at {agent.location} as non-walkable")

        # Log walkable/non-walkable grid for debugging
        print("Walkable grid status for fetcher:")
        for y in range(grid_height):
            row = ""
            for x in range(grid_width):
                if (x, y) in walkable_grid:
                    row += "O" if walkable_grid[(x, y)] else "X"
                else:
                    row += "?"
            print(row)
        return walkable_grid

    def fetch_state(self, obs, other_agent):
        """Fetch the target object using planner-based navigation."""
        # If already holding something, move to delivery
        if self.holding is not None:
            print(f"{self.name} is now holding {self.holding.full_name}, moving to DELIVER state")
            self.state = "DELIVER"
            return self.deliver_state(obs, other_agent)

        # Find locations of target object
        target_locations = []

        try:
            # Look directly for the target object by name
            for obj_name, obj_list in obs.world.objects.items():
                if not obj_list:
                    continue

                # Check if object name matches our target
                if self.target_object.lower() in obj_name.lower() and obj_list:
                    if hasattr(obj_list[0], 'location') and obj_list[0].location:
                        target_locations.append(obj_list[0].location)
                        print(f"Found target {self.target_object} at {obj_list[0].location}")

        except Exception as e:
            print(f"Error finding target object locations: {e}")

        if not target_locations:
            print(f"Can't find the target object {self.target_object}, falling back to fetchable locations")
            target_locations = self.fetchable_locations

        if not target_locations:
            # Still no target, go back to observing
            print(f"No target locations found, going back to OBSERVE state")
            self.state = "OBSERVE"
            return self.observe_state(obs, other_agent)

        # Get walkable grid
        walkable_grid = self.get_walkable_grid(obs)

        # Find closest target location by path length
        closest_loc = None
        min_distance = float('inf')
        closest_approach_point = None

        for loc in target_locations:
            # If we can interact directly with the object
            if self.is_adjacent(self.location, loc):
                closest_loc = loc
                break

            # Otherwise, find the closest cell adjacent to the target
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                adjacent = (loc[0] + dx, loc[1] + dy)
                if adjacent in walkable_grid and walkable_grid[adjacent]:
                    path_length = self.shortest_path_length(walkable_grid, self.location, adjacent)
                    if path_length < min_distance:
                        min_distance = path_length
                        closest_loc = loc
                        closest_approach_point = adjacent

        if closest_loc is None:
            print(f"No reachable target locations, going back to OBSERVE state")
            self.state = "OBSERVE"
            return self.observe_state(obs, other_agent)

        print(f"Heading to closest {self.target_object} at {closest_loc} (path length: {min_distance})")

        # If we're adjacent to the target, interact with it
        if self.is_adjacent(self.location, closest_loc):
            return self.get_direction_to(self.location, closest_loc)
        
        # Otherwise, use pathfinding to navigate
        next_step = self.get_next_step_in_path(walkable_grid, self.location, closest_loc)
        if next_step:
            return self.get_direction_to(self.location, next_step)
        
        # Fallback to basic navigation if pathfinding fails
        #return self.navigate_to(closest_loc, obs)
    
    
    def shortest_path_length(self, walkable_grid, start, goal):
        """Find the length of the shortest path from start to goal using BFS."""
        from collections import deque

        # If start or goal are not walkable, return infinity
        if not walkable_grid.get(start, False) or not walkable_grid.get(goal, False):
            return float('inf')

        # BFS for shortest path
        queue = deque([(start, 0)])  # (position, distance)
        visited = {start}

        # Four possible movement directions: up, right, down, left
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            pos, distance = queue.popleft()

            if pos == goal:
                return distance

            # Try all four directions
            for dx, dy in directions:
                next_pos = (pos[0] + dx, pos[1] + dy)

                # Check if the new position is valid and walkable
                if (next_pos not in visited and 
                    walkable_grid.get(next_pos, False)):

                    visited.add(next_pos)
                    queue.append((next_pos, distance + 1))

        # If no path is found
        return float('inf')
    
    def deliver_state(self, obs, other_agent):
        """Deliver the fetched object to the other agent using pathfinding."""
        if self.holding is None:
            # We don't have the object anymore, transition to RETURN state
            print(f"{self.name} has delivered the object, now returning to original location: {self.original_location}")
            self.state = "RETURN"
            self.returning_home = True
            return self.return_to_origin_state(obs)

        walkable_grid = self.get_walkable_grid(obs)
        
        # If we're adjacent to the target, interact with it
        if self.is_adjacent(self.location, other_agent.location):
            return self.get_direction_to(self.location, other_agent.location)
        
        # Otherwise, use pathfinding to navigate
        next_step = self.get_next_step_in_path(walkable_grid, self.location, other_agent.location)
        if next_step:
            return self.get_direction_to(self.location, next_step)
        
        #return self.navigate_to(other_agent.location, obs)

        
    def get_observation_position(self, obs):
        """Get a good position to observe from."""
        # Stay still during observation
        return (0, 0)
    
    def get_next_step_in_path(self, walkable_grid, start, goal):
        """Find the next step in the shortest path from start to goal."""
        from collections import deque

        # If we're already at the goal, return None
        if start == goal:
            return None

        # If start or goal are not walkable, return None
        if not walkable_grid.get(start, False) or not walkable_grid.get(goal, False):
            return None

        # BFS to find the path
        queue = deque([(start, [])])  # (position, path)
        visited = {start}

        # Four possible movement directions: up, right, down, left
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while queue:
            pos, path = queue.popleft()

            if pos == goal:
                # Return the first step in the path
                return path[0] if path else None

            # Try all four directions
            for dx, dy in directions:
                next_pos = (pos[0] + dx, pos[1] + dy)

                # Check if the new position is valid and walkable
                if (next_pos not in visited and 
                    walkable_grid.get(next_pos, False)):

                    new_path = path + [next_pos]
                    visited.add(next_pos)
                    queue.append((next_pos, new_path))

        # If no path is found
        return None
        
    def navigate_to(self, target_location, obs):
        """
        Simple navigation to move toward a target location.
        Tries to alternate between horizontal and vertical movement
        and checks if the next position is walkable.
        """
        dx = target_location[0] - self.location[0]
        dy = target_location[1] - self.location[1]

        # If we're already at the target, don't move
        if dx == 0 and dy == 0:
            return (0, 0)

        # Initialize last_nav_type if it doesn't exist yet
        if not hasattr(self, 'last_nav_type'):
            self.last_nav_type = 'none'

        walkable_grid = self.get_walkable_grid(obs)

        # Determine possible moves (horizontal and vertical)
        h_move = (1 if dx > 0 else -1, 0) if dx != 0 else None
        v_move = (0, 1 if dy > 0 else -1) if dy != 0 else None

        # Determine which move would be preferred for alternating
        preferred_type = 'horizontal' if self.last_nav_type != 'horizontal' and h_move else 'vertical'
        preferred_move = h_move if preferred_type == 'horizontal' else v_move
        alternate_move = v_move if preferred_type == 'horizontal' else h_move

        # Helper function to check if a move is walkable
        def is_walkable(move):
            if not move or not walkable_grid:
                return True  # Assume walkable if we can't check

            next_pos = (self.location[0] + move[0], self.location[1] + move[1])
            return walkable_grid.get(next_pos, False)

        # Try preferred move first (for alternating behavior)
        if preferred_move and is_walkable(preferred_move):
            self.last_nav_type = 'horizontal' if preferred_move[0] != 0 else 'vertical'
            return preferred_move

        # Then try alternate move
        if alternate_move and is_walkable(alternate_move):
            self.last_nav_type = 'horizontal' if alternate_move[0] != 0 else 'vertical'
            return alternate_move

        # If no walkable direction is found, stay in place
        return (0, 0)
            
    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
    def is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent."""
        return self.manhattan_distance(pos1, pos2) == 1
        
    def get_direction_to(self, from_pos, to_pos):
        """Get the direction to move from from_pos to to_pos."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > abs(dy):
            return (1 if dx > 0 else -1, 0) if dx != 0 else (0, 0)
        else:
            return (0, 1 if dy > 0 else -1) if dy != 0 else (0, 0)
            
    def get_direction_to_closest(self, from_pos, target_locations):
        """Get direction vector to the closest location in a list."""
        if not target_locations:
            return (0, 0)
            
        closest_loc = min(target_locations, key=lambda loc: self.manhattan_distance(from_pos, loc))
        dx = closest_loc[0] - from_pos[0]
        dy = closest_loc[1] - from_pos[1]
        
        # Normalize direction vector
        magnitude = max(1, abs(dx) + abs(dy))
        return (dx/magnitude, dy/magnitude)
        
    def dot_product(self, v1, v2):
        """Calculate dot product of two vectors."""
        return v1[0] * v2[0] + v1[1] * v2[1]
        
    def all_done(self):
        """Return whether this agent is done with all tasks."""
        return True
        
    def refresh_subtasks(self, world):
        """Dummy method for compatibility with metrics tracking."""
        pass
        
    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        import numpy as np
        return tuple(np.asarray(self.location) + np.asarray(self.action))
    
class DQNFetchingAgent:
    """Agent specialized in fetching objects for a "Human" using DQN and belief.
    
    This agent identifies what object (sushi, water, egg, or bread) the other agent is trying to get
    and fetches it for them to optimize team performance. After delivering an object,
    it returns to its original location.
    """
    
    def __init__(self, name, color, arglist=None):
        self.name = name
        self.color = color
        self.holding = None
        self.arglist = arglist
        self.location = None
        self.action = (0, 0)
        # Attributes needed for metrics tracking
        self.subtask = None
        self.subtask_agent_names = []
        self.incomplete_subtasks = []
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.subtask_complete = False

        # Define a no-op is_subtask_complete function for compatibility
        self.is_subtask_complete = lambda w: False
        self.action_map = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_space = spaces.Discrete(len(self.action_map))

        # Mock delegator object for metrics compatibility
        class MockDelegator:
            def __init__(self, agent_name):
                self.probs = self
                self.agent_name = agent_name

            def get_list(self):
                return []  # No probabilities to report

            def select_subtask(self, agent_name):
                return None, []

            def should_reset_priors(self, obs, incomplete_subtasks):
                return False

            def set_priors(self, obs, incomplete_subtasks, priors_type):
                # print(f"{self.agent_name} setting priors")
                pass

            def bayes_update(self, obs_tm1, actions_tm1, beta):
                # print(f"set prior {beta}")
                pass

        self.delegator = MockDelegator(self.name)

    def __str__(self):
        return color(self.name[-1], self.color)
        
    def get_holding(self):
        """Return the name of the held object for metrics tracking."""
        if self.holding is None:
            return 'None'
        return self.holding.full_name
    
    def strip_ansi(self,text):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)
    
    def state_to_dqn_input(self, obs) -> torch.Tensor:
        
        clean_grid = [[self.strip_ansi(cell) for cell in row] for row in obs]
        is_empty = False
        if clean_grid == []:
            is_empty = True

        symbol_to_onehot = {
            '-':    [1,0,0,0,0,0],
            '1':    [0,1,0,0,0,0],
            '2':    [0,0,1,0,0,0],
            '*':    [0,0,0,1,0,0],
            'p-w':  [0,0,0,0,1,0],
            'p-s':  [0,0,0,0,0,1],
        }

        onehot_vectors = []

        for row in clean_grid:
            for cell in row:
                onehot = symbol_to_onehot.get(cell, [0,0,0,0,0,0])  
                onehot_vectors.extend(onehot)

        # Convert list to flat FloatTensor
        input_tensor = torch.FloatTensor(onehot_vectors)

        return input_tensor, is_empty

    
    def select_action(self, obs,env, epsilon, policy):
    
        sim_agent = next((a for a in obs.sim_agents if a.name == self.name), None)
        # Update internal state from SimAgent
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action
            

        if random.random() < epsilon:
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                dqn_indput, is_empty = self.state_to_dqn_input(env.rep)

                if not is_empty:
                    print(policy(dqn_indput))
                    action = policy(dqn_indput).argmax().item()
                else:
                    action = self.action_space.sample()
        return self.action_map[action], action
    
    def all_done(self):
        """Return whether this agent is done with all tasks."""
        return True
        
    def refresh_subtasks(self, world):
        """Dummy method for compatibility with metrics tracking."""
        pass
        
    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))
