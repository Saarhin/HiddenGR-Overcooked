# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS, FetchingAgent, HybridAgent, SimpleAgent
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple

import gym

# -----------------------
# -----------------------
# KIRIN ADDED
from overcooked_gr_experiments import OvercookedGRExperiment
from ml.metrics import kl_divergence_norm_softmax, soft_divergence_point, trajectory_q_value
from ml.rl import TabularQLearner
# -----------------------

from belief import OvercookedBeliefExperiment


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")
    
    # -----------------------
    # -----------------------
    # KIRIN ADDED
    # Goal Recognition arguments
    parser.add_argument("--gr-experiments", action="store_true", default=False, help="Run goal recognition experiments")
    parser.add_argument("--evaluation-method", type=str, default="kl", choices=["kl", "dp", "q_value"],
                        help="Evaluation method for goal recognition: kl (KL divergence), dp (divergence point), q_value (trajectory Q-value)")
    parser.add_argument("--obs-levels", type=str, default="1.0",
                        help="Observation levels for goal recognition (comma-separated)")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Number of trials for each agent-recipe combination")
    parser.add_argument("--train-policies", action="store_true", default=False,
                        help="Force training new policies even if pre-trained ones exist")
    # -----------------------

    # -----------------------
    # SARAH ADDED
    # We want the fetcher to have a belief of what the goal is
    parser.add_argument("--belief-experiments", action="store_true", default=False, help="Run belief agent")
    parser.add_argument("--dqn-input", type=str, default="Summary",
                        help="What type of input does the fetcher get? Full, Summary, Summary+belief(distance), Summary+belief(GVFs)")
    parser.add_argument("--folder-name", type=str, default="fetcherDQN_simpleChef",
                        help="name of the saving folder")
    parser.add_argument("--chef-test", action="store_true", default=False, help="Thest the chef performance in fron of a human player")
    parser.add_argument("--single-agent", action="store_true", default=False, help="Even though there are 2 players the chef should stay still")
    parser.add_argument("--num-eps", type=int, default=100000,
                        help="Number of episodes")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Saves model and reward every n episodes")


    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
#                     real_agent = RealAgent(
#                             arglist=arglist,
#                             name='agent-'+str(len(real_agents)+1),
#                             id_color=COLORS[len(real_agents)],
#                             recipes=recipes)

                    # ----------------------------------------
                    # ----------------------------------------
                    # KIRIN Added
#                     if len(real_agents)+1 == 1:
#                         print(f'Initializing agent {len(real_agents)+1} with no recipes at location ({loc[0]}, {loc[1]})')
#                         real_agent = RealAgent(
#                                 arglist=arglist,
#                                 name='agent-'+str(len(real_agents)+1),
#                                 id_color=COLORS[len(real_agents)],
#                                 recipes=[])
                    
                    # MAKE AGENT 1 A FETCHINGAGENT
                    if len(real_agents)+1 == 1:
                        #print(f'Initializing FetchingAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = FetchingAgent(
                                arglist=arglist,
                                name='agent-'+str(len(real_agents)+1),
                                color=COLORS[len(real_agents)])
                        
                    else:
#                         print(f'Initializing agent {len(real_agents)+1} regularly at location ({loc[0]}, {loc[1]})')
#                         real_agent = RealAgent(
#                                 arglist=arglist,
#                                 name='agent-'+str(len(real_agents)+1),
#                                 id_color=COLORS[len(real_agents)],
#                                 recipes=recipes)

                        # MAKE AGENT 2 A "HUMAN"
                        #print(f'Initializing HybridAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = HybridAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    # ----------------------------------------
    
                    real_agents.append(real_agent)

    return real_agents

def initialize_agents_bl(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')

                    # ----------------------------------------
                    # ----------------------------------------
                    # KIRIN Added
#                     if len(real_agents)+1 == 1:
#                         print(f'Initializing agent {len(real_agents)+1} with no recipes at location ({loc[0]}, {loc[1]})')
#                         real_agent = RealAgent(
#                                 arglist=arglist,
#                                 name='agent-'+str(len(real_agents)+1),
#                                 id_color=COLORS[len(real_agents)],
#                                 recipes=[])
                    
                    # MAKE AGENT 1 A FETCHINGAGENT
                    if len(real_agents)+1 == 1:
                        # print(f'Initializing FetchingAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        # real_agent = FetchingAgent(
                        #         arglist=arglist,
                        #         name='agent-'+str(len(real_agents)+1),
                        #         color=COLORS[len(real_agents)])
                        #print(f'Initializing HybridAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = HybridAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                        
                    else:
#                         print(f'Initializing agent {len(real_agents)+1} regularly at location ({loc[0]}, {loc[1]})')
#                         real_agent = RealAgent(
#                                 arglist=arglist,
#                                 name='agent-'+str(len(real_agents)+1),
#                                 id_color=COLORS[len(real_agents)],
#                                 recipes=recipes)

                        # MAKE AGENT 2 A "HUMAN"
                        #print(f'Initializing HybridAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = HybridAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    # ----------------------------------------
    
                    real_agents.append(real_agent)

    return real_agents

def initialize_agents_test(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1

            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:

                simple_agent = SimpleAgent(
                    arglist=arglist,
                    name='agent-'+str(len(real_agents)+1),
                    id_color=COLORS[len(real_agents)],
                    recipes=recipes) 
                
                real_agents.append(simple_agent)  

                simple_agent = SimpleAgent(
                    arglist=arglist,
                    name='agent-'+str(len(real_agents)+1),
                    id_color=COLORS[len(real_agents)],
                    recipes=recipes) 
                
                real_agents.append(simple_agent)

                return real_agents



def main_loop(arglist):
    """The main loop for running experiments."""
    #print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    # game = GameVisualize(env)
    real_agents = initialize_agents(arglist=arglist)

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    while not env.done():
        action_dict = {}

        for agent in real_agents:
            action = agent.select_action(obs=obs)
            action_dict[agent.name] = action

        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        for agent in real_agents:
            # Only RealAgent needs to refresh subtasks
            if not isinstance(agent, FetchingAgent):
                agent.refresh_subtasks(world=env.world)

        # Saving info
        bag.add_status(cur_time=info['t'], real_agents=real_agents)


    # Saving final information before saving pkl file
    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)


def main_bl_loop(arglist):
    
    fix_seed(seed=arglist.seed)
    experiment = OvercookedBeliefExperiment(arglist)
    experiment.setup()

# -----------------------
# -----------------------
# KIRIN ADDED
def main_gr_experiments(arglist):
    """
    Main function for running goal recognition experiments with all agents using policies
    """
    # Fix random seed for reproducibility
    fix_seed(seed=arglist.seed)
    
    # Parse observation levels
    obs_levels = [float(x) for x in arglist.obs_levels.split(',')]
    
    # Create the experiment with the specified observation levels
    experiment = OvercookedGRExperiment(arglist, obs_levels=obs_levels)
    
    # Setup environment and recognizer
    experiment.setup()
    
    # Run the experiment with all agents using policies
    #print("Running experiments with all agents using policies")
    results = experiment.run(num_trials=arglist.num_trials)
    
    # Visualize both agents working together with the policy
    if arglist.record:
        print("\nVisualizing agent behavior with policy (recording enabled)")
    else:
        print("\nVisualizing agent behavior with policy")
        
#     # Run visualization for each recipe
#     for recipe_idx in range(len(experiment.recipes)):
#         print(f"\nVisualizing recipe {recipe_idx+1}/{len(experiment.recipes)}: {experiment.recipes[recipe_idx]}")
#         success = experiment.visualize_agent_behavior(
#             recipe_idx=recipe_idx,
#             max_steps=arglist.max_num_timesteps,
#             render_delay=0.2  # Adjust this for slower/faster visualization
#         )
#         if success:
#             print(f"Visualization for recipe {recipe_idx+1} completed successfully!")
#         else:
#             print(f"Visualization for recipe {recipe_idx+1} did not reach the goal within {arglist.max_num_timesteps} steps")
    
#     print("\nGoal Recognition experiments completed.")
#     return results
# -----------------------


if __name__ == '__main__':
    arglist = parse_arguments()
    
    # -----------------------
    # -----------------------
    # KIRIN ADDED
    if arglist.gr_experiments:
        #print("Running Goal Recognition experiments")
        main_gr_experiments(arglist)
    # -----------------------

    elif arglist.belief_experiments:
        #print("Running goal recognition using belief")
        main_bl_loop(arglist=arglist)

        
    elif arglist.play:
        #print("Running interactive play mode")
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()

    elif arglist.chef_test:
        agents = initialize_agents_test(arglist)
        print(agents)
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        state = env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents, agents=agents, env = env, state=state)
        game.on_execute()

    else:
        #print("Running regular Overcooked experiment")
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)




