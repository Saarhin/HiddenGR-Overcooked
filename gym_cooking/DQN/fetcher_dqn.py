from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from utils.agent import RealAgent, SimAgent, COLORS, FetchingAgent, HybridAgent, DQNFetchingAgent, SimpleAgent
from recipe_planner.recipe import *
from tqdm import tqdm
import re
from misc.metrics.metrics_bag import Bag
import os
import csv
#import psutil, os
import time
import cProfile
import pstats
# from collections import namedtuple
# from scalene import scalene_profiler
Transition = namedtuple(
    "Transition", ["state", "action", "next_state", "reward", "terminated"]
)

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network Layer
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQNTrainer:
    def __init__(self, fetcher_id, state_dim, action_dim, env, action_list, episodes, target_agent_name, arglist):
        self.env = env
        self.fetcher_id = fetcher_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_DQN = DQN(self.state_dim, 64, self.action_dim)
        self.target_DQN = DQN(self.state_dim, 64, self.action_dim)
        self.target_DQN.load_state_dict(self.policy_DQN.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_DQN.parameters(), lr=1e-3)
        self.action_list = action_list
        self.episodes = episodes
        self.target_agent_name = target_agent_name
        self.action_space = spaces.Discrete(len(self.action_list))
        self.loss_fn = nn.MSELoss()
        

        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.2
        self.target_update_freq = 100
        self.step_counter = 0
        self.epsilon_history = []
        self.x = 0
        self.y = 0
        self.realAgents = None
        self.arglist = arglist
        self.folder_name = f"policies_{self.arglist.dqn_input}_seed{self.arglist.seed}_fetcherDQN_simpleChef"

        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
       

    def initialize_agents(self):
        real_agents = []
        with open('utils/levels/{}.txt'.format(self.arglist.level), 'r') as f:
            phase = 1
            recipes = []
            count = 0
            for line in f:
                
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # phase 2: read in recipe list
                elif phase == 2:
                    recipes.append(globals()[line]())

                # phase 3: read in agent locations (up to num_agents)
                elif phase == 3:
                    loc = line.split(' ')

                    if count == 1:
                        # MAKE AGENT 2 A "HUMAN"
                        # print(f'Initializing HybridAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = SimpleAgent(
                            arglist=self.arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                        
                        real_agents.append(real_agent)
                        break

                    elif count == 0:
                        x = int(loc[0])
                        y = int(loc[1])
                        # print(f'Initializing DQNFetcherAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = DQNFetchingAgent(
                            arglist=self.arglist,
                            name='agent-'+str(len(real_agents)+1),
                            color=COLORS[len(real_agents)+1])
                        
                        real_agents.append(real_agent)
                        count +=1

        return real_agents, x, y

    def train(self):
            
            # scalene_profiler.start()
        # with cProfile.Profile() as profile:

            # might need to hardcode these
            num_states = self.state_dim
            num_actions = self.action_dim
            reward_per_episode = np.zeros(self.episodes)

            step_count = 0

            for i in tqdm(range(self.episodes)):
                self.realAgents, self.x, self.y=self.initialize_agents()
                
                terminated = False
                truncated = False
                sum_reward = 0

                target = "Water" if random.random()<0.5 else "Sushi"
                for agent in self.realAgents:
                        if agent.name == 'agent-2':
                            agent.target_item = target
                state = self.env.reset(target = target)


                while(not terminated and not truncated):
                    action_dict = {}

                    for agent in self.realAgents:
                        if agent.name == 'agent-1':
                            if self.arglist.dqn_input == "Full":
                                dqn_input, is_empty = self.state_to_dqn_input(self.env.rep)
                            elif self.arglist.dqn_input == "Summary":
                                dqn_input = self.get_agent_specific_state(state)
                                is_empty = False
                            action, action_save = agent.select_action(obs=state,env=self.env, epsilon=self.epsilon, policy=self.policy_DQN, dqn_input=dqn_input, is_empty=is_empty)
                        else:
                            action = agent.select_action(obs=state)
                            if action is None:
                                action = (0, 0)
                        action_dict[agent.name] = action
                        

                    new_state, reward, terminated, _ = self.env.step((action_dict, target))

                    sum_reward += reward

                    self.memory.append((state, action_save, new_state, reward, terminated))
                    for agent in self.realAgents:
                        # Only RealAgent needs to refresh subtasks
                        if not isinstance(agent, DQNFetchingAgent):
                            agent.refresh_subtasks(world=self.env.world)

                    state = new_state

                    step_count += 1

                reward_per_episode[i] = sum_reward
                
                if len(self.memory)>self.batch_size:
                    mini_batch = self.memory.sample(self.batch_size)
                    self.optimize(mini_batch, self.policy_DQN, self.target_DQN)
                    

                    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                    self.epsilon_history.append(self.epsilon)

                    if step_count > self.target_update_freq :
                        self.target_DQN.load_state_dict(self.policy_DQN.state_dict())
                        step_count=0

                # if i%5 == 0:
                #     proc = psutil.Process(os.getpid())
                #     rss = proc.memory_info().rss / 1024 ** 2  # in MB
                #     print(f"RAM usage: {rss:.2f} MB")

                if i%1000 == 0:
                    torch.save(self.policy_DQN.state_dict(), f"{self.folder_name}/fetcher_dqn{i}.pt")
                    with open(f"{self.folder_name}/rewards_{i}.csv", "w", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["episode", "reward"])  # optional header
                        for i, reward in enumerate(reward_per_episode):
                            writer.writerow([i, reward])
                

            self.env.close()

            with open(f"{self.folder_name}/rewards.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward"])  # optional header
                for i, reward in enumerate(reward_per_episode):
                    writer.writerow([i, reward])
        
        # results = pstats.Stats(profile)
        # results.sort_stats(pstats.SortKey.TIME)
        # results.print_stats()
        # results.dump_stats("08_commented_copy.prof")

            # scalene_profiler.stop()

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
  
    def get_agent_specific_state(self, env_state):
        """
        Create an agent-specific state representation to encourage role specialization.
        
        Args:
            env_state: The environment state object
            
        Returns:
            A state representation that includes agent-specific information
        """ 
        try:
            target_agent = next((a for a in env_state.sim_agents if a.name == self.target_agent_name), None)
            chef_agent = next((a for a in env_state.sim_agents if a.name != self.target_agent_name), None)

            dqn_state = []

            if target_agent and chef_agent:

                # location of the fetcher
                dqn_state.append(target_agent.location[0])
                dqn_state.append(target_agent.location[1])

                # location of the chef
                dqn_state.append(chef_agent.location[0])
                dqn_state.append(chef_agent.location[1])

                # water location
                water_found = False
                for obj_name, obj in env_state.world.objects.items():
                    if obj_name == "Plate-Water":
                        dqn_state.append(obj[0].location[0])
                        dqn_state.append(obj[0].location[1])
                        water_found = True
                        break

                if not water_found: 
                    dqn_state.append(-1)
                    dqn_state.append(-1)

                #Sushi location
                sushi_found = False
                for obj_name, obj in env_state.world.objects.items():
                    if obj_name == "Plate-Sushi":
                        dqn_state.append(obj[0].location[0])
                        dqn_state.append(obj[0].location[1])
                        sushi_found = True
                        break

                if not sushi_found: 
                    dqn_state.append(-1)
                    dqn_state.append(-1)

                # Delivery location
                delivery_found = False
                for obj_name, obj in env_state.world.objects.items():
                    if obj_name == "Delivery":
                        dqn_state.append(obj[0].location[0])
                        dqn_state.append(obj[0].location[1])
                        delivery_found = True
                        break

                if not delivery_found: 
                    dqn_state.append(-1)
                    dqn_state.append(-1)

                
                if "water" in target_agent.get_holding().lower():
                    dqn_state.append(1)
                elif "sushi" in target_agent.get_holding().lower():
                    dqn_state.append(2)
                else:
                    dqn_state.append(0)

                if "water" in chef_agent.get_holding().lower():
                    dqn_state.append(1)
                elif "sushi" in chef_agent.get_holding().lower():
                    dqn_state.append(2)
                else:
                    dqn_state.append(0)

                # length of this state is 12
               
                return torch.FloatTensor(dqn_state)
        except Exception as e:
            print(f"Error creating agent-specific state: {e}")
            
        # Fallback to global state
        return env_state.get_repr()
    
    def optimize(self, mini_batch, policy_DQN, target_DQN):
        # num_states = policy_DQN.fc1.in_features

        # current_q_list = []
        # target_q_list = []

        # for state, action, new_state, reward, terminated in mini_batch:

        #     if terminated:
        #         target = torch.FloatTensor([reward])
        #     else: 
        #         with torch.no_grad():
                    
        #             if self.arglist.dqn_input == "Full":
        #                 dqn_input, is_empty = self.state_to_dqn_input(self.env.rep)
        #             elif self.arglist.dqn_input == "Summary":
        #                 dqn_input = self.get_agent_specific_state(state)
        #                 is_empty = False
        #             target = torch.FloatTensor(
        #                 reward + self.discount_factor * target_DQN(dqn_input).max()
        #             )

        #     if self.arglist.dqn_input == "Full":
        #         dqn_input, is_empty = self.state_to_dqn_input(self.env.rep)
        #     elif self.arglist.dqn_input == "Summary":
        #         dqn_input = self.get_agent_specific_state(state)
        #         is_empty = False
        #     current_q = policy_DQN(dqn_input)
        #     current_q_list.append(current_q)

        #     target_q = target_DQN(dqn_input)
        #     target_q[action] = target
        #     target_q_list.append(target_q)

        # loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # vectorized
        batch = Transition(*zip(*mini_batch))

        if self.arglist.dqn_input == "Full":
            state_batch = torch.stack(
                [self.state_to_dqn_input(self.env.rep)[0]  # returns (x, is_empty)
                for _ in batch.state])
        else:  # "Summary"
            state_batch = torch.stack(
                [self.get_agent_specific_state(s) for s in batch.state])
            
        non_final_mask = torch.tensor(
        tuple(map(lambda d: not d, batch.terminated)),
        dtype=torch.bool)

        if self.arglist.dqn_input == "Full":
            next_state_batch = torch.stack(
                [self.state_to_dqn_input(self.env.rep)[0]
                for _ in batch.next_state])              
            
        else:
            next_state_batch = torch.stack(
                [self.get_agent_specific_state(s) for s in batch.next_state])
            
        action_batch  = torch.tensor(batch.action,  dtype=torch.long).unsqueeze(1)  # (B,1)
        reward_batch  = torch.tensor(batch.reward,  dtype=torch.float32)

        with torch.no_grad():
            # max_a Q_target(s',a)  for non-terminal samples
            next_q_values = torch.zeros(len(mini_batch))
            if non_final_mask.any():
                q_next_all = target_DQN(next_state_batch[non_final_mask])   # (N_non_final, A)
                next_q_values[non_final_mask] = q_next_all.max(dim=1).values

        q_targets = reward_batch + self.discount_factor * next_q_values     # (B,)

        # Current Q(s,a) for the actions actually taken
        q_policy_all = policy_DQN(state_batch)                              # (B, A)
        q_policy_sa  = q_policy_all.gather(1, action_batch).squeeze(1)      # (B,)

        # -------- 3. Loss & back-prop ----------
        loss = self.loss_fn(q_policy_sa, q_targets)                         # element-wise MSE

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

    def test(self):
        

        self.realAgents, self.x, self.y=self.initialize_agents()
       
        policy_DQN = DQN(self.state_dim, 64, self.action_dim)
        policy_DQN.load_state_dict(torch.load('fetcher_dqn.pt'))
        policy_DQN.eval()


        # target = "Water" if random.random()<0.5 else "Sushi"
        target = "Water"
        for agent in self.realAgents:
                if agent.name == 'agent-2':
                    agent.target_item = target

        state = self.env.reset(target)
        bag = Bag(arglist=self.arglist, filename="test")
        bag.set_recipe(recipe_subtasks=self.env.all_subtasks)
            
        terminated = False
        truncated = False


        while(not terminated and not truncated):
            action_dict = {}

            for agent in self.realAgents:
                if agent.name == 'agent-1':
                    action, action_save = agent.select_action(obs=state,env=self.env, epsilon=0, policy=policy_DQN)
                    # action = (0,0)
                else:
                    action = agent.select_action(obs=state)
                action_dict[agent.name] = action

            state, reward, terminated, info = self.env.step((action_dict, target))
           

            for agent in self.realAgents:
                # Only RealAgent needs to refresh subtasks
                if not isinstance(agent, DQNFetchingAgent):
                    agent.refresh_subtasks(world=self.env.world)
                
            bag.add_status(cur_time=info['t'], real_agents=self.realAgents)
        bag.set_collisions(collisions=self.env.collisions)
        bag.set_termination(termination_info=self.env.termination_info,
                successful=self.env.successful)

        self.env.close()