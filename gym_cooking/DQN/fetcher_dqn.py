from torch import nn
import torch.nn.functional as F
from collections import deque
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from utils.agent import RealAgent, SimAgent, COLORS, FetchingAgent, HybridAgent, DQNFetchingAgent
from recipe_planner.recipe import *
from tqdm import tqdm
import re
from misc.metrics.metrics_bag import Bag

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
    
class FetcherDQNTrainer:
    def __init__(self, fetcher_id, state_dim, action_dim, arglist):
        self.env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        self.fetcher_id = fetcher_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_DQN = DQN(self.state_dim, 64, self.action_dim)
        self.target_DQN = DQN(self.state_dim, 64, self.action_dim)
        self.target_DQN.load_state_dict(self.policy_DQN.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_DQN.parameters(), lr=1e-3)
        self.arglist = arglist
        self.action_map = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_space = spaces.Discrete(len(self.action_map))
        self.loss_fn = nn.MSELoss()
        

        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.05
        self.target_update_freq = 20
        self.step_counter = 0
        self.epsilon_history = []
        self.x = 0
        self.y = 0
        self.realAgents = None

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
                        print(f'Initializing HybridAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = HybridAgent(
                            arglist=self.arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                        
                        real_agents.append(real_agent)
                        break

                    elif count == 0:
                        x = int(loc[0])
                        y = int(loc[1])
                        print(f'Initializing DQNFetcherAgent {len(real_agents)+1} at location ({loc[0]}, {loc[1]})')
                        real_agent = DQNFetchingAgent(
                            arglist=self.arglist,
                            name='agent-'+str(len(real_agents)+1),
                            color=COLORS[len(real_agents)+1])
                        
                        real_agents.append(real_agent)
                        count +=1

        return real_agents, x, y

    def train(self, episodes):

        self.realAgents, self.x, self.y=self.initialize_agents()

        # might need to hardcode these
        num_states = self.state_dim
        num_actions = self.action_dim
        reward_per_episode = np.zeros(episodes)

        step_count = 0

        for i in tqdm(range(episodes)):
            
            terminated = False
            truncated = False
            sum_reward = 0

            target = "Water" if random.random()<0.5 else "Sushi"
            for agent in self.realAgents:
                    if agent.name == 'agent-2':
                        agent.target_item = target
            state = self.env.reset(target)


            while(not terminated and not truncated):
                action_dict = {}

                for agent in self.realAgents:
                    if agent.name == 'agent-1':
                        action, action_save = agent.select_action(obs=state,env=self.env, epsilon=self.epsilon, policy=self.policy_DQN)
                    else:
                        action = agent.select_action(obs=state)
                    action_dict[agent.name] = action

                new_state, reward, terminated, _ = self.env.step(action_dict, target)

                sum_reward += reward

                self.memory.append((state, action_save, new_state, reward, terminated))
                for agent in self.realAgents:
                    # Only RealAgent needs to refresh subtasks
                    if not isinstance(agent, DQNFetchingAgent):
                        agent.refresh_subtasks(world=self.env.world)

                state = new_state

                step_count += 1

            reward_per_episode[i] = sum_reward

            if len(self.memory)>self.batch_size and np.sum(reward_per_episode)>0:
                mini_batch = self.memory.sample(self.batch_size)
                self.optimize(mini_batch, self.policy_DQN, self.target_DQN)
                

                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
                self.epsilon_history.append(self.epsilon)

                if step_count > self.target_update_freq :
                    self.target_DQN.load_state_dict(self.policy_DQN.state_dict())
                    step_count=0

        self.env.close()

        torch.save(self.policy_DQN.state_dict(), "fetcher_dqn.pt")

        plt.figure(1)

        

        plt.subplot(121)
        plt.plot(reward_per_episode)

        plt.subplot(122)
        plt.plot(self.epsilon_history)

        plt.savefig('frozen_lake_dqn.png')

    def strip_ansi(self,text):
        ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
        return ansi_escape.sub('', text)
    
    def state_to_dqn_input(self, state, num_states:int) -> torch.Tensor:
        
        clean_grid = [[self.strip_ansi(cell) for cell in row] for row in self.env.rep]
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

    def optimize(self, mini_batch, policy_DQN, target_DQN):
        num_states = policy_DQN.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target = torch.FloatTensor([reward])
            else: 
                with torch.no_grad():
                    
                    dqn_input, is_empty = self.state_to_dqn_input(new_state, num_states)
                    target = torch.FloatTensor(
                        reward + self.discount_factor * target_DQN(dqn_input).max()
                    )

            dqn_input, is_empty = self.state_to_dqn_input(state, num_states)
            current_q = policy_DQN(dqn_input)
            current_q_list.append(current_q)

            dqn_input, is_empty = self.state_to_dqn_input(state, num_states)
            target_q = target_DQN(dqn_input)
            target_q[action] = target
            target_q_list.append(target_q)

        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes):
        

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

            state, reward, terminated, info = self.env.step(action_dict, target)
           

            for agent in self.realAgents:
                # Only RealAgent needs to refresh subtasks
                if not isinstance(agent, DQNFetchingAgent):
                    agent.refresh_subtasks(world=self.env.world)
                
            bag.add_status(cur_time=info['t'], real_agents=self.realAgents)
        bag.set_collisions(collisions=self.env.collisions)
        bag.set_termination(termination_info=self.env.termination_info,
                successful=self.env.successful)

        self.env.close()