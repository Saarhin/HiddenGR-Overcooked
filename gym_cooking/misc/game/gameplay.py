# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime


class GamePlay(Game):
    def __init__(self, filename, world, sim_agents, agents=None, env=None, state=None):
        Game.__init__(self, world, sim_agents, play=True)
        self.filename = filename
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)
        
        self.sim_agents_store = sim_agents
        self.agents = agents
        self.env = env
        self.state = state


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                # print('just saved image {} to {}'.format(image_name, self.save_dir))
                return
            
            # Switch current agent
            if self.agents is None:
                if pygame.key.name(event.key) in "1234":
                    try:
                        self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                    except:
                        pass
                    return

                # Control current agent
                x, y = self.current_agent.location
                if event.key in KeyToTuple.keys():
                    action = KeyToTuple[event.key]
                    self.current_agent.action = action
                    interact(self.current_agent, self.world, self.sim_agents_store)
            else:
                # Control current agent
                action_dict={}
                self.current_agent = self.sim_agents[0]
                x, y = self.current_agent.location
                if event.key in KeyToTuple.keys():
                    action = KeyToTuple[event.key]
                    self.current_agent.action = action

                    action_dict[self.current_agent.name] = action
                    
                    
                    self.current_agent = self.sim_agents[1]
                    action = self.agents[1].select_action(self.state)
                    self.current_agent.action = action

                    action_dict[self.current_agent.name] = action

                    
                    # self.current_agent = self.sim_agents[0]
                    # interact(self.current_agent, self.world, self.sim_agents_store)
                    # self.current_agent = self.sim_agents[1]
                    # interact(self.current_agent, self.world, self.sim_agents_store)

                    self.state, reward, _, _ = self.env.step(action_dict, self.agents[1].target_item)
                    
                    self.agents[1].refresh_subtasks(world=self.env.world)
                    print(f"step reward: {reward}")


    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()


