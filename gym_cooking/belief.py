import os
import gym

from overcooked_fetcher_DQN import OvercookedFetcherDQN

class OvercookedBeliefExperiment():
    def __init__(self, arlglist):

        self.arglist = arlglist
        self.env = None
        self.fetcher = None
        self.recipes = []

        # Create output directory
        os.makedirs("results", exist_ok=True)

        if self.arglist.record:
            print("Recording enabled in experiment initialization")
        # actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        # fetcher = FetcherDQNTrainer(fetcher_id=1, state_dim=(9*9*6), action_dim=5, arglist=arlglist)
        # fetcher.train(100)
        # fetcher.test(4)

    def setup(self):
        print("Setting up experiment environment...")
        self.env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist = self.arglist) 

        # Verify record setting is properly applied
        if self.arglist.record:
            print("Record flag confirmed in environment setup")

        
        self.fetcher = OvercookedFetcherDQN()

        # training DQN
        self.fetcher.train_dqn(env = self.env, arglist=self.arglist )


