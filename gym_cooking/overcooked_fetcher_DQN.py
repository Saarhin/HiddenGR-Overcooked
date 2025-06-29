from DQN.fetcher_dqn import DQNTrainer


class OvercookedFetcherDQN:
    def __init__(self, method=DQNTrainer):
        self.method = method
        self.recipe_list = []
        self.actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    def train_dqn(self, env, arglist):
        if arglist.dqn_input == "Full":
            state_dim = (9*9*6)
        elif arglist.dqn_input == "Summary":
            state_dim = 12


        policy = DQNTrainer(
            fetcher_id=1, 
            state_dim = state_dim, 
            action_dim=5,
            env=env,
            action_list= self.actions,
            episodes=1000000,
            target_agent_name="agent-1",
            arglist=arglist)
        
        policy.train()
