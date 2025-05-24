from DQN.fetcher_dqn import FetcherDQNTrainer

class Belief():
    def __init__(self, arlglist):
        # actions = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        fetcher = FetcherDQNTrainer(fetcher_id=1, state_dim=(9*9*6), action_dim=5, arglist=arlglist)
        fetcher.train(100)
        # fetcher.test(4)

    
