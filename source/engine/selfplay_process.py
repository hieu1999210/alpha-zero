from torch.multiprocessing import Process

class SelfplayProcess(Process):
    """
    a Self play process concurrently running a batch of selfplay Agent, 
    this is to leverage parallelism of the gpu
    
    
    MAIN FLOW:
    while num_played_games < game_per_iter:
        for i in num_simulation:
            run_simulation
        take_action
        get_next_state (if game terminate, init new game)
    
    attributes:
        MCTSs (list(MCTS)): Monte Carlo tree search for each Agent
        input_tensor (torch.Tensor): store states to predict policy and value
        
    """
    def __init__(self):
        super(SelfplayProcess, self).__init__()
    
    def run(self):
        
    def _run_simulation(self):
        """
        run a single MCTS simulation step
        """
        pass
    
    def _transition(self):
        """
        take action and get next state
        """
        
        
    
    

