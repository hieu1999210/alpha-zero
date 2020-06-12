import logging

from torch.multiprocessing import Process
import torch
import numpy as np

from .mcts import MCTS
from utils import setup_worker_logger

class Selfplay(Process):
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
    def __init__(
        self, pid, model, game, exp_queue, logging_queue, global_games_count,
        barrier, cfg, p1_wins, p2_wins,
    ):
        assert not model.training, "model is not in the eval mode"
        assert model.is_freezed, "model is not freezed"
        
        super(Selfplay, self).__init__()
        batch_size = cfg.SELF_PLAY.BATCH_SIZE
        default_player = cfg.GAME.DEFAULT_PLAYER
        
        self.process_id             = pid
        self.model                  = model
        self.game                   = game
        self.exp_queue              = exp_queue
        self.logging_queue          = logging_queue
        self.global_games_count     = global_games_count
        self.barrier                = barrier
        self.total_num_games        = cfg.SELF_PLAY.GAME_PER_ITER
        self.n_simuls               = cfg.MCTS.NUM_SIMULATION_PER_STEP
        self.worker_games_count     = 0
        self.temp_thresh            = cfg.SELF_PLAY.TEMP_THRESH
        self.cfg                    = cfg
        self.batch_size             = batch_size
        self.default_player         = default_player
        self.win_count              = [None, p1_wins, p2_wins]
        self.verbose_freq           = cfg.SELF_PLAY.VERBOSE_FREQ
        self.worker_name            = f"self_worker_{self.process_id:0>2}"
        self.verbose                = cfg.SELF_PLAY.VERBOSE
        
        # init for batch of players
        self.players                = [default_player]*batch_size
        self.steps_count            = [1]*batch_size
        self.canonicals             = [None]*batch_size
        self.histories              = [[] for _ in range(batch_size)]
        self.MCTSs = [MCTS(game, cfg) for _ in range(batch_size)]
        self.states = [game.getInitBoard() for _ in range(batch_size)]
        self.inp_tensor = torch.zeros(
            size=(batch_size, 1, *game.getBoardSize())
        ).pin_memory().cuda()

    def run(self):
        setup_worker_logger(self.logging_queue)
        if self.verbose:
            logging.info(f"{self.worker_name}: started")
        
        while self.global_games_count.value < self.total_num_games:
            self._gen_canonicals()
            for _ in range(self.n_simuls):
                self._run_simulation()
            self._transition()
        
        if self.verbose:
            logging.info(
                f"{self.worker_name}: finished, "
                f"runned {self.worker_games_count} games")
        # torch.cuda.empty_cache()
        self.barrier.wait()
    
    def _gen_canonicals(self):
        for i in range(self.batch_size):
            self.canonicals[i] = self.game.getCanonicalForm(
                self.states[i], self.players[i]
            )
    
    def _run_simulation(self):
        """
        run a single MCTS simulation step:
            - search leaf nodes
            - predict
            - backward
        """
        # search leaf nodes
        for i in range(self.batch_size):
            leaf_node = self.MCTSs[i].search(self.canonicals[i])
            if leaf_node is not None:
                self.inp_tensor[i][0] = torch.from_numpy(leaf_node)
        
        # predict policy, value for leaf nodes
        with torch.no_grad():
            policies, values = self.model(self.inp_tensor)
            policies = policies.detach().cpu().numpy()
            values = values.detach().cpu().numpy()
        
        # update tree
        for i in range(self.batch_size):
            self.MCTSs[i].process_result(policies[i], values[i])

    def _transition(self):
        """
        take action and get next state
        """
        
        for i in range(self.batch_size):
            # take action
            temp = int(self.steps_count[i] < self.temp_thresh)
            policy = self.MCTSs[i].get_policy(self.canonicals[i], temp)
            action = np.random.choice(len(policy), p=policy)
            
            # update history
            self.histories[i].append((
                self.canonicals[i], self.players[i], policy
            ))
            
            # get next state
            next_state, next_player = self.game.getNextState(
                self.states[i], self.players[i], action
            )
            
            winner = self.game.getGameEnded(next_state, next_player)*next_player
            
            # game is not end
            if winner == 0:
                self.steps_count[i] += 1
                self.states[i] = next_state
                self.players[i] = next_player
            # end game
            else:
                with self.global_games_count.get_lock():
                    self.global_games_count.value += 1
                    game_count = self.global_games_count.value
                    if game_count % self.verbose_freq == 0:
                        logging.info(
                            f"{self.worker_name}: "
                            f"{game_count:0>3}-th game: "
                            f"tree_size: {len(self.MCTSs[i])}; "
                            f"num_steps: {self.steps_count[i]}; "
                            f"winner: {winner}"
                        )
                        logging.info(
                            f"{self.worker_name}: "
                            "player1 vs player2: "
                            f"{self.win_count[1].value:0>3} - "
                            f"{self.win_count[-1].value:0>3}")
                win_count = self.win_count[winner]
                with win_count.get_lock():
                    win_count.value +=1
                self.worker_games_count += 1
                self._process_history(i, winner)
                self._reset_game(i)
    
    def _process_history(self, game_id, winner):
        """
        put history to exp_queue
        an item is a tuple (board, policy, value)
        """
        for hist in self.histories[game_id]:
            v = 1 if winner == hist[1] else -1
            self.exp_queue.put((hist[0], hist[2], v))
    
    def _reset_game(self, game_id):
        """
        reset atributes for a new game
        
        NOTE:
            need not to reset canonical board and input_tensor
        """
        self.MCTSs[game_id]         = MCTS(self.game, self.cfg)
        self.states[game_id]        = self.game.getInitBoard()
        self.players[game_id]       = self.default_player
        self.steps_count[game_id]   = 1
        self.histories[game_id]     = []


