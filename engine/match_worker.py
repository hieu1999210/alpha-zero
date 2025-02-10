import logging

import numpy as np
import torch
from torch.multiprocessing import Process

from utils import setup_worker_logger

from .mcts import MCTS


class MatchWorker(Process):
    """
    NOTE: currently support batch_size = 1 only

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
        self,
        pid,
        model1,
        model2,
        game,
        logging_queue,
        global_match_count,
        barrier,
        p1_wins,
        p2_wins,
        cfg,
        cfg2=None,
    ):
        assert not model1.training, "model1 is not in the eval mode"
        assert not model2.training, "model2 is not in the eval mode"
        assert model1.is_freezed, "model1 is not freezed"
        assert model2.is_freezed, "model1 is not freezed"

        super(MatchWorker, self).__init__()

        if cfg2 is None:
            cfg2 = cfg

        first_player = cfg.GAME.FIRST_PLAYER
        self.process_id = pid
        # player1 id=1, player2 id=-1

        # swap model for odd process
        swap = pid % 2
        self.swap = swap
        if swap:
            self.models_list = [None, model2, model1]
            self.win_count = [None, p2_wins, p1_wins]
            self.MCTSs = [None, MCTS(game, cfg2), MCTS(game, cfg)]
        else:
            self.models_list = [None, model1, model2]
            self.win_count = [None, p1_wins, p2_wins]
            self.MCTSs = [None, MCTS(game, cfg), MCTS(game, cfg2)]

        # fmt: off
        self.game                   = game
        self.logging_queue          = logging_queue
        self.global_match_count     = global_match_count
        self.barrier                = barrier
        self.total_num_matches      = cfg.MATCH.NUM_MATCHES
        self.n_simuls               = cfg.MCTS.NUM_SIMULATION_PER_STEP
        self.worker_match_count     = 0
        self.temp                   = cfg.MATCH.TEMP
        self.cfg                    = cfg
        self.cfg2                   = cfg2
        self.first_player           = first_player
        self.verbose_freq           = cfg.MATCH.VERBOSE_FREQ
        self.worker_name            = f"match_worker_{self.process_id:0>2}"
        self.verbose                = cfg.MATCH.VERBOSE
        
        # init for batch of players
        self.player                 = first_player
        self.steps_count            = 1
        self.canonical              = None
        # fmt: on

        self.state = game.getInitBoard()
        self.inp_tensor = (
            torch.zeros(size=(1, 1, *game.getBoardSize())).pin_memory().cuda()
        )

    def run(self):
        setup_worker_logger(self.logging_queue)
        if self.verbose:
            logging.info(f"{self.worker_name}: started")

        while self.global_match_count.value < self.total_num_matches:
            self._gen_canonicals()
            for _ in range(self.n_simuls):
                self._run_simulation()
            self._transition()
        if self.verbose:
            logging.info(
                f"{self.worker_name}: finished, "
                f"runned {self.worker_match_count} matches"
            )
        self.barrier.wait()

    def _gen_canonicals(self):
        # for i in range(self.batch_size):
        self.canonical = self.game.getCanonicalForm(self.state, self.player)

    def _run_simulation(self):
        """
        run a single MCTS simulation step:
            - search leaf nodes
            - predict
            - backward
        """
        # search leaf nodes
        player = self.player
        leaf_node = self.MCTSs[player].search(self.canonical)
        if leaf_node is not None:
            self.inp_tensor[0][0] = torch.from_numpy(leaf_node)

        # predict policy, value for leaf nodes
        with torch.no_grad():
            policies, values = self.models_list[player](self.inp_tensor)
            policies = policies.detach().cpu().numpy()
            values = values.detach().cpu().numpy()

        # update tree
        self.MCTSs[player].process_result(policies[0], values[0])

    def _transition(self):
        """
        take action and get next state
        """

        player = self.player
        policy = self.MCTSs[player].get_policy(self.canonical, self.temp)
        action = np.argmax(policy)

        # get next state
        next_state, next_player = self.game.getNextState(self.state, player, action)

        winner = self.game.getGameEnded(next_state, next_player) * next_player

        # game is not end
        if winner == 0:
            self.steps_count += 1
            self.state = next_state
            self.player = next_player
        # end game
        else:
            with self.global_match_count.get_lock():
                self.global_match_count.value += 1
                match_count = self.global_match_count.value
                if match_count % self.verbose_freq == 0:
                    logging.info(
                        f"{self.worker_name}: "
                        f"{match_count:0>3}-th game: "
                        f"tree1_size: {len(self.MCTSs[1])}; "
                        f"tree2_size: {len(self.MCTSs[-1])};"
                        f"num_steps: {self.steps_count}; "
                        f"winner: {winner}"
                    )
                    logging.info(
                        f"{self.worker_name}: "
                        f"player1 vs player2: {self.win_count[1].value:0>3} - "
                        f"{self.win_count[-1].value:0>3}"
                    )
            with self.win_count[winner].get_lock():
                self.win_count[winner].value += 1
            self.worker_match_count += 1
            self._reset_game()

    def _reset_game(self):
        """
        reset atributes for a new game

        NOTE:
            need not to reset canonical board and input_tensor
        """
        if self.swap:
            self.MCTSs = [None, MCTS(self.game, self.cfg2), MCTS(self.game, self.cfg)]
        else:
            self.MCTSs = [None, MCTS(self.game, self.cfg), MCTS(self.game, self.cfg2)]
        self.state = self.game.getInitBoard()
        self.player = self.first_player
        self.steps_count = 1
