import logging
import math

import numpy as np
import torch

np.seterr(over="raise")


class MCTS:
    """
    In this implementation, a MCTS simulation involves 2 steps:
        i.  search:
            go to a leaf node and query prediction from model
        ii. process_result:
            update info of nodes in the search path from the given predicted results
    NOTE: all states given to MCTS must be in canonical form (e.i view of a chosen player only)

    attributes:
        game
        Ns (dict(s: int)): count visit
        Qsa (dict((s,a): q value)): Q value
        Ps (dict(s: np.array)): policy given by model
        v_moves (dict(s: np.array)): valid move of a state
        Es (dict(s: end_game_value)): end_game value of a state w.r.t a player
        path: path in a search
        _is_terminate: show whether current processing leaf node is terminate state
    """

    def __init__(self, game, cfg, model=None, debug=False):
        # fmt: off
        self.game           = game
        self.Es             = {}
        self.Ns             = {}
        self.Nsa            = {}
        self.Qsa            = {}
        self.v_moves        = {}
        self.Ps             = {}
        self.path           = []
        self.c_puct         = cfg.MCTS.C_PUCT
        self._is_terminate  = False
        self._end_value     = 0
        self.cfg            = cfg
        self.default_player = cfg.GAME.DEFAULT_PLAYER
        self.debug          = debug
        self.model          = model
        self.device         = cfg.DEVICE
        # fmt: on

    def __len__(self):
        return len(self.Ps)

    def get_policy_infer(self, state, temp):
        """
        get policy in inference mode
        """
        assert self.model

        n_simuls = self.cfg.MCTS.NUM_SIMULATION_PER_STEP

        # run simulation
        for _ in range(n_simuls):
            # search
            new_state = self.search(state)

            # evaluate new state
            if new_state is None:
                p, v = None, None
            else:
                with torch.no_grad():
                    h, w = new_state.shape
                    input_tensor = (
                        torch.from_numpy(new_state)
                        .float()
                        .view(1, 1, h, w)
                        .to(self.device)
                    )
                    policies, values = self.model(input_tensor)
                    p = policies.detach().cpu().numpy()[0]
                    v = values.detach().cpu().numpy()[0]

            # expand and backprop
            self.process_result(p, v)

        return self.get_policy(state, temp)

    def get_policy(self, state, temp):
        """
        return policy from simulation results

        args:
            --state(np.array): canonical board
            --temp(float): temperature that control level of exploration
        return:
            --policy (np.array)
        NOTE: this does not run any simulation
        """
        state_str = self.game.stringRepresentation(state)
        counts = [
            self.Nsa[(state_str, a)] if (state_str, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]
        prob = np.array(counts, dtype=np.float32)
        if temp == 0:
            best_act = np.argmax(prob)
            prob = np.zeros_like(prob)
            prob[best_act] = 1
            return prob

        try:
            prob = np.power(prob, 1.0 / temp)
            prob = prob / prob.sum()

            return prob
        except FloatingPointError as e:
            if "overflow" in str(e):
                best_act = np.argmax(prob)
                prob = np.zeros_like(prob)
                prob[best_act] = 1
                return prob
            else:
                print(e)
                exit()

    def search(self, state):
        """
        this a the first step of a simulation
        search from given state guided by UCB to a leaf-node (new state or terminate state)

        args:
            state (np.array) cannonical state
        return:
            np.array of current state if it's not terminate state
            None if current state is a terminate one

        NOTE:
            --attribute _is_terminate is modified according to result
            --attribute path must be empty before the call, and be updated by the search
            --no others attributes are modified EXCEPT for Es

            after the call, the path includes the leaf or the terminal state
        """

        assert len(self.path) == 0, "cached path is not cleared"
        assert not self._is_terminate, "attrib _is_terminate is not cleared"

        # search loop
        state_str = self.game.stringRepresentation(state)
        while state_str in self.Ps:
            a = self._get_best_action(state_str)
            self.path.append((state_str, a))

            state, player = self.game.getNextState(state, self.default_player, a)
            state = self.game.getCanonicalForm(state, player)
            state_str = self.game.stringRepresentation(state)

        # append the new state / terminate state
        self.path.append((state_str, None))

        # get end game value, get 0 if the game is not ended
        if state_str not in self.Es:
            end = self.game.getGameEnded(state, self.default_player)
            self.Es[state_str] = end
        else:
            end = self.Es[state_str]

        if self.debug:
            print(self.path, f"\nwinner: {end}\n")
        # if terminal state
        if end:
            self._end_value = end
            self._is_terminate = True
            # print(" end game", end)
            return None

        # if non-terminal state
        # get update validmove for the leaf node
        self.v_moves[state_str] = self.game.getValidMoves(state, self.default_player)

        return state

    def process_result(self, p, v):
        """
        process predict

        termimal state is not added to the tree but st
        """

        # process the final state of the path
        final_s = self.path.pop()[0]

        # if non-terminal leaf, add to the tree
        if not self._is_terminate:
            self._add_state(final_s, p)
        # if terminal state
        else:
            v = self._end_value

        # update properties of state along the path
        path = self.path[::-1]

        for s_a in path:
            v = -v
            q = self.Qsa[s_a]
            n = self.Nsa[s_a]
            self.Qsa[s_a] = (n * q + v) / (n + 1)
            self.Nsa[s_a] = n + 1
            self.Ns[s_a[0]] += 1

        # reset for next simulation
        self._is_terminate = False
        self.path = []

    def _get_best_action(self, state_str):
        """
        get best action according to UCB
        args:
            state_str(str): string representation of canonical board
        NOTE: some asumptions:
            state is already in the tree as well as all of its properties
        """
        assert state_str in self.Ps, "state's not in the tree"

        v_moves = self.v_moves[state_str]
        current_best_v = -float("inf")
        current_best_act = -1
        for a in range(self.game.getActionSize()):
            if v_moves[a]:
                Qsa = self.Qsa[(state_str, a)]
                nsa = self.Nsa[(state_str, a)]

                # q plus u
                q_u = Qsa + self.c_puct * self.Ps[state_str][a] * math.sqrt(
                    self.Ns[state_str]
                ) / (1 + nsa)

                if q_u > current_best_v:
                    current_best_v = q_u
                    current_best_act = a

        return current_best_act

    def _add_state(self, state_str, policy):
        """
        add policy and other attribute of new state to tree

        args:
            --state_str(str): string representation of canonical board
            --policy(np.array): policy return by the model
        NOTE: policy is added dirichlet noise to enforce exploration
        """
        v_moves = self.v_moves[state_str]
        # print("count valid move", v_moves.sum())
        # print(policy)
        # normalize policy
        pi = policy * v_moves
        sum_valid_policy = pi.sum()
        if sum_valid_policy <= 0:
            logging.warning("got zero for all valid move")
            # print("got zero for all valid move")
            pi += v_moves
            sum_valid_policy = pi.sum()
        pi /= sum_valid_policy

        # gen dirichlet noise
        alpha = self.cfg.MCTS.DIRICHLET_ALPHA
        weight = self.cfg.MCTS.DIRICHLET_WEIGHT
        num_action = v_moves.sum()
        v_moves = v_moves.astype(np.bool)
        dirichlet = np.random.dirichlet(alpha * np.ones((num_action,)))

        # add dirichlet noise
        pi[v_moves] = (1 - weight) * pi[v_moves] + weight * dirichlet

        # add state attribute to tree
        self.Ps[state_str] = pi
        self.Ns[state_str] = 0
        for i in range(len(pi)):
            if v_moves[i]:
                self.Qsa[(state_str, i)] = 0
                self.Nsa[(state_str, i)] = 0
