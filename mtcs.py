from renju import b2s, get_action_mask, get_next_state, get_valid_actions
import numpy as np
import math
import random
import renju


class PVmtcs:
    def __init__(self, network, alpha, c_puct=1.0, epsilon=0.25):
        self.network = network
        self.alpha = alpha
        self.c_puct = c_puct
        self.eps = epsilon

        self.P = {}
        self.N = {}
        self.W = {}
        self.Done = {}
        self.IsWin = {}

        self.next_states = {}
        self.network.eval()

    def search(self, root_state, num_sims):
        s = b2s(root_state)

        if s not in self.P:
            _ = self.expand(root_state)

        valid_actions = get_valid_actions(root_state)

        #: root状態にだけは事前確立にディリクレノイズをのせて探索を促進する
        dirichlet_noise = np.random.dirichlet(alpha=[self.alpha] * len(valid_actions))
        for a, noise in zip(valid_actions, dirichlet_noise):
            self.P[s][a] = (1 - self.eps) * self.P[s][a] + self.eps * noise

        #: MCTS simulationの実行
        for _ in range(num_sims):

            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(renju.ACTION_SPACE)]
            Q = [w / n if n != 0 else 0 for w, n in zip(self.W[s], self.N[s])]

            # print(self.P[s], self.N[s], self.W[s])

            #: PUCTスコアの算出
            scores = [u + q for u, q in zip(U, Q)]
            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            #: スコアのもっとも高いactionを選択
            if len(scores) == 0:
                print(self.P[s], self.U, self.P, self.next_states[s])
            action = random.choice(np.where(scores == scores.max())[0])
            next_state = self.next_states[s][action]

            #: 選択した行動を評価（次は相手番なので評価値に-1をかける）
            v = -self.evaluate(next_state)

            self.W[s][action] += v
            self.N[s][action] += 1

        #: mcts_policyは全試行回数に占める各アクションの試行回数の割合
        mcts_policy = [n / sum(self.N[s]) for n in self.N[s]]

        return mcts_policy

    def expand(self, state):
        s = b2s(state)
        action_mask = get_action_mask(state).ravel()
        policy, value = self.network.predict(state, action_mask)
        self.P[s] = policy.detach().numpy()
        self.N[s] = [0] * renju.ACTION_SPACE
        self.W[s] = [0] * renju.ACTION_SPACE

        actions = np.where(action_mask > 0)[0]
        next_states = {}
        for action in actions:
            next_state, done, win = get_next_state(state, action)
            next_states[action] = next_state
            _s = b2s(next_state)
            self.Done[_s] = done
            self.IsWin[_s] = win
        self.next_states[s] = next_states
        # print(self.next_states)
        # print(value.shape)
        return value.detach().numpy()[0]

    def evaluate(self, state):
        s = b2s(state)

        if self.Done[s]:
            if self.IsWin[s]:
                return 1.0
            return 0
        elif s not in self.P:
            value = self.expand(state)
            return value
        else:
            U = [self.c_puct * self.P[s][a] * math.sqrt(sum(self.N[s])) / (1 + self.N[s][a])
                 for a in range(renju.ACTION_SPACE)]
            Q = [q / n if n != 0 else q for q, n in zip(self.W[s], self.N[s])]
            valid_actions = get_valid_actions(state)
            scores = [u + q for u, q in zip(U, Q)]

            scores = np.array([score if action in valid_actions else -np.inf
                               for action, score in enumerate(scores)])

            best_action = random.choice(np.where(scores == scores.max())[0])
            # if best_action not in self.next_states[s].keys():
            #     print(self.next_states[s], valid_actions)
            next_state = self.next_states[s][best_action]

            v = -self.evaluate(next_state)

            self.W[s][best_action] += v
            self.N[s][best_action] += 1

            return v
