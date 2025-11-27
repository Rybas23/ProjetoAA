import random
from simulador.agente_base import AgenteBase

class QAgent(AgenteBase):
    def __init__(self, id, actions, alpha=0.5, gamma=0.9, epsilon=0.1, modo='learn'):
        super().__init__(id, modo)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}
        self.last_state = None
        self.last_action = None

    @classmethod
    def cria(cls, params_file):
        return cls('qagent', actions=['UP','DOWN','LEFT','RIGHT','PICK','DROP','STAY'])

    def _to_state(self, obs):
        if 'dir' in obs:
            return tuple(obs['dir'])
        if 'neigh' in obs:
            return (obs['pos'], tuple(sorted(obs['neigh'].items())), obs['carrying'])
        return str(obs)

    def observacao(self, obs):
        self.cur_obs = obs

    def age(self):
        state = self._to_state(self.cur_obs)
        self.last_state = state
        if random.random() < self.epsilon and self.modo=='learn':
            action = random.choice(self.actions)
        else:
            action = self._best_action(state)
        self.last_action = action
        return action

    def _best_action(self, state):
        self.q.setdefault(state, {a:0.0 for a in self.actions})
        best = max(self.q[state].items(), key=lambda x: x[1])[0]
        return best

    def avaliacaoEstadoAtual(self, recompensa):
        if self.modo!='learn':
            return
        prev = self.last_state
        action = self.last_action
        if prev is None or action is None:
            return
        self.q.setdefault(prev, {a:0.0 for a in self.actions})
        next_state = self._to_state(self.cur_obs)
        self.q.setdefault(next_state, {a:0.0 for a in self.actions})
        max_next = max(self.q[next_state].values())
        self.q[prev][action] += self.alpha * (recompensa + self.gamma * max_next - self.q[prev][action])

    def reset(self, ep):
        if self.modo=='learn':
            self.epsilon = max(0.01, self.epsilon * 0.995)