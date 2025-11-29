# agentes module
import random
from enum import Enum

class AgenteBase:
    def __init__(self,id,modo='test'):
        self.id=id; self.modo=modo; self.ambiente=None
        self.sensores=[]; self.verbose=False
        self.ultima_observacao=None
    @classmethod
    def cria(cls,p): raise NotImplementedError
    def instala_ambiente(self,a): self.ambiente=a
    def instala(self,s): self.sensores.append(s)
    def observacao(self,o):
        self.ultima_observacao=o
        if self.verbose: print(f"[{self.id}] obs: {o}")
    def comunica(self,m,ag):
        if self.verbose: print(f"[{self.id}] msg {m} de {ag.id}")
    def age(self): raise NotImplementedError
    def avaliacaoEstadoAtual(self,r): pass
    def reset(self,ep): pass

class FixedAgent(AgenteBase):
    def __init__(self,id,policy,modo='test'):
        super().__init__(id,modo); self.policy=policy
    @classmethod
    def cria(cls,p): return cls('fixed',lambda o:'STAY')
    def age(self): return self.policy(self.ultima_observacao or {})

#########################################
# QAGENT BASE – Q-learning comum
#########################################

class QAgentBase(AgenteBase):
    def __init__(self, id, actions,
                 alpha=0.4, gamma=0.95,
                 epsilon=0.2, modo='learn'):

        super().__init__(id, modo)

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q = {}               # Q-table
        self.last_state = None
        self.last_action = None

    # -------------------------------------------------------------
    # ESCOLHA DA AÇÃO
    # -------------------------------------------------------------
    def age(self):
        s = self._to_state(self.ultima_observacao)

        if s not in self.q:
            self.q[s] = {a: 0.0 for a in self.actions}

        if self.modo == 'learn' and random.random() < self.epsilon:
            a = random.choice(self.actions)
        else:
            max_q = max(self.q[s].values())
            best_actions = [a for a, v in self.q[s].items() if v == max_q]
            a = random.choice(best_actions)

        self.last_state = s
        self.last_action = a
        return a

    # -------------------------------------------------------------
    # Q-LEARNING UPDATE
    # -------------------------------------------------------------
    def avaliacaoEstadoAtual(self, r):
        if self.modo != 'learn':
            return

        ps = self.last_state
        a = self.last_action
        if ps is None or a is None:
            return

        ns = self._to_state(self.ultima_observacao)

        if ns not in self.q:
            self.q[ns] = {aa: 0.0 for aa in self.actions}

        max_next = max(self.q[ns].values())

        self.q[ps][a] += self.alpha * (
            r + self.gamma * max_next - self.q[ps][a]
        )

    # -------------------------------------------------------------
    # EPSILON DECAY POR EPISÓDIO
    # -------------------------------------------------------------
    def reset(self, ep):
        if self.modo == 'learn':
            self.epsilon = max(0.05, self.epsilon * 0.995)

#########################################
# QAGENT FAROL – Ambiente FarolEnv
#########################################

class QAgentFarol(QAgentBase):

    @classmethod
    def cria(cls, p):
        # APENAS AÇÕES VÁLIDAS NO FAROL!
        return cls('q_farol',
                   ['UP','DOWN','LEFT','RIGHT','STAY'])

    # -------------------------------------------------------------
    # ESTADO: direção relativa quantizada (dx, dy)
    # Marcoviano e ótimo para Q-learning
    # -------------------------------------------------------------
    def _to_state(self, o):
        if not o:
            return ('NONE',)

        if 'direcao_farol' in o and 'pos' in o:
            (xa, ya) = o['pos']
            xf, yf = self.ambiente.farol  # FarolEnv já guarda isto

            dx = xf - xa
            dy = yf - ya

            dx = 1 if dx > 0 else -1 if dx < 0 else 0
            dy = 1 if dy > 0 else -1 if dy < 0 else 0

            return ('FAROL', dx, dy)

        return ('GEN', str(o))


#########################################
# QAGENT FORAGING – Ambiente ForagingEnv
#########################################

class QAgentForaging(QAgentBase):

    @classmethod
    def cria(cls, p):
        return cls('q_forage',
                   ['UP','DOWN','LEFT','RIGHT','PICK','DROP','STAY'])

    # -------------------------------------------------------------
    # ESTADO: visão binária + carrying
    # Filtra corretamente Foraging vs Farol
    # -------------------------------------------------------------
    def _to_state(self, o):
        if not o:
            return ('NONE',)

        if 'visao' in o:
            vis = o['visao']

            def has_resource(v):
                if isinstance(v, int):
                    return v > 0
                return v == 'FAROL'  # compatível com FarolEnv, não estraga nada

            L = int(has_resource(vis.get('L')))
            R = int(has_resource(vis.get('R')))
            U = int(has_resource(vis.get('U')))
            D = int(has_resource(vis.get('D')))
            C = int(has_resource(vis.get('C')))

            carrying = o.get('carrying', 0)

            return ('FORAGE', L, R, U, D, C, carrying)

        return ('GEN', str(o))
