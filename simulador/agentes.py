import random

class AgenteBase:
    def __init__(self, id, modo='test'):
        self.id = id
        self.modo = modo           # "learn" ou "test"
        self.ambiente = None
        self.sensores = []
        self.ultima_observacao = None
        self.logs = False

        # Métricas por episódio (genéricas)
        self._current_reward = 0.0
        self._current_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    @classmethod
    def cria(cls, p):
        return cls(id=p.get('id', 'agente'), modo=p.get('mode', 'test'))

    def instala_ambiente(self, ambiente):
        self.ambiente = ambiente

    def instala(self, sensor):
        self.sensores.append(sensor)

    def observacao(self, observacao_dict):
        self.ultima_observacao = observacao_dict

    def comunica(self, mensagem, agente_origem):
        if self.logs:
            print(f"[{self.id}] recebeu mensagem de {agente_origem.id}: {mensagem}")

    # --- interface que subclasses devem implementar ---

    def age(self):
        raise NotImplementedError

    def avaliacaoEstadoAtual(self, recompensa):
        # Atualiza métricas genéricas
        self._current_reward += float(recompensa)

    def reset(self, episodio):
        # Fecha episódio anterior e guarda métricas
        if self._current_steps > 0 or self._current_reward != 0.0:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_steps)

        # Reinicia contadores
        self._current_reward = 0.0
        self._current_steps = 0
        self.ultima_observacao = None

    def regista_passo(self):
        # Chamado em cada passo pelo motor (ou dentro de age())
        self._current_steps += 1

    def get_metrics(self):
        # Devolve cópias para análise externa
        return {
            'rewards': list(self.episode_rewards),
            'lengths': list(self.episode_lengths),
        }

#  AGENTE FIXO (NÃO APRENDE)
class FixedAgent(AgenteBase):
    def __init__(self, id, politica, modo='test'):
        super().__init__(id=id, modo=modo)
        self.politica = politica  # função: observacao_dict -> acao

    def age(self):
        if self.ultima_observacao is None:
            raise RuntimeError(f"[{self.id}] age() chamado sem observação")

        acao = self.politica(self.ultima_observacao)
        self.regista_passo()
        return acao

    def avaliacaoEstadoAtual(self, recompensa):
        # FixedAgent não aprende, apenas regista métricas
        super().avaliacaoEstadoAtual(recompensa)

#  BASE DE UM AGENTE COM Q-LEARNING
class QAgentBase(AgenteBase):
    def __init__(self, id, lista_acoes,
                 alpha=0.4, gamma=0.95,
                 epsilon=0.2, modo='learn'):

        super().__init__(id=id, modo=modo)

        # Q-learning params
        self.acoes = list(lista_acoes)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

        # Q-table: dict[state][action] = valor-Q
        self.qtable = {}

        # Último estado/ação (para update)
        self.estado_anterior = None
        self.acao_anterior = None

        # epsilon mínimo e taxa de decaimento (para learning)
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # --------- Representação de estado (override em subclasses) ---------

    def _to_state(self, observacao):
        # Fallback genérico: transforma dict em tuplo ordenado
        if not isinstance(observacao, dict):
            return tuple(observacao)
        itens = []
        for k in sorted(observacao.keys()):
            v = observacao[k]
            if isinstance(v, dict):
                v = tuple(sorted(v.items()))
            itens.append((k, v))
        return tuple(itens)

    # --------- Escolha de ação ---------

    def age(self):
        if self.ultima_observacao is None:
            raise RuntimeError(f"[{self.id}] age() chamado sem observação")

        estado_atual = self._to_state(self.ultima_observacao)

        # Garantir estado na Q-table
        if estado_atual not in self.qtable:
            self.qtable[estado_atual] = {a: 0.0 for a in self.acoes}

        # Política de seleção:
        if self.modo == 'learn' and random.random() < self.epsilon:
            # exploração apenas em modo aprendizagem
            acao_escolhida = random.choice(self.acoes)
        else:
            # modo test ou exploração desativada -> greedy sobre Q-table
            q_vals = self.qtable[estado_atual]
            max_q = max(q_vals.values())
            melhores = [a for a, v in q_vals.items() if v == max_q]
            acao_escolhida = random.choice(melhores)

        # Guardar para update futuro
        self.estado_anterior = estado_atual
        self.acao_anterior = acao_escolhida

        # Contabilizar passo para métricas de episódio
        self.regista_passo()

        return acao_escolhida

    # --------- Atualização Q-learning ---------

    def avaliacaoEstadoAtual(self, recompensa):
        # Atualiza métricas genéricas
        super().avaliacaoEstadoAtual(recompensa)

        if self.modo != 'learn':
            # Em modo teste nunca se altera a política / Q-table
            return

        if self.estado_anterior is None or self.acao_anterior is None:
            return

        estado_passado = self.estado_anterior
        acao_passada = self.acao_anterior

        estado_atual = self._to_state(self.ultima_observacao)

        if estado_atual not in self.qtable:
            self.qtable[estado_atual] = {a: 0.0 for a in self.acoes}

        q_antigo = self.qtable[estado_passado][acao_passada]
        q_max_prox = max(self.qtable[estado_atual].values())

        q_novo = q_antigo + self.alpha * (
            recompensa + self.gamma * q_max_prox - q_antigo
        )
        self.qtable[estado_passado][acao_passada] = q_novo

    # --------- Gestão de episódios e política pré-treinada ---------

    def reset(self, episodio):
        # fecha episódio anterior e reinicia métricas genéricas
        super().reset(episodio)

        # limpar memória temporária de estado/ação
        self.estado_anterior = None
        self.acao_anterior = None

        # epsilon decay apenas em aprendizagem
        if self.modo == 'learn':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_qtable(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.qtable, f)

    def load_qtable(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.qtable = pickle.load(f)

#  Q-AGENT PARA O AMBIENTE FAROL (FarolEnv)
class QAgentFarol(QAgentBase):
    def __init__(self, id='QFarol', lista_acoes=None, modo='learn'):
        if lista_acoes is None:
            lista_acoes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        super().__init__(id=id, lista_acoes=lista_acoes, modo=modo)

    @classmethod
    def cria(cls, p):
        if p is None:
            return cls()
        return cls(
            id=p.get('id', 'QFarol'),
            lista_acoes=p.get('actions', ['UP', 'DOWN', 'LEFT', 'RIGHT']),
            modo=p.get('mode', 'test')
        )

    def _to_state(self, observacao):
        """Estado compacto para o FarolEnv.

        Usa apenas:
          - direcao_farol: direção relativa ao farol (N,S,E,O,NONE)
          - paredes locais nas direções L,R,U,D (1 se parede, 0 se livre)
        """
        direcao = observacao.get('direcao_farol', 'NONE')

        visao = observacao.get('visao', {})
        # No FarolEnv, visao[L/R/U/D] é 'PAREDE', 'VAZIO', 'FAROL', 'AG_X', etc.
        paredes = (
            1 if visao.get('L') == 'PAREDE' else 0,
            1 if visao.get('R') == 'PAREDE' else 0,
            1 if visao.get('U') == 'PAREDE' else 0,
            1 if visao.get('D') == 'PAREDE' else 0,
        )

        return (direcao, paredes)

#  Q-AGENT PARA O AMBIENTE DE FORAGING (ForagingEnv)
class QAgentForaging(QAgentBase):
    def __init__(self, id='QForaging', lista_acoes=None, modo='learn'):
        if lista_acoes is None:
            lista_acoes = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        super().__init__(id=id, lista_acoes=lista_acoes, modo=modo)
        # memória simples: última ação
        self.ultima_acao = None

    @classmethod
    def cria(cls, p):
        if p is None:
            return cls()
        return cls(
            id=p.get('id', 'QForaging'),
            lista_acoes=p.get('actions', ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']),
            modo=p.get('mode', 'test')
        )

    def _to_state(self, observacao):
        """Estado compacto e local para o ForagingEnv.

        Usa apenas a visão local (valores L,R,U,D,C) e o flag carrying,
        mais a última ação para dar um pouco de memória.
        Evita posição e ninho absolutos para melhorar generalização.
        """
        visao = observacao.get('visao', {})
        visao_state = (
            visao.get('L', 0),
            visao.get('R', 0),
            visao.get('U', 0),
            visao.get('D', 0),
            visao.get('C', 0),
        )

        carrying = int(observacao.get('carrying', 0))

        # usa string da última ação ou um marcador neutro
        last_action = self.ultima_acao or 'NONE'

        return (visao_state, carrying, last_action)

    def age(self):
        """Override para guardar última ação escolhida, reutilizando a lógica base."""
        acao = super().age()
        self.ultima_acao = acao
        return acao

    def reset(self, episodio):
        """Limpa memória de episódio (inclui última ação)."""
        super().reset(episodio)
        self.ultima_acao = None
