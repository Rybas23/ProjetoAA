import random
import math
from enum import Enum

class AgenteBase:
    def __init__(self, id, modo='test'):
        self.id = id
        self.modo = modo           # "learn" ou "test"
        self.ambiente = None
        self.sensores = []
        self.ultima_observacao = None
        self.verbose = False

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
        if self.verbose:
            print(f"[{self.id}] recebeu mensagem de {agente_origem.id}: {mensagem}")

    # --- interface que subclasses devem implementar ---

    def age(self):
        raise NotImplementedError

    def avaliacaoEstadoAtual(self, recompensa):
        # Atualiza métricas genéricas; subclasses podem sobrescrever mas
        # devem chamar super() para manter estes contadores.
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
        # FarolEnv: usa posicao, direcao do farol e visão (se existir)
        pos = tuple(observacao.get('pos', (0, 0)))
        direcao = observacao.get('direcao_farol', 'NONE')

        visao = observacao.get('visao')
        if isinstance(visao, dict):
            visao_state = tuple(sorted(visao.items()))
        else:
            visao_state = None

        return ('pos', pos, 'dir', direcao, 'visao', visao_state)

#  Q-AGENT PARA O AMBIENTE DE FORAGING (ForagingEnv)
class QAgentForaging(QAgentBase):
    def __init__(self, id='QForaging', lista_acoes=None, modo='learn'):
        if lista_acoes is None:
            lista_acoes = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        super().__init__(id=id, lista_acoes=lista_acoes, modo=modo)

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
        # ForagingEnv: pos, visão de recursos, carregando, ninho relativo
        pos = tuple(observacao.get('pos', (0, 0)))

        visao = observacao.get('visao', {})
        if isinstance(visao, dict):
            visao_state = tuple(sorted(visao.items()))
        else:
            visao_state = None

        carrying = int(observacao.get('carrying', 0))

        nest = observacao.get('nest', None)
        if isinstance(nest, tuple):
            nest_state = nest
        else:
            nest_state = None

        return ('pos', pos,
                'visao', visao_state,
                'carry', carrying,
                'nest', nest_state)

#  BASE DE UM AGENTE COM ALGORITMO GENETICO (REDES NEURONAIS SIMPLES)
class GAAgentBase(AgenteBase):
    """
    Simple genetic / evolutionary agent:
    - Policy is a linear mapping features -> action scores (weights = genome).
    - Fitness = total return per episode.
    - After each episode, if fitness improved, keep genome; otherwise mutate.
    """
    def __init__(self, id, lista_acoes, feature_dim, modo='learn',
                 mutation_rate=0.1, mutation_scale=0.1):
        super().__init__(id=id, modo=modo)
        self.lista_acoes = list(lista_acoes)
        self.n_actions = len(self.lista_acoes)
        self.feature_dim = feature_dim
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)

        # Genome = flattened weight matrix [n_actions x feature_dim]
        self.genome = [random.uniform(-0.1, 0.1)
                       for _ in range(self.n_actions * self.feature_dim)]
        self.best_genome = list(self.genome)
        self.best_fitness = -math.inf

        # Last episode fitness
        self._episode_reward = 0.0

    # ----- representation hooks (subclasses override) -----

    def _to_features(self, observacao):
        """
        Map raw observation dict -> feature vector (list[float]) of length feature_dim.
        Subclasses must override.
        """
        raise NotImplementedError

    # ----- policy evaluation -----

    def _forward(self, features):
        """Compute action scores = W * features."""
        scores = [0.0] * self.n_actions
        idx = 0
        for a in range(self.n_actions):
            s = 0.0
            for f in range(self.feature_dim):
                s += self.genome[idx] * features[f]
                idx += 1
            scores[a] = s
        return scores

    def age(self):
        """
        Selects an action using the current genome.
        No exploration noise here; learning happens between episodes via mutation.
        """
        if self.ultima_observacao is None:
            # fallback: random until we see an observation
            acao = random.choice(self.lista_acoes)
            self.regista_passo()
            return acao

        feats = self._to_features(self.ultima_observacao)
        # ensure correct size
        if len(feats) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {len(feats)}")

        scores = self._forward(feats)
        # greedy action
        best_idx = max(range(self.n_actions), key=lambda i: scores[i])
        acao = self.lista_acoes[best_idx]

        self.regista_passo()
        return acao

    # ----- episodic update (fitness) -----

    def avaliacaoEstadoAtual(self, recompensa):
        # track reward as fitness signal
        super().avaliacaoEstadoAtual(recompensa)
        self._episode_reward += float(recompensa)

    def _mutate(self, base_genome):
        """
        Simple Gaussian mutation applied elementwise with given mutation_rate.
        """
        new_genome = []
        for w in base_genome:
            if random.random() < self.mutation_rate:
                w = w + random.gauss(0.0, self.mutation_scale)
            new_genome.append(w)
        return new_genome

    def reset(self, episodio):
        """
        Called at start of each episode by the engine.
        Here we treat the reward accumulated in the *previous* episode
        as fitness, and possibly update the genome.
        """
        # close previous episode if any steps happened
        if self._current_steps > 0:
            fitness = self._episode_reward
            if self.modo == "learn":
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_genome = list(self.genome)
                else:
                    # revert to best and mutate
                    self.genome = self._mutate(self.best_genome)

        # reset counters and reward
        self._current_reward = 0.0
        self._current_steps = 0
        self._episode_reward = 0.0
        self.ultima_observacao = None

    def get_metrics(self):
        base = super().get_metrics()
        base.update({
            "best_fitness": self.best_fitness,
            "genome_size": len(self.genome),
        })
        return base

#  AGENTE GENÉTICO PARA O AMBIENTE FAROL (FarolEnv)
class GAAgentFarol(GAAgentBase):
    """
    Genetic‑algorithm agent for FarolEnv.
    Features:
    - pos (x,y) normalized by grid size (assume 10 by default)
    - one‑hot direction to farol: N,S,E,O,NONE (5)
    - local vision (L,R,U,D,C) each encoded as one‑hot: {PAREDE,FAROL,OUTRO} (3*5)
    Total feature_dim = 2 + 5 + 15 = 22
    """
    def __init__(self, id="GAFarol", lista_acoes=None, modo="learn",
                 mutation_rate=0.1, mutation_scale=0.1):
        if lista_acoes is None:
            lista_acoes = ["UP", "DOWN", "LEFT", "RIGHT"]

        feature_dim = 2 + 5 + 15  # = 22

        super().__init__(
            id=id,
            lista_acoes=lista_acoes,
            feature_dim=feature_dim,
            modo=modo,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
        )

    @classmethod
    def cria(cls, p):
        if p is None:
            p = {}
        return cls(
            id=p.get("id", "GAFarol"),
            lista_acoes=p.get("actions", None),
            modo=p.get("mode", "learn"),
            mutation_rate=p.get("mutation_rate", 0.1),
            mutation_scale=p.get("mutation_scale", 0.1),
        )

    def _to_features(self, observacao):
        """
        Map Farol observation dict into a length‑22 feature vector.
        observacao:
          - "pos": (x,y)
          - "direcao_farol": "N","S","E","O","NONE"
          - "visao": {L,R,U,D,C} -> "PAREDE","FAROL","AG_x" or "VAZIO"
        """
        # 1) position
        x, y = observacao.get("pos", (0, 0))
        # assume size 10 if not known; still works scaled
        pos_x = float(x) / 10.0
        pos_y = float(y) / 10.0
        features = [pos_x, pos_y]

        # 2) one‑hot direction to farol
        dir_str = observacao.get("direcao_farol", "NONE")
        dirs = ["N", "S", "E", "O", "NONE"]
        for d in dirs:
            features.append(1.0 if dir_str == d else 0.0)

        # 3) vision encoding: for each of L,R,U,D,C
        #    types: PAREDE, FAROL, OUTRO(agente ou vazio)
        vis = observacao.get("visao", {})
        keys = ["L", "R", "U", "D", "C"]

        def encode_cell(val):
            t_parede = 1.0 if val == "PAREDE" else 0.0
            t_farol = 1.0 if val == "FAROL" else 0.0
            # any agent or vazio counts as "OUTRO"
            t_outro = 0.0
            if val not in ("PAREDE", "FAROL"):
                t_outro = 1.0
            return [t_parede, t_farol, t_outro]

        for k in keys:
            v = vis.get(k, "PAREDE")
            features.extend(encode_cell(v))

        # safety: ensure correct size
        if len(features) != self.feature_dim and self.verbose:
            print(f"[{self.id}] feature length {len(features)} != {self.feature_dim}")
        return features

#  AGENTE GENÉTICO PARA O AMBIENTE DE FORAGING (ForagingEnv)
class GAAgentForaging(GAAgentBase):
    """
    Genetic-algorithm agent for ForagingEnv.
    Observation -> small feature vector -> linear policy.
    """
    def __init__(
        self,
        id="GAForaging",
        lista_acoes=None,
        modo="learn",
        mutation_rate=0.1,
        mutation_scale=0.1,
    ):
        if lista_acoes is None:
            # same action set as QAgentForaging
            lista_acoes = ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"]

        feature_dim = 10  # pos(2) + vis(5) + carrying(1) + nest vec(2)

        super().__init__(
            id=id,
            lista_acoes=lista_acoes,
            feature_dim=feature_dim,
            modo=modo,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
        )

    @classmethod
    def cria(cls, p):
        """
        Factory compatible with JSON agent config.
        Example agent block:
        {
          "type": "GAAgentForaging",
          "id": "GA1",
          "mode": "learn",
          "mutation_rate": 0.1,
          "mutation_scale": 0.1
        }
        """
        if p is None:
            return cls()
        return cls(
            id=p.get("id", "GAForaging"),
            lista_acoes=p.get(
                "actions",
                ["UP", "DOWN", "LEFT", "RIGHT", "PICK", "DROP"],
            ),
            modo=p.get("mode", "learn"),
            mutation_rate=p.get("mutation_rate", 0.1),
            mutation_scale=p.get("mutation_scale", 0.1),
        )

    def _to_features(self, observacao):
        """
        Map Foraging observation dict into a length-10 feature vector.
        observacao keys:
          - "pos": (x,y)
          - "visao": dict L,R,U,D,C -> int (resources or -1 for wall)
          - "carrying": 0/1
          - "nest": (nx,ny)
        """
        # 1) position (x,y) normalized by 10 (still reasonable for other sizes)
        x, y = observacao.get("pos", (0, 0))
        pos_x = float(x) / 10.0
        pos_y = float(y) / 10.0

        # 2) local vision: L,R,U,D,C
        vis = observacao.get("visao", {})

        def norm_res(v):
            # wall or out-of-grid -> -1, treat as 0
            if v is None:
                return 0.0
            if v < 0:
                return 0.0
            # clip and normalize count [0..5] -> [0..1]
            return min(float(v), 5.0) / 5.0

        vL = norm_res(vis.get("L", 0))
        vR = norm_res(vis.get("R", 0))
        vU = norm_res(vis.get("U", 0))
        vD = norm_res(vis.get("D", 0))
        vC = norm_res(vis.get("C", 0))

        # 3) carrying flag
        carrying = float(observacao.get("carrying", 0))

        # 4) relative nest vector (nx - x, ny - y) normalized by 10
        nest = observacao.get("nest", (0, 0))
        nx, ny = nest
        dx = float(nx - x) / 10.0
        dy = float(ny - y) / 10.0

        return [
            pos_x,
            pos_y,
            vL,
            vR,
            vU,
            vD,
            vC,
            carrying,
            dx,
            dy,
        ]