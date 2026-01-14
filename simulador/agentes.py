import random
import math

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
                 alpha=0.3, gamma=0.99,
                 epsilon=0.9, modo='learn'):

        super().__init__(id=id, modo=modo)

        # Q-learning params
        self.acoes = list(lista_acoes)
        self.alpha = float(alpha)
        self.alpha_inicial = float(alpha)  # Guardar alpha inicial
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_inicial = float(epsilon)

        # Q-table: dict[state][action] = valor-Q
        self.qtable = {}

        # Último estado/ação (para update)
        self.estado_anterior = None
        self.acao_anterior = None

        # epsilon e alpha: mínimos e taxas de decaimento
        # DEFINITIVO: Convergência MUITO rápida para estabilidade máxima
        self.epsilon_min = 0.01   # 1% exploração mínima
        self.epsilon_decay = 0.997  # Decay MUITO rápido
        self.alpha_min = 0.01      # Learning rate mínimo
        self.alpha_decay = 0.98    # Decay rápido

        # Contador de episódios para decay adaptativo
        self.episodio_atual = 0

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

    def _inicializar_estado(self, estado):
        """Inicializa os valores Q para um novo estado.

        Por padrão, inicializa todas as ações com 0.0.
        Subclasses podem sobrescrever para inicialização otimista.
        """
        return {acao: 0.0 for acao in self.acoes}

    # --------- Escolha de ação ---------

    def age(self):
        if self.ultima_observacao is None:
            raise RuntimeError(f"[{self.id}] age() chamado sem observação")

        estado_atual = self._to_state(self.ultima_observacao)

        # Garantir estado na Q-table
        if estado_atual not in self.qtable:
            self.qtable[estado_atual] = self._inicializar_estado(estado_atual)

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

        # CRÍTICO: Guardar estado ANTES da ação para Q-Learning correto
        # Este será o estado "s" em Q(s,a)
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

        self.episodio_atual = episodio

        # epsilon e alpha decay apenas em aprendizagem
        if self.modo == 'learn':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

            # Decay SUPER AGRESSIVO após ep 100 para forçar estabilização rápida
            if episodio > 100:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.80)
                self.alpha = max(self.alpha_min, self.alpha * 0.85)

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

    def _inicializar_estado(self, estado):
        """Inicialização otimista para o problema do farol.

        Estado: (direcao, paredes)
        onde direcao é a direção relativa ao farol (N,S,E,O,NONE)
        e paredes é uma tupla de 4 valores (L,R,U,D) indicando se há parede
        """
        direcao, paredes = estado

        valores = {}
        mapa_acao = {'N': 'UP', 'S': 'DOWN', 'E': 'RIGHT', 'O': 'LEFT'}

        for acao in self.acoes:
            # Se já estamos no farol (NONE), todas as ações neutras
            if direcao == 'NONE':
                valores[acao] = 0.5
            # Se ação vai na direção do farol
            elif direcao in mapa_acao and mapa_acao[direcao] == acao:
                # Verificar se há parede nessa direção
                idx_parede = {'LEFT': 0, 'RIGHT': 1, 'UP': 2, 'DOWN': 3}
                if acao in idx_parede and paredes[idx_parede[acao]] == 1:
                    # Há parede, valor baixo
                    valores[acao] = 0.0
                else:
                    # Sem parede, otimista
                    valores[acao] = 2.0
            else:
                # Outras ações começam neutras
                valores[acao] = 0.0

        return valores


#  Q-AGENT PARA O AMBIENTE DE FORAGING (ForagingEnv)
class QAgentForaging(QAgentBase):
    def __init__(self, id='QForaging', lista_acoes=None, modo='learn'):
        if lista_acoes is None:
            # Apenas movimentos - PICK e DROP são automáticos
            lista_acoes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        super().__init__(id=id, lista_acoes=lista_acoes, modo=modo)

    @classmethod
    def cria(cls, p):
        if p is None:
            return cls()
        return cls(
            id=p.get('id', 'QForaging'),
            lista_acoes=p.get('actions', ['UP', 'DOWN', 'LEFT', 'RIGHT']),
            modo=p.get('mode', 'test')
        )

    def _to_state(self, observacao):
        """Estado BALANCEADO: compacto mas com informação essencial.

        Total: 2 × 5 × 2 × 2 = 40 estados
        """
        carrying = int(observacao.get('carrying', 0))
        visao = observacao.get('visao', {})
        pos = observacao.get('pos', (0, 0))
        nest = observacao.get('nest', (0, 0))

        # Determinar direção do objetivo
        if carrying == 0:
            direcao_objetivo = observacao.get('direcao_recurso', 'NONE')
        else:
            if nest[0] > pos[0]:
                direcao_objetivo = 'E'
            elif nest[0] < pos[0]:
                direcao_objetivo = 'O'
            elif nest[1] < pos[1]:
                direcao_objetivo = 'N'
            elif nest[1] > pos[1]:
                direcao_objetivo = 'S'
            else:
                direcao_objetivo = 'NONE'

        # CRÍTICO: Estamos NO objetivo?
        no_objetivo = False
        if carrying == 0:
            no_objetivo = (visao.get('C', 0) == 1)  # Recurso na célula atual
        else:
            no_objetivo = (pos == nest)  # No ninho

        # Parede na direção do objetivo?
        mapa_direcao = {'N': 'U', 'S': 'D', 'E': 'R', 'O': 'L'}
        parede_bloqueando = False
        if direcao_objetivo in mapa_direcao:
            tecla_visao = mapa_direcao[direcao_objetivo]
            parede_bloqueando = (visao.get(tecla_visao, 0) == -1)

        return (carrying, direcao_objetivo, no_objetivo, parede_bloqueando)

    def _inicializar_estado(self, estado):
        """Inicialização otimista para foraging.

        Estado: (carrying, direcao_objetivo, no_objetivo, parede_bloqueando)
        """
        carrying, direcao_obj, no_obj, parede = estado

        valores = {}
        mapa_acao = {'N': 'UP', 'S': 'DOWN', 'E': 'RIGHT', 'O': 'LEFT'}

        for acao in self.acoes:
            # Se já estamos no objetivo, todas as ações neutras
            if no_obj:
                valores[acao] = 0.5
            # Se ação vai na direção do objetivo
            elif direcao_obj in mapa_acao and mapa_acao[direcao_obj] == acao:
                # Se há parede, valor baixo; senão, otimista
                valores[acao] = 0.0 if parede else 2.0
            else:
                # Outras ações começam neutras
                valores[acao] = 0.0

        return valores


#  BASE DE UM AGENTE COM ALGORITMO GENETICO (REDES NEURONAIS SIMPLES)
class GAAgentBase(AgenteBase):
    """
    Simple genetic / evolutionary agent:
    - Policy is a linear mapping features -> action scores (weights = genome).
    - Fitness = total return per episode.
    - After each episode, if fitness improved, keep genome; otherwise mutate.
    """
    def __init__(self, id, lista_acoes, feature_dim, modo='learn',
                 mutation_rate=0.2, mutation_scale=0.2):
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
        Select an action using the current genome.
        In 'learn' mode we use a softmax policy (stochastic),
        in other modes we act greedily.
        """
        if self.ultima_observacao is None:
            # fallback: random until we see an observation
            acao = random.choice(self.lista_acoes)
            self.regista_passo()
            return acao

        feats = self._to_features(self.ultima_observacao)
        if len(feats) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {len(feats)}")

        scores = self._forward(feats)

        if self.modo == "learn":
            # --- softmax exploration ---
            tau = 0.7  # temperature; tune as needed
            # numeric stability
            max_s = max(scores)
            exp_scores = [math.exp((s - max_s) / tau) for s in scores]
            total = sum(exp_scores)
            if total <= 0.0 or any(math.isnan(e) for e in exp_scores):
                # fallback: uniform random
                acao = random.choice(self.lista_acoes)
            else:
                probs = [e / total for e in exp_scores]
                r = random.random()
                cumsum = 0.0
                idx = 0
                for i, p in enumerate(probs):
                    cumsum += p
                    if r <= cumsum:
                        idx = i
                        break
                acao = self.lista_acoes[idx]
        else:
            # greedy action (test mode)
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

    def get_metrics(self):
        base = super().get_metrics()
        base.update({
            "best_fitness": self.best_fitness,
            "genome_size": len(self.genome),
        })
        return base

    def _calc_fitness(self):
        """Default fitness: total episode reward."""
        return self._episode_reward

    def reset(self, episodio):
        """
        Called at start of each episode.
        Treat accumulated reward (or custom metric) as fitness.
        """
        if self._current_steps > 0:
            fitness = self._calc_fitness()
            if self.modo == "learn":
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_genome = list(self.genome)
                else:
                    # revert to best and mutate
                    self.genome = self._mutate(self.best_genome)

        # reset state for new episode
        self._current_reward = 0.0
        self._current_steps = 0
        self._episode_reward = 0.0
        self.ultima_observacao = None

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
            # Apenas movimentos - PICK e DROP são automáticos
            lista_acoes = ["UP", "DOWN", "LEFT", "RIGHT"]

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
                ["UP", "DOWN", "LEFT", "RIGHT"],
            ),
            modo=p.get("mode", "learn"),
            mutation_rate=p.get("mutation_rate", 0.1),
            mutation_scale=p.get("mutation_scale", 0.1),
        )

    def _calc_fitness(self):
        """
        Fitness: prioritize delivered resources, then reward.
        Assumes ambiente has total_delivered.
        """
        ambiente = getattr(self, "ambiente", None)
        delivered = 0
        if ambiente is not None:
            delivered = getattr(ambiente, "total_delivered", 0)

        # weight delivered heavily so policies that actually solve the task win
        return 10.0 * delivered + self._episode_reward

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