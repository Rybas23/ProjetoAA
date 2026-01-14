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

    def save_heatmap(self, filename):
        """Salva heatmap de posições visitadas (implementação padrão: não faz nada)."""
        # Agentes que não rastreiam posições simplesmente ignoram
        pass

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

        # Tracking de posições para heatmap
        self.position_heatmap = {}  # {(x,y): count}

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

        # Tracking de posição para heatmap
        if 'pos' in self.ultima_observacao:
            pos = tuple(self.ultima_observacao['pos'])
            if pos not in self.position_heatmap:
                self.position_heatmap[pos] = 0
            self.position_heatmap[pos] += 1

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

    def save_heatmap(self, filename):
        """Salva heatmap de posições visitadas em formato CSV."""
        if not self.position_heatmap:
            return  # Sem dados para salvar

        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'visits'])
            for (x, y), count in sorted(self.position_heatmap.items()):
                writer.writerow([x, y, count])

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

        # POPULAÇÃO de genomes para diversidade
        self.population_size = 30  # Manter 30 melhores genomes (aumentado de 15)
        self.population = []  # Lista de (fitness, genome)

        # Contador de episódios para estratégia adaptativa
        self.episode_count = 0

        # Parâmetros adaptativos de mutação (guardar valores iniciais)
        self.initial_mutation_rate = float(mutation_rate)
        self.initial_mutation_scale = float(mutation_scale)

        # Genome atual = flattened weight matrix [n_actions x feature_dim]
        # Inicialização com valores ligeiramente maiores para dar "força" inicial
        self.genome = [random.uniform(-0.5, 0.5)
                       for _ in range(self.n_actions * self.feature_dim)]
        self.best_genome = list(self.genome)
        self.best_fitness = -math.inf

        # Last episode fitness
        self._episode_reward = 0.0

        # Tracking de fitness por genome (para cálculo de média)
        self.genome_fitness_history = {}

        # Tracking de posições para heatmap
        self.position_heatmap = {}  # {(x,y): count} - acumula TODOS os episódios
        self.episode_heatmap = {}   # {(x,y): count} - apenas episódio atual

        # Anti-loop: memória de últimas posições para detectar bloqueio
        self.recent_positions = []  # últimas 8 posições
        self.stuck_counter = 0  # contador de vezes em loop

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

        # Tracking de posição para heatmap
        if 'pos' in self.ultima_observacao:
            pos = tuple(self.ultima_observacao['pos'])

            # Heatmap global (todos os episódios) - para CSV
            if pos not in self.position_heatmap:
                self.position_heatmap[pos] = 0
            self.position_heatmap[pos] += 1

            # Heatmap do episódio atual (para fitness/bloqueios)
            if pos not in self.episode_heatmap:
                self.episode_heatmap[pos] = 0
            self.episode_heatmap[pos] += 1

            # ANTI-LOOP AGRESSIVO: Detectar bloqueios rapidamente
            self.recent_positions.append(pos)
            if len(self.recent_positions) > 8:
                self.recent_positions.pop(0)

            # DETECÇÃO DE LOOP MULTINÍVEL:
            if len(self.recent_positions) >= 4:
                unique_positions = len(set(self.recent_positions))
                recent_4 = self.recent_positions[-4:]
                unique_recent_4 = len(set(recent_4))

                # Nível 1: Últimas 4 posições são apenas 1-2 diferentes
                if unique_recent_4 <= 2:
                    self.stuck_counter += 2  # Incremento maior!
                # Nível 2: Últimas 8 posições são apenas 2-3 diferentes
                elif len(self.recent_positions) >= 8 and unique_positions <= 3:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1)  # Decai lentamente

                # Se preso há 2+ incrementos, FORÇA saída
                if self.stuck_counter >= 2 and self.modo == "learn":
                    # Escolher ação diferente da última
                    available_actions = [a for a in self.lista_acoes]
                    acao = random.choice(available_actions)
                    self.stuck_counter = 0  # Reset após forçar
                    self.regista_passo()
                    return acao

        feats = self._to_features(self.ultima_observacao)
        if len(feats) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {len(feats)}")

        scores = self._forward(feats)

        if self.modo == "learn":
            # --- softmax exploration com temperatura DECRESCENTE ---
            # Temperatura alta inicial (exploração) → temperatura baixa final (exploitação)
            # Episódio 0: tau = 1.0 (muito estocástico)
            # Episódio 100: tau = 0.2 (mais determinístico)
            # Episódio 200+: tau = 0.05 (quase greedy)

            # Calcular episódio atual baseado em episode_rewards
            current_episode = len(self.episode_rewards)

            if current_episode < 100:
                # Fase 1: tau decresce de 1.0 → 0.2
                progress = current_episode / 100.0
                tau = 1.0 - (0.8 * progress)  # 1.0 → 0.2
            elif current_episode < 200:
                # Fase 2: tau decresce de 0.2 → 0.05
                progress = (current_episode - 100) / 100.0
                tau = 0.2 - (0.15 * progress)  # 0.2 → 0.05
            else:
                # Fase 3: tau fixo muito baixo (quase greedy)
                tau = 0.05

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

    def _mutate(self, base_genome, generation=0):
        """
        Mutação ULTRA-conservadora - apenas refinamento mínimo.

        Args:
            base_genome: Genome base para mutar
            generation: Geração atual (para calcular decaimento)
        """
        # Taxa MUITO baixa: 2% → 0.5%
        decay_factor = max(0.25, 1.0 - (generation / 300.0))
        effective_rate = 0.02 * decay_factor

        # Força MUITO pequena: 0.05 → 0.01
        effective_scale = 0.05 * decay_factor

        new_genome = []
        for w in base_genome:
            if random.random() < effective_rate:
                w = w + random.gauss(0.0, effective_scale)
            new_genome.append(w)
        return new_genome

    def _crossover(self, parent1, parent2):
        """
        Uniform crossover melhorado: cada peso vem de um dos pais com 50% probabilidade.
        Preserva estrutura dos pais em blocos contíguos.
        """
        child = []
        # Crossover por blocos para preservar estrutura
        for i, (w1, w2) in enumerate(zip(parent1, parent2)):
            # Alterna entre pais em blocos de feature_dim
            if (i // self.feature_dim) % 2 == 0:
                child.append(w1 if random.random() < 0.7 else w2)
            else:
                child.append(w2 if random.random() < 0.7 else w1)
        return child

    def _tournament_selection(self, tournament_size=3):
        """
        Seleciona um genome da população usando torneio de tamanho 3.
        Retorna o melhor genome entre tournament_size candidatos aleatórios.
        """
        if len(self.population) < tournament_size:
            # Se população pequena, retorna o melhor
            return max(self.population, key=lambda x: x[0])[1]

        candidates = random.sample(self.population, tournament_size)
        winner = max(candidates, key=lambda x: x[0])
        return winner[1]

    def get_metrics(self):
        base = super().get_metrics()
        base.update({
            "best_fitness": self.best_fitness,
            "genome_size": len(self.genome),
            "population_size": len(self.population),
            "population_fitness": [f for f, _ in self.population] if self.population else [],
        })
        return base

    def _calc_fitness(self):
        """Default fitness: total episode reward."""
        return self._episode_reward

    def reset(self, episodio):
        """
        Called at start of each episode.
        ESTRATÉGIA SIMPLES: Elitismo forte com refinamento mínimo.
        """
        if self._current_steps > 0:
            fitness = self._calc_fitness()

            if self.modo == "learn":
                # Adicionar genome atual à população
                self.population.append((fitness, list(self.genome)))

                # Manter apenas os N melhores
                self.population.sort(key=lambda x: x[0], reverse=True)
                self.population = self.population[:self.population_size]

                # Atualizar melhor fitness global
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_genome = list(self.genome)

                # ESTRATÉGIA SIMPLES: 90% usar o melhor, 10% exploração mínima
                if random.random() < 0.10 and episodio < 400:
                    # 10% exploração nos primeiros 400 episódios
                    if len(self.population) >= 2 and random.random() < 0.5:
                        # Crossover entre top 2
                        parent1 = self.best_genome
                        parent2 = self.population[1][1] if len(self.population) > 1 else self.best_genome
                        child = self._crossover(parent1, parent2)
                        self.genome = self._mutate(child, episodio)
                    else:
                        # Mutação leve do melhor
                        self.genome = self._mutate(list(self.best_genome), episodio)
                else:
                    # 90%: Usar sempre o melhor genome
                    self.genome = list(self.best_genome)

        # reset state for new episode
        self._current_reward = 0.0
        self._current_steps = 0
        self._episode_reward = 0.0
        self.ultima_observacao = None

        # Resetar anti-loop
        self.recent_positions = []
        self.stuck_counter = 0

        # Resetar heatmap do episódio
        self.episode_heatmap = {}

        # Resetar anti-loop
        self.recent_positions = []
        self.stuck_counter = 0

        # Resetar heatmap do episódio (para detecção de bloqueios)
        self.episode_heatmap = {}

    def save_heatmap(self, filename):
        """Salva heatmap de posições visitadas em formato CSV."""
        if not self.position_heatmap:
            return  # Sem dados para salvar

        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'visits'])
            for (x, y), count in sorted(self.position_heatmap.items()):
                writer.writerow([x, y, count])

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

        # INICIALIZAÇÃO BALANCEADA - permite exploração mantendo conhecimento básico
        self._init_intelligent_genome()

    def _init_intelligent_genome(self):
        """
        Inicialização INTELIGENTE FORTE para garantir boa performance.

        Bias FORTE:
        - Direção farol: +1.0 a +1.5 (forte tendência)
        - Evitar paredes: -4.0 a -5.0 (penalização forte)

        Resultado: Agente começa já com bom desempenho e refina através de GA
        """
        for action_idx, action in enumerate(self.lista_acoes):
            for feat_idx in range(self.feature_dim):
                genome_idx = action_idx * self.feature_dim + feat_idx

                # BIAS FORTE: Seguir direção do farol
                if feat_idx == 2 and action == 'UP':  # dir_N -> UP
                    self.genome[genome_idx] = random.uniform(1.0, 1.5)
                elif feat_idx == 3 and action == 'DOWN':  # dir_S -> DOWN
                    self.genome[genome_idx] = random.uniform(1.0, 1.5)
                elif feat_idx == 4 and action == 'RIGHT':  # dir_E -> RIGHT
                    self.genome[genome_idx] = random.uniform(1.0, 1.5)
                elif feat_idx == 5 and action == 'LEFT':  # dir_O -> LEFT
                    self.genome[genome_idx] = random.uniform(1.0, 1.5)

                # Bias negativo para ir CONTRA direção do farol
                elif feat_idx == 2 and action == 'DOWN':  # dir_N mas vai DOWN
                    self.genome[genome_idx] = random.uniform(-2.0, -1.5)
                elif feat_idx == 3 and action == 'UP':  # dir_S mas vai UP
                    self.genome[genome_idx] = random.uniform(-2.0, -1.5)
                elif feat_idx == 4 and action == 'LEFT':  # dir_E mas vai LEFT
                    self.genome[genome_idx] = random.uniform(-2.0, -1.5)
                elif feat_idx == 5 and action == 'RIGHT':  # dir_O mas vai RIGHT
                    self.genome[genome_idx] = random.uniform(-2.0, -1.5)

                # BIAS FORTE: Evitar paredes
                if feat_idx == 7 and action == 'LEFT':  # parede à esquerda -> não LEFT
                    self.genome[genome_idx] = random.uniform(-5.0, -4.0)
                if feat_idx == 10 and action == 'RIGHT':  # parede à direita -> não RIGHT
                    self.genome[genome_idx] = random.uniform(-5.0, -4.0)
                if feat_idx == 13 and action == 'UP':  # parede acima -> não UP
                    self.genome[genome_idx] = random.uniform(-5.0, -4.0)
                if feat_idx == 16 and action == 'DOWN':  # parede abaixo -> não DOWN
                    self.genome[genome_idx] = random.uniform(-5.0, -4.0)

        self.best_genome = list(self.genome)

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

    def _calc_fitness(self):
        """
        Fitness especializado para Farol com FOCO TOTAL EM SUCESSO + EFICIÊNCIA.

        OBJETIVOS:
        1. Agentes que ATINGEM o farol têm fitness >> 0
        2. Agentes mais RÁPIDOS têm fitness maior
        3. Agentes que FALHAM têm fitness muito baixo ou zero

        Formula melhorada:
        - SUCESSO (reward >= 100): fitness = 1000 + (200 - steps) * 10
          * Base: 1000 pontos por sucesso
          * Bonus eficiência: Até 2000 pontos (se steps=0) a 0 pontos (se steps=200)
          * Range: [1000, 3000]

        - FALHA (reward < 100): fitness = max(0, reward - steps * 0.5)
          * Recompensa parcial por aproximação
          * Penalização por gastar muitos passos
          * Range: [0, ~100]

        RESULTADO: Agentes bem-sucedidos têm fitness 10-30x maior que falhas
        """
        reward = self._episode_reward
        steps = self._current_steps

        # Threshold de sucesso: chegou ao farol
        SUCCESS_THRESHOLD = 100.0

        if reward >= SUCCESS_THRESHOLD:
            # ✓ SUCESSO! Fitness MASSIVO com bonus ENORME por eficiência
            # OBJETIVO: Agente de 14 steps >> Agente de 50 steps

            base_success = 10000.0  # Base MUITO alta (aumentado de 1000)

            # Bonus EXPONENCIAL por eficiência
            # Quanto menos steps, MUITO maior o bonus
            max_steps = 200
            steps_ratio = steps / max_steps  # 0.0 (melhor) a 1.0 (pior)

            # Bonus exponencial: premia MUITO mais os agentes rápidos
            # steps=10 → ratio=0.05 → efficiency=0.95 → bonus=~18000
            # steps=20 → ratio=0.10 → efficiency=0.90 → bonus=~16000
            # steps=50 → ratio=0.25 → efficiency=0.75 → bonus=~10000
            # steps=100 → ratio=0.50 → efficiency=0.50 → bonus=~5000
            efficiency = 1.0 - steps_ratio
            efficiency_bonus = (efficiency ** 2) * 20000.0  # Quadrático!

            fitness = base_success + efficiency_bonus

            # Range de fitness:
            # 14 steps:  10000 + 19000 = 29000
            # 20 steps:  10000 + 18000 = 28000
            # 50 steps:  10000 + 11250 = 21250
            # 100 steps: 10000 + 5000  = 15000
            # 200 steps: 10000 + 0     = 10000

        else:
            # ✗ FALHA: Fitness muito baixo
            # Pequena recompensa por aproximação, mas grande penalização

            # Penalização por steps: cada step custa 0.5 fitness
            step_penalty = steps * 0.5

            # Fitness pode ser zero se gastou muitos steps sem resultado
            fitness = max(0.0, reward - step_penalty)

            # Penalização extra por bloqueio (ficar preso)
            if hasattr(self, 'episode_heatmap') and self.episode_heatmap:
                max_visits = max(self.episode_heatmap.values())
                avg_visits = sum(self.episode_heatmap.values()) / len(self.episode_heatmap)

                # Se visitou mesma posição muitas vezes = preso
                if max_visits > avg_visits * 4:
                    stuck_penalty = (max_visits - avg_visits) * 2.0
                    fitness = max(0.0, fitness - stuck_penalty)

        return fitness

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
        Fitness para Foraging:
        - PRIORIZA: Recursos entregues (×100 cada)
        - BONUS: Eficiência (recursos/passo)
        - INCLUI: Reward acumulado

        Formula:
        fitness = (recursos × 100) + (recursos/steps × 50) + reward
        """
        ambiente = getattr(self, "ambiente", None)
        delivered = 0
        if ambiente is not None:
            delivered = getattr(ambiente, "total_delivered", 0)

        steps = max(self._current_steps, 1)  # Evitar divisão por zero

        # Componente principal: recursos entregues
        resource_score = delivered * 100.0

        # Bonus de eficiência: recursos por passo
        if delivered > 0:
            efficiency_bonus = (delivered / steps) * 50.0
        else:
            efficiency_bonus = 0.0

        # Reward acumulado (aproximação, paredes, etc)
        reward_score = self._episode_reward

        return resource_score + efficiency_bonus + reward_score

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