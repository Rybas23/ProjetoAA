import random
import math

class AgenteBase:
    """
    Classe base para todos os agentes no sistema multi-agente.

    Fornece funcionalidades básicas de interação com o ambiente, gestão de sensores,
    e rastreamento de métricas de desempenho ao longo dos episódios.

    Attributes:
        id: Identificador único do agente
        modo: Modo de operação ('learn' para treino, 'test' para avaliação)
        ambiente: Referência ao ambiente onde o agente opera
        sensores: Lista de sensores instalados no agente
        ultima_observacao: Última percepção recebida do ambiente
        episode_rewards: Histórico de recompensas acumuladas por episódio
        episode_lengths: Histórico de passos executados por episódio
    """
    def __init__(self, id, modo='test'):
        self.id = id
        self.modo = modo
        self.ambiente = None
        self.sensores = []
        self.ultima_observacao = None
        self.logs = False

        # Contadores do episódio atual
        self._current_reward = 0.0
        self._current_steps = 0

        # Histórico de desempenho
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
        """Recebe mensagens de outros agentes (comunicação multi-agente)."""
        if self.logs:
            print(f"[{self.id}] recebeu mensagem de {agente_origem.id}: {mensagem}")

    def age(self):
        """Decide e retorna a próxima ação baseada na observação atual."""
        raise NotImplementedError

    def avaliacaoEstadoAtual(self, recompensa):
        """Processa a recompensa recebida após executar uma ação."""
        self._current_reward += float(recompensa)

    def reset(self, episodio):
        """Prepara o agente para um novo episódio, salvando métricas do anterior."""
        if self._current_steps > 0 or self._current_reward != 0.0:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_steps)

        # Reinicia contadores
        self._current_reward = 0.0
        self._current_steps = 0
        self.ultima_observacao = None

    def regista_passo(self):
        """Incrementa o contador de passos do episódio atual."""
        self._current_steps += 1

    def get_metrics(self):
        """Retorna cópias das métricas de desempenho para análise externa."""
        return {
            'rewards': list(self.episode_rewards),
            'lengths': list(self.episode_lengths),
        }

    def save_heatmap(self, filename):
        """Salva mapa de calor das posições visitadas (implementação padrão vazia)."""
        pass

class FixedAgent(AgenteBase):
    """
    Agente com política fixa pré-definida.

    Executa ações baseadas numa função de política determinística,
    sem realizar aprendizagem. Útil como baseline ou para políticas conhecidas.

    Attributes:
        politica: Função que mapeia observação -> ação
    """
    def __init__(self, id, politica, modo='test'):
        super().__init__(id=id, modo=modo)
        self.politica = politica

    def age(self):
        if self.ultima_observacao is None:
            raise RuntimeError(f"[{self.id}] age() chamado sem observação")

        acao = self.politica(self.ultima_observacao)
        self.regista_passo()
        return acao

    def avaliacaoEstadoAtual(self, recompensa):
        """Regista recompensa mas não altera a política (agente não aprende)."""
        super().avaliacaoEstadoAtual(recompensa)

class QAgentBase(AgenteBase):
    """
    Classe base para agentes que utilizam Q-Learning.

    Implementa o algoritmo Q-Learning com ε-greedy exploration e decay gradual
    de hiperparâmetros. Subclasses devem definir _to_state() e _inicializar_estado()
    conforme as características do problema específico.

    Attributes:
        acoes: Lista de ações disponíveis
        alpha: Taxa de aprendizagem (learning rate)
        gamma: Fator de desconto para recompensas futuras
        epsilon: Taxa de exploração (ε-greedy)
        qtable: Dicionário que mapeia (estado, ação) -> valor Q
        epsilon_decay: Fator de decaimento do epsilon por episódio
        alpha_decay: Fator de decaimento do alpha por episódio
    """
    def __init__(self, id, lista_acoes,
                 alpha=0.3, gamma=0.99,
                 epsilon=0.9, modo='learn'):

        super().__init__(id=id, modo=modo)

        self.acoes = list(lista_acoes)
        self.alpha = float(alpha)
        self.alpha_inicial = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_inicial = float(epsilon)

        self.qtable = {}

        self.estado_anterior = None
        self.acao_anterior = None

        # Valores padrão de decay - subclasses devem sobrescrever conforme necessidade
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha_min = 0.01
        self.alpha_decay = 0.995

        self.episodio_atual = 0
        self.position_heatmap = {}

    def _to_state(self, observacao):
        """Converte observação do ambiente em representação de estado para Q-table."""
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
        """
        Define valores Q iniciais para um estado novo.

        Implementação padrão retorna zeros. Subclasses podem usar
        inicialização otimista para acelerar exploração dirigida.
        """
        return {acao: 0.0 for acao in self.acoes}

    # --------- Escolha de ação ---------

    def age(self):
        """
        Seleciona ação usando política ε-greedy.

        Em modo 'learn', explora com probabilidade ε ou escolhe ação greedy.
        Em modo 'test', sempre escolhe a melhor ação conhecida.
        """
        if self.ultima_observacao is None:
            raise RuntimeError(f"[{self.id}] age() chamado sem observação")

        if 'pos' in self.ultima_observacao:
            pos = tuple(self.ultima_observacao['pos'])
            if pos not in self.position_heatmap:
                self.position_heatmap[pos] = 0
            self.position_heatmap[pos] += 1

        estado_atual = self._to_state(self.ultima_observacao)

        if estado_atual not in self.qtable:
            self.qtable[estado_atual] = self._inicializar_estado(estado_atual)

        if self.modo == 'learn' and random.random() < self.epsilon:
            acao_escolhida = random.choice(self.acoes)
        else:
            q_vals = self.qtable[estado_atual]
            max_q = max(q_vals.values())
            melhores = [a for a, v in q_vals.items() if v == max_q]
            acao_escolhida = random.choice(melhores)

        # Guarda estado e ação para atualização Q-Learning posterior
        self.estado_anterior = estado_atual
        self.acao_anterior = acao_escolhida

        self.regista_passo()
        return acao_escolhida

    def avaliacaoEstadoAtual(self, recompensa):
        """
        Atualiza Q-table usando a regra de Q-Learning.

        Fórmula: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        Apenas atualiza em modo 'learn'.
        """
        super().avaliacaoEstadoAtual(recompensa)

        if self.modo != 'learn':
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
        """
        Prepara agente para novo episódio e aplica decay aos hiperparâmetros.

        Em modo 'learn', reduz gradualmente epsilon e alpha para convergência.
        Após episódio 100, aplica decay acelerado para estabilização rápida.
        """
        super().reset(episodio)

        self.estado_anterior = None
        self.acao_anterior = None
        self.episodio_atual = episodio

        if self.modo == 'learn':
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)

            if episodio > 100:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.80)
                self.alpha = max(self.alpha_min, self.alpha * 0.85)

    def save_qtable(self, path):
        """Persiste Q-table em disco usando pickle."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.qtable, f)

    def load_qtable(self, path):
        """Carrega Q-table previamente salva de disco."""
        import pickle
        with open(path, 'rb') as f:
            self.qtable = pickle.load(f)

    def save_heatmap(self, filename):
        """Exporta mapa de calor das posições visitadas em formato CSV."""
        if not self.position_heatmap:
            return

        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'visits'])
            for (x, y), count in sorted(self.position_heatmap.items()):
                writer.writerow([x, y, count])

class QAgentFarol(QAgentBase):
    """
    Agente Q-Learning especializado para o problema do Farol.

    Estado compacto: (direção_farol, paredes_adjacentes, zona_posicional)
    Otimizado para ambientes com obstáculos e objetivo único.

    Configurações:
        - Epsilon decay lento (0.998) para exploração prolongada
        - Epsilon mínimo de 10% para manter adaptabilidade
        - Inicialização otimista com bias direcional ao farol
    """
    def __init__(self, id='QFarol', lista_acoes=None, modo='learn'):
        if lista_acoes is None:
            lista_acoes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        super().__init__(id=id, lista_acoes=lista_acoes, modo=modo)

        # Configurações otimizadas para ambientes com paredes
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.998
        self.alpha_min = 0.05
        self.alpha_decay = 0.998

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
        """
        Converte observação em estado compacto anti-loop.

        Componentes do estado:
            - direcao_farol: Direção relativa (N/S/E/O/NONE)
            - paredes: Tupla binária (L,R,U,D) indicando obstáculos adjacentes
            - zona: Coordenada discretizada (3×3) para evitar ciclos infinitos

        Espaço de estados: 5 direções × 16 configurações × 9 zonas = 720 estados
        """
        direcao = observacao.get('direcao_farol', 'NONE')

        visao = observacao.get('visao', {})
        paredes = (
            1 if visao.get('L') == 'PAREDE' else 0,
            1 if visao.get('R') == 'PAREDE' else 0,
            1 if visao.get('U') == 'PAREDE' else 0,
            1 if visao.get('D') == 'PAREDE' else 0,
        )

        pos = observacao.get('pos', (0, 0))
        zona_x = 0 if pos[0] < 3 else (1 if pos[0] < 6 else 2)
        zona_y = 0 if pos[1] < 3 else (1 if pos[1] < 6 else 2)
        zona = (zona_x, zona_y)

        return (direcao, paredes, zona)

    def _inicializar_estado(self, estado):
        """
        Inicialização otimista com bias direcional e ruído para exploração.

        Estratégia:
            - Ações alinhadas com o farol: valores 1.5-2.5 (moderadamente otimistas)
            - Ações neutras: valores 0.0-0.5 (permite consideração)
            - Ações bloqueadas por paredes: valores -0.5-0.0 (evita colisões)

        O ruído aleatório garante diversidade nas rotas exploradas inicialmente.
        """
        direcao, paredes, zona = estado

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
                    # Há parede, valor baixo com ruído
                    valores[acao] = random.uniform(-0.5, 0.0)
                else:
                    # Sem parede, otimista MODERADO com ruído
                    # Reduzido de 5.0 para 2.0 + ruído para forçar exploração
                    valores[acao] = random.uniform(1.5, 2.5)
            else:
                # Outras ações começam com valores pequenos aleatórios
                # Isso permite que rotas alternativas sejam consideradas
                valores[acao] = random.uniform(0.0, 0.5)

        return valores


class QAgentForaging(QAgentBase):
    """
    Agente Q-Learning especializado para o problema de Foraging.

    Estado: (carrying, direção_objetivo, no_objetivo, parede_bloqueando)
    Otimizado para tarefas de coleta e entrega com dois objetivos alternados.

    Configurações:
        - Epsilon decay rápido (0.997) para convergência acelerada
        - Epsilon mínimo de 1% para exploração residual mínima
        - Espaço de estados reduzido (40 estados) permite aprendizagem rápida
    """
    def __init__(self, id='QForaging', lista_acoes=None, modo='learn'):
        if lista_acoes is None:
            lista_acoes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        super().__init__(id=id, lista_acoes=lista_acoes, modo=modo)

        # Configurações otimizadas para ambientes com feedback frequente
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.alpha_min = 0.01
        self.alpha_decay = 0.99

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
        """
        Converte observação em estado compacto focado na tarefa.

        Componentes do estado:
            - carrying: Transportando recurso (0/1)
            - direcao_objetivo: Direção ao recurso (se carrying=0) ou ninho (se carrying=1)
            - no_objetivo: Está sobre o alvo atual (True/False)
            - parede_bloqueando: Parede na direção do objetivo (True/False)

        Espaço de estados: 2 × 5 × 2 × 2 = 40 estados (muito compacto)
        """
        carrying = int(observacao.get('carrying', 0))
        visao = observacao.get('visao', {})
        pos = observacao.get('pos', (0, 0))
        nest = observacao.get('nest', (0, 0))

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

        no_objetivo = False
        if carrying == 0:
            no_objetivo = (visao.get('C', 0) == 1)
        else:
            no_objetivo = (pos == nest)

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


class GAAgentBase(AgenteBase):
    """
    Classe base para agentes que utilizam Algoritmo Genético.

    Implementa evolução de políticas através de mutação e seleção baseada em fitness.
    A política é representada por um genoma (pesos lineares) que mapeia
    features de observação para scores de ação.

    Características:
        - População de 30 genomas mantém diversidade genética
        - Seleção elitista: genomas top-k sobrevivem
        - Mutação adaptativa: decresce ao longo do treino
        - Softmax com temperatura decrescente para exploração->exploitação
        - Detecção anti-loop para evitar bloqueios

    Attributes:
        genome: Vetor de pesos (n_actions × feature_dim)
        population: Lista de (fitness, genome) dos melhores indivíduos
        mutation_rate: Probabilidade de mutação por peso
        mutation_scale: Magnitude das mutações
    """
    def __init__(self, id, lista_acoes, feature_dim, modo='learn',
                 mutation_rate=0.2, mutation_scale=0.2):
        super().__init__(id=id, modo=modo)
        self.lista_acoes = list(lista_acoes)
        self.n_actions = len(self.lista_acoes)
        self.feature_dim = feature_dim
        self.mutation_rate = float(mutation_rate)
        self.mutation_scale = float(mutation_scale)

        self.population_size = 30
        self.population = []
        self.episode_count = 0

        self.initial_mutation_rate = float(mutation_rate)
        self.initial_mutation_scale = float(mutation_scale)

        self.genome = [random.uniform(-0.5, 0.5)
                       for _ in range(self.n_actions * self.feature_dim)]
        self.best_genome = list(self.genome)
        self.best_fitness = -math.inf

        self._episode_reward = 0.0
        self.genome_fitness_history = {}

        self.position_heatmap = {}
        self.episode_heatmap = {}

        self.recent_positions = []
        self.stuck_counter = 0

    def _to_features(self, observacao):
        """Converte observação do ambiente em vetor de features para a política."""
        raise NotImplementedError

    # ----- policy evaluation -----

    def _forward(self, features):
        """Calcula scores de ação através de produto matriz-vetor: scores = W × features."""
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
        Seleciona ação usando política softmax com temperatura adaptativa.

        Em modo 'learn':
            - Episódios 0-100: Alta exploração (temperatura 1.0 → 0.2)
            - Episódios 100-200: Transição (temperatura 0.2 → 0.05)
            - Episódios 200+: Quase greedy (temperatura 0.05)
            - Sistema anti-loop: força ação aleatória se detectar bloqueio

        Em outros modos: Seleção greedy determinística.
        """
        if self.ultima_observacao is None:
            acao = random.choice(self.lista_acoes)
            self.regista_passo()
            return acao

        if 'pos' in self.ultima_observacao:
            pos = tuple(self.ultima_observacao['pos'])

            if pos not in self.position_heatmap:
                self.position_heatmap[pos] = 0
            self.position_heatmap[pos] += 1

            if pos not in self.episode_heatmap:
                self.episode_heatmap[pos] = 0
            self.episode_heatmap[pos] += 1

            self.recent_positions.append(pos)
            if len(self.recent_positions) > 8:
                self.recent_positions.pop(0)

            if len(self.recent_positions) >= 4:
                unique_positions = len(set(self.recent_positions))
                recent_4 = self.recent_positions[-4:]
                unique_recent_4 = len(set(recent_4))

                if unique_recent_4 <= 2:
                    self.stuck_counter += 2
                elif len(self.recent_positions) >= 8 and unique_positions <= 3:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = max(0, self.stuck_counter - 1)

                if self.stuck_counter >= 2 and self.modo == "learn":
                    available_actions = [a for a in self.lista_acoes]
                    acao = random.choice(available_actions)
                    self.stuck_counter = 0
                    self.regista_passo()
                    return acao

        feats = self._to_features(self.ultima_observacao)
        if len(feats) != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {len(feats)}")

        scores = self._forward(feats)

        if self.modo == "learn":
            current_episode = len(self.episode_rewards)

            if current_episode < 100:
                progress = current_episode / 100.0
                tau = 1.0 - (0.8 * progress)
            elif current_episode < 200:
                progress = (current_episode - 100) / 100.0
                tau = 0.2 - (0.15 * progress)
            else:
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

class GAAgentFarol(GAAgentBase):
    """
    Agente de Algoritmo Genético especializado para o problema do Farol.
    
    Representação de features (22 dimensões):
        - Posição normalizada (x, y): 2 features
        - Direção ao farol one-hot (N,S,E,O,NONE): 5 features
        - Visão local one-hot (L,R,U,D,C) × (PAREDE,FAROL,OUTRO): 15 features
    
    Inicialização inteligente com bias forte:
        - Favorece movimento na direção do farol (+1.0 a +1.5)
        - Penaliza fortemente colisões com paredes (-4.0 a -5.0)
    """
    def __init__(self, id="GAFarol", lista_acoes=None, modo="learn",
                 mutation_rate=0.1, mutation_scale=0.1):
        if lista_acoes is None:
            lista_acoes = ["UP", "DOWN", "LEFT", "RIGHT"]

        feature_dim = 22

        super().__init__(
            id=id,
            lista_acoes=lista_acoes,
            feature_dim=feature_dim,
            modo=modo,
            mutation_rate=mutation_rate,
            mutation_scale=mutation_scale,
        )

        self._init_intelligent_genome()

    def _init_intelligent_genome(self):
        """
        Inicializa genoma com conhecimento prévio do problema.
        
        Bias estratégico:
            - Direção farol: +1.0 a +1.5 (incentiva movimento correto)
            - Paredes: -4.0 a -5.0 (evita colisões)
            
        Garante desempenho inicial razoável, refinado posteriormente pela evolução.
        """
        for action_idx, action in enumerate(self.lista_acoes):
            for feat_idx in range(self.feature_dim):
                genome_idx = action_idx * self.feature_dim + feat_idx

                if feat_idx == 2 and action == 'UP':
                    self.genome[genome_idx] = random.uniform(1.0, 1.5)
                elif feat_idx == 3 and action == 'DOWN':
                    self.genome[genome_idx] = random.uniform(1.0, 1.5)
                elif feat_idx == 4 and action == 'RIGHT':
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
            # steps=200 → ratio=1.00 → efficiency=0.00 → bonus=~0
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
    Agente de Algoritmo Genético especializado para o problema de Foraging.
    
    Representação de features (10 dimensões):
        - Posição normalizada (x, y): 2 features
        - Visão de recursos (L,R,U,D,C): 5 features (0-1 normalizado)
        - Estado de transporte (carrying): 1 feature (binário)
        - Vetor relativo ao ninho (dx, dy): 2 features
    
    Inicialização com bias duplo-objetivo:
        - Sem recurso: Favorece movimento para recursos visíveis (+4.0 a +5.0)
        - Com recurso: Favorece movimento para o ninho (+3.5 a +4.5)
        - Penaliza movimentos que se afastam dos objetivos (-3.0 a -2.0)
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

        self._init_intelligent_genome()

    def _init_intelligent_genome(self):
        """
        Inicializa genoma com conhecimento do ciclo coleta-entrega.
        
        Comportamento esperado:
            - Fase busca (carrying=0): Siga recursos visíveis
            - Fase entrega (carrying=1): Retorne ao ninho
        
        Bias implementado através de pesos pré-configurados que guiam
        a política inicial, refinada posteriormente pela evolução genética.
        """
        for action_idx, action in enumerate(self.lista_acoes):
            for feat_idx in range(self.feature_dim):
                genome_idx = action_idx * self.feature_dim + feat_idx

                if feat_idx == 2 and action == 'LEFT':
                    self.genome[genome_idx] = random.uniform(4.0, 5.0)
                elif feat_idx == 3 and action == 'RIGHT':
                    self.genome[genome_idx] = random.uniform(4.0, 5.0)
                elif feat_idx == 4 and action == 'UP':
                    self.genome[genome_idx] = random.uniform(4.0, 5.0)
                elif feat_idx == 5 and action == 'DOWN':
                    self.genome[genome_idx] = random.uniform(4.0, 5.0)

                elif feat_idx == 2 and action == 'RIGHT':
                    self.genome[genome_idx] = random.uniform(-3.0, -2.0)
                elif feat_idx == 3 and action == 'LEFT':
                    self.genome[genome_idx] = random.uniform(-3.0, -2.0)
                elif feat_idx == 4 and action == 'DOWN':
                    self.genome[genome_idx] = random.uniform(-3.0, -2.0)
                elif feat_idx == 5 and action == 'UP':
                    self.genome[genome_idx] = random.uniform(-3.0, -2.0)

                # === FASE 2: COM RECURSO → Voltar ao ninho ===

                # Vetor ninho: dx (feat 8)
                if feat_idx == 8:
                    if action == 'RIGHT':
                        # dx > 0 (ninho à direita) → favorece RIGHT
                        self.genome[genome_idx] = random.uniform(3.5, 4.5)
                    elif action == 'LEFT':
                        # dx < 0 (ninho à esquerda) → favorece LEFT
                        self.genome[genome_idx] = random.uniform(-4.5, -3.5)

                # Vetor ninho: dy (feat 9)
                if feat_idx == 9:
                    if action == 'DOWN':
                        # dy > 0 (ninho abaixo) → favorece DOWN
                        self.genome[genome_idx] = random.uniform(3.5, 4.5)
                    elif action == 'UP':
                        # dy < 0 (ninho acima) → favorece UP
                        self.genome[genome_idx] = random.uniform(-4.5, -3.5)

                # Carrying flag (feat 7): modula importância do vetor ninho
                if feat_idx == 7:
                    # Quando carrying=1, amplifica resposta ao vetor ninho
                    if action in ['LEFT', 'RIGHT']:
                        # Amplifica sensibilidade a dx
                        self.genome[genome_idx] = random.uniform(1.5, 2.0)
                    elif action in ['UP', 'DOWN']:
                        # Amplifica sensibilidade a dy
                        self.genome[genome_idx] = random.uniform(1.5, 2.0)

        self.best_genome = list(self.genome)

    @classmethod
    def cria(cls, p):
        """
        Factory compatible with JSON agent config.
        """
        if p is None:
            return cls()
        return cls(
            id=p.get("id", "GAForaging"),
            lista_acoes=p.get("actions", ["UP", "DOWN", "LEFT", "RIGHT"]),
            modo=p.get("mode", "learn"),
            mutation_rate=p.get("mutation_rate", 0.1),
            mutation_scale=p.get("mutation_scale", 0.1),
        )

    def _calc_fitness(self):
        """
        Fitness AGRESSIVO para Foraging: prioriza recursos entregues.

        COMPONENTES:
        1. **Recursos entregues** (×10000 cada) - DOMINANTE
        2. **Eficiência** (recursos/step × 5000) - Bonus grande
        3. **Progresso parcial** (reward acumulado × 1.0) - Marginal
        4. **Penalização por bloqueio** (posições repetidas)

        RANGE ESPERADO:
        - 0 recursos: 0-500 fitness
        - 1 recurso (50 steps): ~10000 + 1000 = 11000
        - 1 recurso (20 steps): ~10000 + 2500 = 12500
        - 3 recursos (100 steps): ~30000 + 1500 = 31500
        - 5 recursos (150 steps): ~50000 + 1666 = 51666
        - 8 recursos (200 steps): ~80000 + 2000 = 82000
        """
        ambiente = getattr(self, "ambiente", None)
        delivered = 0
        if ambiente is not None:
            delivered = getattr(ambiente, "total_delivered", 0)

        steps = max(self._current_steps, 1)

        # === COMPONENTE 1: Recursos entregues (DOMINANTE) ===
        resource_score = delivered * 10000.0

        # === COMPONENTE 2: Eficiência (bonus grande) ===
        if delivered > 0:
            efficiency_ratio = delivered / steps
            # Bonus QUADRÁTICO por alta eficiência
            efficiency_bonus = (efficiency_ratio ** 1.5) * 5000.0
        else:
            efficiency_bonus = 0.0

        # === COMPONENTE 3: Reward parcial (exploração inicial) ===
        # Ajuda agentes que se aproximam mas não entregam ainda
        reward_score = self._episode_reward * 1.0

        # === COMPONENTE 4: Bonus de exploração ===
        exploration_bonus = 0.0
        if hasattr(self, 'episode_heatmap') and self.episode_heatmap:
            unique_cells = len(self.episode_heatmap)
            # Bonus por visitar mais células únicas (incentiva exploração)
            exploration_bonus = unique_cells * 50.0

        # === COMPONENTE 5: Penalização por bloqueio ===
        stuck_penalty = 0.0
        if hasattr(self, 'episode_heatmap') and self.episode_heatmap:
            max_visits = max(self.episode_heatmap.values())
            avg_visits = sum(self.episode_heatmap.values()) / len(self.episode_heatmap)

            # Se visitou mesma célula muitas vezes = loop/bloqueio
            if max_visits > avg_visits * 5:
                # Penalização proporcional ao grau de bloqueio
                stuck_penalty = (max_visits - avg_visits * 2) * 100.0

        fitness = resource_score + efficiency_bonus + reward_score + exploration_bonus - stuck_penalty

        return max(0.0, fitness)

    def _to_features(self, observacao):
        """
        Map Foraging observation dict into a length-10 feature vector.
        observacao keys:
          - "pos": (x,y)
          - "visao": dict L,R,U,D,C -> int (resources or -1 for wall)
          - "carrying": 0/1
          - "nest": (nx,ny)
        """
        # 1) position (x,y) normalized by 10
        x, y = observacao.get("pos", (0, 0))
        pos_x = float(x) / 10.0
        pos_y = float(y) / 10.0

        # 2) local vision: L,R,U,D,C (resources normalized)
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

        return [pos_x, pos_y, vL, vR, vU, vD, vC, carrying, dx, dy]
