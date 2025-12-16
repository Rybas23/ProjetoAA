import random
from enum import Enum

class AgenteBase:
    def __init__(self, id, modo='test'):
        self.id = id                     # Nome ou identificador do agente
        self.modo = modo                 # 'test' ou 'learn'
        self.ambiente = None             # Referência ao ambiente
        self.sensores = []               # Lista de sensores instalados
        self.verbose = False             # Debug
        self.ultima_observacao = None    # Última observação recebida

    @classmethod
    def cria(cls, p):
        raise NotImplementedError

    def instala_ambiente(self, ambiente):
        self.ambiente = ambiente

    def instala(self, sensor):
        self.sensores.append(sensor)

    def observacao(self, observacao_dict):
        self.ultima_observacao = observacao_dict
        if self.verbose:
            print(f"[{self.id}] observação: {observacao_dict}")

    def comunica(self, mensagem, agente_origem):
        if self.verbose:
            print(f"[{self.id}] recebeu msg '{mensagem}' de {agente_origem.id}")

    def age(self):
        raise NotImplementedError

    def avaliacaoEstadoAtual(self, recompensa):
        """Chamado após o ambiente dar a recompensa."""
        pass

    def reset(self, episodio):
        """Chamado no início de cada episódio."""
        pass

#  AGENTE FIXO (NÃO APRENDE)
class FixedAgent(AgenteBase):
    def __init__(self, id, politica, modo='test'):
        super().__init__(id, modo)
        self.politica = politica

    @classmethod
    def cria(cls, p):
        # Por omissão, usar uma política que se move sempre 'UP'
        return cls('fixed', lambda o: 'UP')

    def age(self):
        return self.politica(self.ultima_observacao or {})


#  BASE DE UM AGENTE COM Q-LEARNING
class QAgentBase(AgenteBase):
    def __init__(self, id, lista_acoes,
                 alpha=0.4, gamma=0.95,
                 epsilon=0.2, modo='learn'):

        super().__init__(id, modo)

        # Hiperparâmetros do Q-learning
        self.acoes = lista_acoes      # Lista de ações válidas no ambiente
        self.alpha = alpha            # Taxa de aprendizagem
        self.gamma = gamma            # Fator de desconto do futuro
        self.epsilon = epsilon        # Probabilidade de explorar (ε-greedy)

        # Estrutura principal de aprendizagem
        # Q-table: dict[state][action] = valor-Q
        self.qtable = {}

        # Guardar a última decisão tomada pelo agente
        self.estado_anterior = None
        self.acao_anterior = None

    # ESCOLHER UMA AÇÃO (ε-greedy)
    def age(self):
        observacao_atual = self.ultima_observacao
        estado_atual = self._to_state(observacao_atual)

        # Garante que o estado existe na Q-table
        if estado_atual not in self.qtable:
            self.qtable[estado_atual] = {
                acao: 0.0 for acao in self.acoes
            }

        # Política ε-greedy
        if self.modo == 'learn' and random.random() < self.epsilon:
            # EXPLORAÇÃO → escolhe ação aleatória
            acao_escolhida = random.choice(self.acoes)
        else:
            # EXPLORAÇÃO → escolhe a ação com maior Q
            valor_maximo = max(self.qtable[estado_atual].values())
            melhores_acoes = [
                acao for acao, q in self.qtable[estado_atual].items()
                if q == valor_maximo
            ]
            acao_escolhida = random.choice(melhores_acoes)

        # Guarda o que fez para poder aprender com o resultado
        self.estado_anterior = estado_atual
        self.acao_anterior = acao_escolhida

        return acao_escolhida

    # -------------------------------------------------------------
    # ATUALIZAÇÃO DO Q-LEARNING (RECEBE RECOMPENSA)
    # -------------------------------------------------------------
    def avaliacaoEstadoAtual(self, recompensa):
        """Atualiza o valor Q(s, a) seguindo a fórmula do Q-learning."""

        if self.modo != 'learn':
            return

        estado_passado = self.estado_anterior
        acao_passada = self.acao_anterior

        # Se o agente ainda não tomou nenhuma ação (primeira iteração)
        if estado_passado is None or acao_passada is None:
            return

        # Determina o novo estado (depois da ação anterior)
        estado_atual = self._to_state(self.ultima_observacao)

        # Garante existência do estado atual na Q-table
        if estado_atual not in self.qtable:
            self.qtable[estado_atual] = {a: 0.0 for a in self.acoes}

        # Máximo valor futuro — Q(s', a')
        valor_maximo_proximo = max(self.qtable[estado_atual].values())

        # Fórmula do Q-learning:
        # Q(s,a) ← Q(s,a) + α * (r + γ max_a' Q(s',a') − Q(s,a))
        valor_antigo = self.qtable[estado_passado][acao_passada]
        valor_novo = valor_antigo + self.alpha * (
            recompensa + self.gamma * valor_maximo_proximo - valor_antigo
        )

        self.qtable[estado_passado][acao_passada] = valor_novo

    # DECAY DO EPSILON A CADA EPISÓDIO
    def reset(self, episodio):
        if self.modo == 'learn':
            # Diminui o epsilon gradualmente até um mínimo
            self.epsilon = max(0.05, self.epsilon * 0.995)


#  Q-AGENT PARA O AMBIENTE FAROL (FarolEnv)
class QAgentFarol(QAgentBase):

    @classmethod
    def cria(cls, p):
        # Ações válidas no ambiente Farol
        return cls(
            id='q_farol',
            lista_acoes=['UP', 'DOWN', 'LEFT', 'RIGHT']
        )

    def _to_state(self, observacao):
        """Transforma a observação em um estado discreto (tupla)."""

        if not observacao:
            return ('NONE',)

        # Verifica se a observação contém dados do farol
        if 'direcao_farol' in observacao and 'pos' in observacao:
            (x_ag, y_ag) = observacao['pos']
            (x_f, y_f) = self.ambiente.farol  # posição do farol

            dx = x_f - x_ag
            dy = y_f - y_ag

            # Quantização da direção → {-1,0,1}
            dx = 1 if dx > 0 else -1 if dx < 0 else 0
            dy = 1 if dy > 0 else -1 if dy < 0 else 0

            return ('FAROL', dx, dy)

        return ('GEN', str(observacao))


#  Q-AGENT PARA O AMBIENTE DE FORAGING (ForagingEnv)
class QAgentForaging(QAgentBase):

    @classmethod
    def cria(cls, p):
        return cls(
            id='q_forage',
            lista_acoes=['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        )

    def _to_state(self, observacao):
        if not observacao:
            return ('NONE',)

        # Ambiente de Foraging fornece uma visão local
        if 'visao' in observacao:
            vis = observacao['visao']

            # Função auxiliar para verificar se há recurso
            def tem_recurso(v):
                if isinstance(v, int):
                    return v > 0
                return v == 'FAROL'  # compatível com ambientes mistos

            # Codificação binária da visão
            L = int(tem_recurso(vis.get('L')))
            R = int(tem_recurso(vis.get('R')))
            U = int(tem_recurso(vis.get('U')))
            D = int(tem_recurso(vis.get('D')))
            C = int(tem_recurso(vis.get('C')))

            carregando = observacao.get('carrying', 0)

            # Estado final (tupla → bom para Q-table)
            return ('FORAGE', L, R, U, D, C, carregando)

        # Caso genérico (fallback)
        return ('GEN', str(observacao))
