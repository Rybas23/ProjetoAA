#  AMBIENTE FAROL (FarolEnv)
#  - Ambiente simples onde agentes tentam alcançar um farol.
#  - O farol é um ponto fixo na grelha.
#  - Agentes movem-se e recebem recompensa ao alcançá-lo.

import random
from enum import Enum


class TipoDirecao(Enum):
    N = "N"      # Norte
    S = "S"      # Sul
    E = "E"      # Este
    O = "O"      # Oeste
    NONE = "NONE"


class FarolEnv:
    def __init__(self, size=10, max_steps=200, farol_fixo=None):
        self.size = size                     # Tamanho da grelha NxN
        self.max_steps = max_steps           # Limite máximo de passos
        self.farol = farol_fixo or (size//2, size//2)  # Posição do farol

        self.step = 0                        # Passo atual do episódio
        self.agent_ids = []                  # IDs dos agentes registados
        self.agent_pos = {}                  # Posição atual de cada agente
        self.done_agents = set()             # Agentes que já chegaram ao farol

    # ------------------------------------------------------------
    # Regista os agentes no ambiente
    # ------------------------------------------------------------
    def registar_agentes(self, agentes):
        self.agent_ids = [ag.id for ag in agentes]

    # ------------------------------------------------------------
    # Reinicia o ambiente e posiciona agentes em locais aleatórios
    # ------------------------------------------------------------
    def reset(self):
        self.step = 0
        self.done_agents = set()

        # Todas as posições possíveis excepto a posição do farol
        posicoes_disponiveis = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) != self.farol
        ]
        random.shuffle(posicoes_disponiveis)

        # Distribui os agentes pelas posições aleatórias
        for i, agent_id in enumerate(self.agent_ids):
            self.agent_pos[agent_id] = posicoes_disponiveis[i]

        return self._state()

    # Estado global do ambiente
    def _state(self):
        return {
            'farol': self.farol,
            'agents': dict(self.agent_pos)
        }

    # Gera observação para um agente específico
    def observacaoPara(self, agente):
        (x, y) = self.agent_pos[agente.id]
        observacao = {'pos': (x, y)}

        for sensor in agente.sensores:
            if sensor.tipo == 'farol':
                observacao['direcao_farol'] = self._dir((x, y)).value

            if sensor.tipo == 'visao':
                observacao['visao'] = self._visao(x, y, sensor.alcance)

        return observacao

    # Indica direção relativa do farol
    def _dir(self, pos):
        xa, ya = pos
        xf, yf = self.farol

        if xf > xa:
            return TipoDirecao.E
        if xf < xa:
            return TipoDirecao.O
        if yf > ya:
            return TipoDirecao.S
        if yf < ya:
            return TipoDirecao.N

        return TipoDirecao.NONE

    # Retorna visão local do agente (L,R,U,D,C)
    def _visao(self, x, y, alcance):
        resultado = {}

        vizinhos = {
            'L': (x - 1, y),
            'R': (x + 1, y),
            'U': (x, y - 1),
            'D': (x, y + 1),
            'C': (x, y)
        }

        for chave, (nx, ny) in vizinhos.items():
            if 0 <= nx < self.size and 0 <= ny < self.size:

                # Farol na visão
                if (nx, ny) == self.farol:
                    resultado[chave] = 'FAROL'
                    continue

                # Verificar se existe algum agente na posição
                encontrado = None
                for ag_id, pos in self.agent_pos.items():
                    if pos == (nx, ny):
                        encontrado = f"AG_{ag_id}"
                        break

                resultado[chave] = encontrado or 'VAZIO'

            else:
                resultado[chave] = 'PAREDE'

        return resultado

    # Aplica uma ação ao agente
    def agir(self, acao, agente):
        agente_id = agente.id

        # Se já terminou, devolve recompensa neutra
        if agente_id in self.done_agents:
            return 0.0, True

        x, y = self.agent_pos[agente_id]

        recompensa = -0.01   # pequeno custo por movimento
        terminou = False

        # Movimentos válidos
        if acao == 'UP' and y > 0:
            y -= 1
        elif acao == 'DOWN' and y < self.size - 1:
            y += 1
        elif acao == 'LEFT' and x > 0:
            x -= 1
        elif acao == 'RIGHT' and x < self.size - 1:
            x += 1
        elif acao == 'STAY':
            pass
        else:
            recompensa = -0.1  # penalização por ação inválida

        self.agent_pos[agente_id] = (x, y)

        # Recompensa ao chegar ao farol
        if (x, y) == self.farol:
            recompensa = 10.0
            self.done_agents.add(agente_id)
            terminou = True

        return recompensa, terminou

    # Atualiza o contador global
    def atualizacao(self):
        self.step += 1

    # Condição de fim do episódio
    def is_episode_done(self):
        return (
            self.step >= self.max_steps or
            len(self.done_agents) == len(self.agent_ids)
        )
