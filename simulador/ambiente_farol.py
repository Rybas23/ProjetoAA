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
    def __init__(self, size=10, max_steps=200, farol_fixo=None, paredes=None):
        self.size = size                     # Tamanho da grelha NxN
        self.max_steps = max_steps           # Limite máximo de passos
        self.farol = farol_fixo or (size // 2, size // 2)  # Posição do farol

        # Conjunto de posições ocupadas por paredes/obstáculos
        # Cada parede é um tuplo (x, y)
        self.walls = set(paredes or [])

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
    # Reinicia o ambiente e posiciona agentes com spawn fixo
    # ------------------------------------------------------------
    def reset(self, agent_spawns=None):
        self.step = 0
        self.done_agents = set()

        # Todas as posições possíveis excepto a posição do farol e paredes
        posicoes_disponiveis = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) != self.farol and (x, y) not in self.walls
        ]
        random.shuffle(posicoes_disponiveis)

        self.agent_pos = {}
        agent_spawns = agent_spawns or {}

        # Primeiro, aplica spawns se definidos e válidos
        usados = set()
        for agent_id in self.agent_ids:
            if agent_id in agent_spawns:
                sx, sy = agent_spawns[agent_id]
                # Usa o spawn apenas se estiver dentro da grelha e não for parede/farol
                if (
                    0 <= sx < self.size
                    and 0 <= sy < self.size
                    and (sx, sy) != self.farol
                    and (sx, sy) not in self.walls
                ):
                    self.agent_pos[agent_id] = (sx, sy)
                    usados.add((sx, sy))

        # Depois, preenche restantes agentes com posições aleatórias livres
        livres = [p for p in posicoes_disponiveis if p not in usados]
        for agent_id in self.agent_ids:
            if agent_id not in self.agent_pos:
                if not livres:
                    raise RuntimeError("Sem posições livres suficientes para todos os agentes")
                self.agent_pos[agent_id] = livres.pop()

        return self._state()

    # Estado global do ambiente
    def _state(self):
        return {
            "farol": self.farol,
            "agents": dict(self.agent_pos),
            "walls": list(self.walls),
        }

    # Gera observação para um agente específico
    def observacaoPara(self, agente):
        (x, y) = self.agent_pos[agente.id]
        observacao = {"pos": (x, y)}

        for sensor in agente.sensores:
            if sensor.tipo == "farol":
                observacao["direcao_farol"] = self._dir((x, y)).value

            if sensor.tipo == "visao":
                observacao["visao"] = self._visao(x, y, sensor.alcance)

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
            "L": (x - 1, y),
            "R": (x + 1, y),
            "U": (x, y - 1),
            "D": (x, y + 1),
            "C": (x, y),
        }

        for chave, (nx, ny) in vizinhos.items():
            if 0 <= nx < self.size and 0 <= ny < self.size:

                # Posição é uma parede
                if (nx, ny) in self.walls:
                    resultado[chave] = "PAREDE"
                    continue

                # Farol na visão
                if (nx, ny) == self.farol:
                    resultado[chave] = "FAROL"
                    continue

                # Verificar se existe algum agente na posição
                encontrado = None
                for ag_id, pos in self.agent_pos.items():
                    if pos == (nx, ny):
                        encontrado = f"AG_{ag_id}"
                        break

                resultado[chave] = encontrado or "VAZIO"

            else:
                resultado[chave] = "PAREDE"

        return resultado

    def _dist_manhattan(self, pos):
        x, y = pos
        fx, fy = self.farol
        return abs(fx - x) + abs(fy - y)

    def _efeito_celula(self, x, y):
        """Devolve o tipo de célula e efeitos básicos de recompensa/bloqueio."""
        # Paredes
        if (x, y) in self.walls:
            return {
                "tipo": "parede",
                "recompensa": -0.2,
                "bloqueia": True,
            }

        # Farol
        if (x, y) == self.farol:
            return {
                "tipo": "farol",
                "recompensa": 10.0,
                "bloqueia": False,
            }

        # Vazio
        return {
            "tipo": "vazio",
            "recompensa": 0.0,
            "bloqueia": False,
        }

    def agir(self, acao, agente):
        agente_id = agente.id

        # Se já terminou, devolve recompensa neutra e não se mexe mais
        if agente_id in self.done_agents:
            return 0.0, True

        x, y = self.agent_pos[agente_id]
        novo_x, novo_y = x, y

        terminou = False

        # Distância antes do movimento
        dist_antes = self._dist_manhattan((x, y))

        # 1) Propor novo movimento (ainda sem aplicar)
        if acao == "UP" and y > 0:
            novo_y = y - 1
        elif acao == "DOWN" and y < self.size - 1:
            novo_y = y + 1
        elif acao == "LEFT" and x > 0:
            novo_x = x - 1
        elif acao == "RIGHT" and x < self.size - 1:
            novo_x = x + 1
        else:
            # Ação inválida (inclui qualquer coisa que não seja movimento)
            return -0.05, False

        # 2) Consultar efeito da célula de destino
        efeito = self._efeito_celula(novo_x, novo_y)

        # Se for parede, não se mexe e aplica apenas a recompensa da parede
        if efeito["tipo"] == "parede" and efeito.get("bloqueia", False):
            self.agent_pos[agente_id] = (x, y)
            return efeito["recompensa"], False

        # 3) Aplicar movimento
        self.agent_pos[agente_id] = (novo_x, novo_y)

        # 4) Se chegou ao farol → episódio termina para este agente
        if efeito["tipo"] == "farol":
            recompensa = efeito["recompensa"]
            self.done_agents.add(agente_id)
            terminou = True
            return recompensa, terminou

        # 5) Recompensa por aproximação real ao farol (shaping adicional)
        dist_depois = self._dist_manhattan((novo_x, novo_y))

        if dist_depois < dist_antes:
            recompensa = +0.05  # aproximou\-se
        elif dist_depois > dist_antes:
            recompensa = -0.05  # afastou\-se
        else:
            recompensa = -0.01  # custo neutro por não mudar distância

        # Soma eventual recompensa base do tipo de célula (normalmente 0)
        if efeito["tipo"] == "vazio":
            recompensa += efeito.get("recompensa", 0.0)

        return recompensa, terminou

    # Atualiza o contador global
    def atualizacao(self):
        self.step += 1

    # Condição de fim do episódio
    def is_episode_done(self):
        return self.step >= self.max_steps or len(self.done_agents) == len(self.agent_ids)
