#  AMBIENTE DE FORAGING (ForagingEnv)
#  - Agentes recolhem recursos espalhados na grelha.
#  - Podem carregar 1 recurso de cada vez.
#  - Ganham recompensa ao entregar recursos no ninho.

import random


class ForagingEnv:
    def __init__(
        self,
        width=10,
        height=10,
        n_resources=10,
        ninho=(0, 0),
        max_steps=200,
        paredes=None,
    ):

        self.width = width                   # Largura da grelha
        self.height = height                 # Altura da grelha
        self.n_resources = n_resources       # Nº inicial de recursos
        self.ninho = ninho                   # Posição do ninho
        self.max_steps = max_steps

        self.step = 0
        self.agent_ids = []
        self.agent_pos = {}                  # Posição dos agentes
        self.carrying = {}                  # 1 ou 0 (se carrega recurso)
        self.resources = {}                 # (x,y) → quantidade no tile
        # Novo: paredes/obstáculos fixos no mapa
        self.walls = set(paredes or [])

    # Registar agentes
    def registar_agentes(self, agentes):
        self.agent_ids = [ag.id for ag in agentes]

    # Reiniciar episódio
    def reset(self):
        self.step = 0
        self.resources = {}

        # Espalhar recursos aleatoriamente
        for _ in range(self.n_resources):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.resources[(x, y)] = self.resources.get((x, y), 0) + 1

        # Colocar agentes no ninho
        for aid in self.agent_ids:
            self.agent_pos[aid] = self.ninho
            self.carrying[aid] = 0

        self.total_delivered = 0  # Contagem global de entregas

        return self._state()

    # Estado global do ambiente
    def _state(self):
        return {
            "resources": dict(self.resources),
            "agents": dict(self.agent_pos),
            "ninho": self.ninho,
            "walls": list(self.walls),
        }

    def _tipo_celula(self, x, y):
        """Classifica a célula para efeitos lógicos (ninho, recurso, parede, vazio)."""
        if (x, y) == self.ninho:
            return "ninho"
        if (x, y) in self.walls:
            return "parede"
        if (x, y) in self.resources:
            return "recurso"
        return "vazio"

    # Observação para agente
    def observacaoPara(self, agente):
        x, y = self.agent_pos[agente.id]

        obs = {
            "pos": (x, y),
        }

        # Descobrir que sensores este agente tem
        tipos_sensores = {s.tipo for s in getattr(agente, "sensores", [])}

        # Sensor de visão → fornece mapa de recursos ao redor
        if "visao" in tipos_sensores:
            vis = {}
            direcoes = [
                (-1, 0, "L"),
                (1, 0, "R"),
                (0, -1, "U"),
                (0, 1, "D"),
            ]

            for dx, dy, k in direcoes:
                nx, ny = x + dx, y + dy

                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) in self.walls:
                        vis[k] = -1  # parede
                    else:
                        vis[k] = self.resources.get((nx, ny), 0)
                else:
                    vis[k] = -1  # fora da grelha tratado como parede

            vis["C"] = self.resources.get((x, y), 0)  # recurso no tile atual
            obs["visao"] = vis

        # SensorCarregando → informa se está a carregar recurso
        if "carregando" in tipos_sensores:
            obs["carrying"] = self.carrying[agente.id]

        # SensorNinho → informa posição do ninho
        if "ninho" in tipos_sensores:
            obs["nest"] = self.ninho

        return obs

    # Executar ação do agente
    def agir(self, acao, agente):
        ag_id = agente.id
        recompensa = -0.01

        # 1) Movimento (sem STAY)
        if acao in ["UP", "DOWN", "LEFT", "RIGHT"]:
            x, y = self.agent_pos[ag_id]
            novo_x, novo_y = x, y

            if acao == "UP" and y > 0:
                novo_y -= 1
            elif acao == "DOWN" and y < self.height - 1:
                novo_y += 1
            elif acao == "LEFT" and x > 0:
                novo_x -= 1
            elif acao == "RIGHT" and x < self.width - 1:
                novo_x += 1

            # Verificar parede: se destino for parede, não mexe e penaliza um pouco
            if (novo_x, novo_y) in self.walls:
                return -0.1, False

            # Atualiza posição
            self.agent_pos[ag_id] = (novo_x, novo_y)

        # 2) Ações de interação com recursos (PICK/DROP)
        elif acao == "PICK":
            x, y = self.agent_pos[ag_id]
            tipo = self._tipo_celula(x, y)
            tem_recurso = tipo == "recurso" and self.resources.get((x, y), 0) > 0
            livre = self.carrying[ag_id] == 0

            if tem_recurso and livre:
                # PICK válido
                self.resources[(x, y)] -= 1
                if self.resources[(x, y)] == 0:
                    del self.resources[(x, y)]
                self.carrying[ag_id] = 1
                recompensa = 1.0
            else:
                # PICK inválido (sem recurso ou já a carregar)
                recompensa = -0.1

        elif acao == "DROP":
            x, y = self.agent_pos[ag_id]
            tipo = self._tipo_celula(x, y)
            no_ninho = tipo == "ninho"
            a_carregar = self.carrying[ag_id] == 1

            if no_ninho and a_carregar:
                # DROP válido
                self.carrying[ag_id] = 0
                self.total_delivered += 1
                recompensa = 5.0
            else:
                # DROP inválido (fora do ninho ou sem recurso)
                recompensa = -0.1

        else:
            # Ação inválida
            recompensa = -0.05

        return recompensa, False

    # Atualizar contador global
    def atualizacao(self):
        self.step += 1

    # Fim do episódio
    def is_episode_done(self):
        return self.step >= self.max_steps or len(self.resources) == 0
