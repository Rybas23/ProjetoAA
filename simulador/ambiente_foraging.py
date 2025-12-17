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
        ninho=(0, 0),
        paredes=None,
        recursos=None,
    ):

        self.width = width                   # Largura da grelha
        self.height = height                 # Altura da grelha
        self.ninho = ninho                   # Posição do ninho

        self.step = 0
        self.agent_ids = []
        self.agent_pos = {}                  # Posição dos agentes
        self.carrying = {}                  # 1 ou 0 (se carrega recurso)
        # guardar configuração inicial dos recursos para poder repor em cada episódio
        self.initial_resources = {tuple(r) for r in (recursos or [])}
        # estado corrente de recursos (set mutável)
        self.resources = set(self.initial_resources)
        self.walls = {tuple(w) for w in (paredes or [])}

    # Registar agentes
    def registar_agentes(self, agentes):
        self.agent_ids = [ag.id for ag in agentes]

    # Reiniciar episódio
    def reset(self):
        self.step = 0

        # repor recursos iniciais em cada novo episódio
        self.resources = set(self.initial_resources)

        # Colocar agentes no ninho
        for aid in self.agent_ids:
            self.agent_pos[aid] = self.ninho
            self.carrying[aid] = 0

        self.total_delivered = 0  # Contagem global de entregas

        return self._state()

    # Estado global do ambiente
    def _state(self):
        return {
            "agents": dict(self.agent_pos),
            "resources": list(self.resources),
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
                        # 1 se existir recurso, 0 caso contrário
                        vis[k] = 1 if (nx, ny) in self.resources else 0
                else:
                    vis[k] = -1  # fora da grelha tratado como parede

            # recurso no tile atual (1 ou 0)
            vis["C"] = 1 if (x, y) in self.resources else 0
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
        # penalização base por passo
        recompensa = -0.01

        # guardar posição anterior para calcular distância manhattan
        x_old, y_old = self.agent_pos.get(ag_id, self.ninho)
        pos_old = (x_old, y_old)

        carrying_before = self.carrying.get(ag_id, 0)

        # 1) Movimento
        if acao in ["UP", "DOWN", "LEFT", "RIGHT"]:
            x, y = pos_old
            novo_x, novo_y = x, y

            if acao == "UP" and y > 0:
                novo_y -= 1
            elif acao == "DOWN" and y < self.height - 1:
                novo_y += 1
            elif acao == "LEFT" and x > 0:
                novo_x -= 1
            elif acao == "RIGHT" and x < self.width - 1:
                novo_x += 1

            # Verificar parede: se destino for parede, não mexe e penaliza mais
            if (novo_x, novo_y) in self.walls:
                return -0.2, False

            # Atualiza posição
            self.agent_pos[ag_id] = (novo_x, novo_y)

        # 2) Ações de interação com recursos (PICK/DROP)
        elif acao == "PICK":
            x, y = self.agent_pos[ag_id]
            tipo = self._tipo_celula(x, y)
            tem_recurso = tipo == "recurso"
            livre = self.carrying[ag_id] == 0

            if tem_recurso and livre:
                # PICK válido → remove recurso dessa célula
                self.resources.discard((x, y))
                self.carrying[ag_id] = 1
                recompensa = 0.5
            else:
                # PICK inválido (sem recurso ou já a carregar)
                recompensa = -2

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
                recompensa = -2

        else:
            # Ação inválida
            recompensa = -0.1

        # ---------- DISTÂNCIA DE MANHATTAN ----------
        # Depois de aplicar a ação (ou não, se for PICK/DROP), calcular mudança de distância
        pos_new = self.agent_pos.get(ag_id, pos_old)

        shaping = 0.0

        # Se estiver a carregar recurso → queremos aproximar do ninho
        if carrying_before == 1:
            d_old = self._manhattan(pos_old, self.ninho)
            d_new = self._manhattan(pos_new, self.ninho)
            if d_new < d_old:
                shaping += 0.05  # prémio pequeno por aproximar
            elif d_new > d_old:
                shaping -= 0.05  # penalização por afastar

        # Se não estiver a carregar → queremos aproximar de algum recurso
        else:
            d_old = self._dist_to_closest_resource(pos_old)
            d_new = self._dist_to_closest_resource(pos_new)
            if d_old is not None and d_new is not None:
                if d_new < d_old:
                    shaping += 0.02
                elif d_new > d_old:
                    shaping -= 0.02

        recompensa += shaping

        return recompensa, False

    # Atualizar contador global
    def atualizacao(self):
        self.step += 1

    # Fim do episódio
    def is_episode_done(self):
        # Episódio termina quando não houver mais recursos;
        # o limite máximo de passos é controlado pelo motor.
        return len(self.resources) == 0

    def _manhattan(self, a, b):
        ax, ay = a
        bx, by = b
        return abs(ax - bx) + abs(ay - by)

    def _dist_to_closest_resource(self, pos):
        if not self.resources:
            return None
        return min(self._manhattan(pos, r) for r in self.resources)
