#  AMBIENTE DE FORAGING (ForagingEnv)
#  - Agentes recolhem recursos espalhados na grelha.
#  - Podem carregar 1 recurso de cada vez.
#  - Ganham recompensa ao entregar recursos no ninho.


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
    def reset(self, agent_spawns=None):
        self.step = 0

        # repor recursos iniciais em cada novo episódio
        self.resources = set(self.initial_resources)

        # Colocar agentes nas posições de spawn (ou ninho por padrão)
        for aid in self.agent_ids:
            if agent_spawns and aid in agent_spawns:
                self.agent_pos[aid] = agent_spawns[aid]
            else:
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

        # SensorRecursoMaisProximo → indica direção do recurso mais próximo
        if "recurso_mais_proximo" in tipos_sensores:
            if len(self.resources) > 0:
                # Encontrar recurso mais próximo por distância Manhattan
                recursos_com_dist = [
                    (self._manhattan((x, y), r), r) for r in self.resources
                ]
                dist_min, recurso_proximo = min(recursos_com_dist, key=lambda t: t[0])
                rx, ry = recurso_proximo

                # Calcular direção relativa (prioridade: horizontal depois vertical)
                if rx > x:
                    direcao_recurso = "E"  # Este
                elif rx < x:
                    direcao_recurso = "O"  # Oeste
                elif ry < y:
                    direcao_recurso = "N"  # Norte
                elif ry > y:
                    direcao_recurso = "S"  # Sul
                else:
                    direcao_recurso = "NONE"  # Agente está em cima do recurso
            else:
                direcao_recurso = "NONE"  # Sem recursos disponíveis

            obs["direcao_recurso"] = direcao_recurso

        return obs

    # Executar ação do agente
    def agir(self, acao, agente):
        ag_id = agente.id
        recompensa = 0

        # guardar posição anterior
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

            # Verificar parede: se destino for parede, não mexe
            if (novo_x, novo_y) in self.walls:
                return -0.01, False

            # Atualiza posição
            self.agent_pos[ag_id] = (novo_x, novo_y)

            # Penalização por movimento
            recompensa = -0.01

        else:
            # Ação inválida (não devia acontecer)
            return 0.0, False

        # 2) LÓGICA AUTOMÁTICA: PICK e DROP após movimento
        x_new, y_new = self.agent_pos[ag_id]
        tipo = self._tipo_celula(x_new, y_new)

        # PICK: se estiver em cima de recurso e não estiver a carregar
        if tipo == "recurso" and self.carrying[ag_id] == 0:
            self.resources.discard((x_new, y_new))
            self.carrying[ag_id] = 1
            recompensa += 2.0

        # DROP: se estiver no ninho e estiver a carregar
        elif tipo == "ninho" and self.carrying[ag_id] == 1:
            self.carrying[ag_id] = 0
            self.total_delivered += 1
            recompensa += 5.0

        # 3) REWARD SHAPING: Recompensa por aproximar-se do objetivo
        shaping = 0.0

        # Determinar objetivo baseado no estado de carrying ANTES do movimento
        if carrying_before == 0:
            # Não estava a carregar → objetivo era o recurso mais próximo
            # IMPORTANTE: usar self.resources + recurso que acabou de apanhar (se aplicável)
            recursos_para_calculo = set(self.resources)

            # Se apanhou recurso, adicionar de volta para cálculo justo
            if tipo == "recurso" and self.carrying[ag_id] == 1:
                recursos_para_calculo.add((x_new, y_new))

            if len(recursos_para_calculo) > 0:
                # Distância antes do movimento
                dist_antes = min(self._manhattan(pos_old, r) for r in recursos_para_calculo)
                # Distância depois do movimento
                dist_depois = min(self._manhattan((x_new, y_new), r) for r in recursos_para_calculo)

                # Recompensa se aproximou, penalização se afastou
                if dist_depois < dist_antes:
                    shaping = 0.05  # Reduzido de 0.1 para 0.05
                elif dist_depois > dist_antes:
                    shaping = -0.02  # Reduzido de -0.05 para -0.02
        else:
            # Estava a carregar → objetivo era o ninho
            dist_antes = self._manhattan(pos_old, self.ninho)
            dist_depois = self._manhattan((x_new, y_new), self.ninho)

            # Recompensa se aproximou, penalização se afastou
            if dist_depois < dist_antes:
                shaping = 0.05  # Reduzido de 0.1 para 0.05
            elif dist_depois > dist_antes:
                shaping = -0.02  # Reduzido de -0.05 para -0.02

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

