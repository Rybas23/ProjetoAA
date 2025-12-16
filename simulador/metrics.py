from collections import defaultdict

# Classe MetricsTracker
# Regista métricas para:
# - ambiente Farol
# - ambiente Foraging
class MetricsTracker:
    def __init__(self, agentes):
        self.agentes = agentes
        self.data = defaultdict(list)

    # Distância Manhattan entre dois pontos
    def _manhattan(self, ponto1, ponto2):
        return abs(ponto1[0] - ponto2[0]) + abs(ponto1[1] - ponto2[1])

    # Registo de métricas específicas do ambiente Farol
    def regista_farol(self, ambiente, recompensa_episodio, steps=None):
        for agente in self.agentes:
            agent_id = agente.id
            posicao_agente = ambiente.agent_pos.get(agent_id, None)

            if posicao_agente is None:
                # Agente não existe mais — registar valores vazios
                self.data[f"dist_final_{agent_id}"].append(None)
                self.data[f"sucesso_{agent_id}"].append(0)
            else:
                distancia = self._manhattan(posicao_agente, ambiente.farol)
                self.data[f"dist_final_{agent_id}"].append(distancia)
                self.data[f"sucesso_{agent_id}"].append(
                    1 if posicao_agente == ambiente.farol else 0
                )

        if steps is not None:
            self.data["steps_ep"].append(steps)

        # Também guardar recompensa total por agente, se fornecido
        for agente in self.agentes:
            agent_id = agente.id
            self.data[f"reward_{agent_id}"].append(
                recompensa_episodio.get(agent_id, 0)
            )

    # Registo de métricas específicas do ambiente Foraging
    def regista_foraging(self, ambiente, recompensa_episodio, steps=None):
        # Total de recursos entregues no episódio
        self.data["resources_delivered"].append(
            getattr(ambiente, "total_delivered", 0)
        )

        if steps is not None:
            self.data["steps_ep"].append(steps)

        # Guarda recompensa total por agente
        for agente in self.agentes:
            agent_id = agente.id
            self.data[f"reward_{agent_id}"].append(
                recompensa_episodio.get(agent_id, 0)
            )
