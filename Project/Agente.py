from typing import Optional, List
from numpy import double
from Project.Accao import Accao
from Project.IAgente import IAgente
from Project.Observacao import Observacao
from Project.Sensor import Sensor

class Agente(IAgente):
    def __init__(self):
        self.nome: Optional[str] = None
        self.sensor: Optional[Sensor] = None
        self.ultima_observacao: Optional[Observacao] = None
        self.ultima_recompensa: Optional[float] = None
        self.mensagens_recebidas: List[str] = []

    # -------------------------------------------------------
    # Inicializa o agente com ficheiro de parâmetros
    # -------------------------------------------------------
    def agenteCria(self, nome_do_ficheiro_parametros: str) -> None:
        print(f"[Agente] Inicializado com parâmetros de: {nome_do_ficheiro_parametros}")
        self.nome = nome_do_ficheiro_parametros

    # -------------------------------------------------------
    # Recebe uma observação do ambiente
    # -------------------------------------------------------
    def observação(self, obs: Observacao) -> None:
        print(f"[Agente] Observação recebida: {obs}")
        self.ultima_observacao = obs

    # -------------------------------------------------------
    # Decide e executa uma ação
    # -------------------------------------------------------
    def accaoAge(self) -> Accao:
        print(f"[Agente] A agir com base na última observação: {self.ultima_observacao}")
        # Aqui podes implementar lógica de decisão real
        acao = Accao()  # Placeholder
        return acao

    # -------------------------------------------------------
    # Recebe a recompensa do ambiente
    # -------------------------------------------------------
    def avaliacaoEstadoAtual(self, recompensa: double) -> None:
        print(f"[Agente] Recompensa recebida: {recompensa}")
        self.ultima_recompensa = float(recompensa)

    # -------------------------------------------------------
    # Instala um sensor no agente
    # -------------------------------------------------------
    def instala(self, sensor: Sensor) -> None:
        print(f"[Agente] Sensor instalado: {sensor}")
        self.sensor = sensor

    # -------------------------------------------------------
    # Recebe mensagem de outro agente
    # -------------------------------------------------------
    def comunica(self, mensagem: str, de_agente: "Agente") -> None:
        print(f"[Agente] Mensagem recebida de {de_agente.nome}: {mensagem}")
        self.mensagens_recebidas.append(f"{de_agente.nome}: {mensagem}")