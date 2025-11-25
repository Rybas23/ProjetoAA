from abc import ABC, abstractmethod
from numpy import double
from Project.Observacao import Observacao
from Project.Sensor import Sensor

class IAgente(ABC):
    @abstractmethod
    def agenteCria(self, nome_do_ficheiro_parametros: str) -> None:
        pass

    @abstractmethod
    def observação(self, obs: Observacao) -> None:
        pass

    @abstractmethod
    def accaoAge(self) -> None:
        pass

    @abstractmethod
    def avaliacaoEstadoAtual(self, recompensa: double) -> None:
        pass

    @abstractmethod
    def instala(self, sensor: Sensor) -> None:
        pass

    @abstractmethod
    def comunica(self, mensagem: str, de_agente: "Agente") -> None:
        pass