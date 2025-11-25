from abc import ABC, abstractmethod
from Project.Agente import Agente
from Project.MotorDeSimulacao import MotorDeSimulacao

class ISimulador(ABC):
    @abstractmethod
    def cria(self, nome_do_ficheiro_parametros: str) -> MotorDeSimulacao:
        pass

    @abstractmethod
    def listaAgentes(self) -> list[Agente]:
        pass

    @abstractmethod
    def executa(self) -> None:
        pass