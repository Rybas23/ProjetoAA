from abc import ABC, abstractmethod
from Project import Observacao, Agente, Accao

class IAmbiente(ABC):
    @abstractmethod
    def observacaoPara(self, agente: Agente) -> Observacao:
        pass

    @abstractmethod
    def atualizacao(self) -> None:
        pass

    @abstractmethod
    def agir(self, accao: Accao, agente: Agente) -> None:
        pass