from abc import ABC, abstractmethod

class IMotorDeSimulacao(ABC):
    @abstractmethod
    def passo(self) -> None:
        pass