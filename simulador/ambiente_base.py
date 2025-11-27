from abc import ABC, abstractmethod

class AmbienteBase(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def observacaoPara(self, agente):
        pass

    @abstractmethod
    def atualizacao(self):
        pass

    @abstractmethod
    def agir(self, acao, agente):
        """Return (reward, terminated_bool)"""
        pass

    @abstractmethod
    def is_episode_done(self):
        pass