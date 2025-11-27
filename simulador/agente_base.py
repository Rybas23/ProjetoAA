import abc

class AgenteBase(metaclass=abc.ABCMeta):
    def __init__(self, id, modo='test'):
        self.id = id
        self.modo = modo
        self.ambiente = None

    @classmethod
    def cria(cls, params_file):
        raise NotImplementedError

    def instala_ambiente(self, ambiente):
        self.ambiente = ambiente

    @abc.abstractmethod
    def observacao(self, obs):
        pass

    @abc.abstractmethod
    def age(self):
        pass

    @abc.abstractmethod
    def avaliacaoEstadoAtual(self, recompensa):
        pass

    def reset(self, ep):
        pass

    def instala(self, sensor):
        self.sensor = sensor

    def comunica(self, mensagem, de_agente):
        print(f'[{self.id}] received from {de_agente}: {mensagem}')