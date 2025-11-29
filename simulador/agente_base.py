# agente_base.py
import abc

class AgenteBase(metaclass=abc.ABCMeta):
    def __init__(self, id, modo='test'):
        self.id = id
        self.modo = modo
        self.ambiente = None
        self.sensores = []
        self.verbose = False
        self.ultima_observacao = None

    @classmethod
    def cria(cls, params_file):
        raise NotImplementedError

    def instala_ambiente(self, ambiente):
        self.ambiente = ambiente

    def instala(self, sensor):
        """Instala um sensor - apenas armazena a configuração"""
        self.sensores.append(sensor)
        if self.verbose:
            print(f"🔧 [{self.id}] Instalado: {sensor}")

    def observacao(self, obs):
        """Recebe observação do ambiente"""
        self.ultima_observacao = obs
        if self.verbose:
            print(f"👀 [{self.id}] Observação: {obs}")

    @abc.abstractmethod
    def age(self):
        """Decide ação baseada na última observação"""
        pass

    @abc.abstractmethod
    def avaliacaoEstadoAtual(self, recompensa):
        """Processa recompensa da ação"""
        pass

    def reset(self, ep):
        pass

    def comunica(self, mensagem, de_agente):
        if self.verbose:
            print(f'📨 [{self.id}] Mensagem de {de_agente.id}: {mensagem}')