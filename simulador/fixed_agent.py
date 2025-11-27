from simulador.agente_base import AgenteBase

class FixedAgent(AgenteBase):
    def __init__(self, id, policy=None, modo='test'):
        super().__init__(id, modo)
        self.policy = policy or (lambda obs: 'STAY')

    @classmethod
    def cria(cls, params_file):
        return cls('fixed')

    def observacao(self, obs):
        self.last_obs = obs

    def age(self):
        return self.policy(self.last_obs)

    def avaliacaoEstadoAtual(self, recompensa):
        pass
