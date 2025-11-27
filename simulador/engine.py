import time
import json
from collections import defaultdict

class MotorDeSimulacao:
    def __init__(self, params):
        self.params = params
        self.ambiente = None
        self.agentes = []
        self.max_steps = params.get('max_steps', 500)
        self.metrics = defaultdict(list)

    @classmethod
    def cria(cls, params_file: str):
        with open(params_file, 'r') as f:
            params = json.load(f)
        return cls(params)

    def adiciona_ambiente(self, ambiente):
        self.ambiente = ambiente

    def adiciona_agente(self, agente):
        self.agentes.append(agente)
        agente.instala_ambiente(self.ambiente)

    def listaAgentes(self):
        return list(self.agentes)

    def executa(self, render=False):
        if self.ambiente is None:
            raise RuntimeError('Ambiente não definido')

        num_episodes = self.params.get('episodes', 10)
        for ep in range(num_episodes):
            state = self.ambiente.reset()
            for ag in self.agentes:
                ag.reset(ep)
            step = 0
            ep_reward = {ag.id: 0 for ag in self.agentes}
            done = False
            while step < self.max_steps and not done:
                for ag in self.agentes:
                    obs = self.ambiente.observacaoPara(ag)
                    ag.observacao(obs)
                acts = [(ag, ag.age()) for ag in self.agentes]
                for ag, act in acts:
                    reward, terminated = self.ambiente.agir(act, ag)
                    ag.avaliacaoEstadoAtual(reward)
                    ep_reward[ag.id] += reward
                done = self.ambiente.is_episode_done()
                self.ambiente.atualizacao()
                step += 1
                if render and hasattr(self.ambiente, 'render'):
                    self.ambiente.render()
                    time.sleep(self.params.get('render_delay', 0.05))

            for ag in self.agentes:
                self.metrics['reward_'+ag.id].append(ep_reward[ag.id])
                self.metrics['steps'].append(step)

        return self.metrics
