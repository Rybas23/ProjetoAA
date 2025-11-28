import random
from simulador.ambiente_base import AmbienteBase
from simulador.visualizador import Visualizador

class FarolEnv(AmbienteBase):
    def __init__(self, size=10, n_agents=1, max_steps=200):
        self.size = size
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.step = 0
        self.viewer = Visualizador(size, size, title="Farol")

    def reset(self):
        self.step = 0
        self.farol = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.agent_pos = {f'a{i}': (random.randint(0, self.size-1), random.randint(0, self.size-1))
                          for i in range(self.n_agents)}
        self.done_agents = set()  # mantém agentes que chegaram ao farol
        return self._state()

    def _state(self):
        return {'farol': self.farol, 'agents': dict(self.agent_pos)}

    def observacaoPara(self, agente):
        pos = self.agent_pos[agente.id]
        dx = self.farol[0] - pos[0]
        dy = self.farol[1] - pos[1]
        sdx = 0 if dx==0 else (1 if dx>0 else -1)
        sdy = 0 if dy==0 else (1 if dy>0 else -1)
        return {'dir': (sdx, sdy), 'pos': pos}

    def agir(self, acao, agente):
        if agente.id in self.done_agents:
            return 0.0, True  # agente já terminou

        x,y = self.agent_pos[agente.id]
        reward = 0.0

        if acao == 'UP': y = max(0, y-1)
        elif acao == 'DOWN': y = min(self.size-1, y+1)
        elif acao == 'LEFT': x = max(0, x-1)
        elif acao == 'RIGHT': x = min(self.size-1, x+1)
        # 'STAY' ou inválida: não move

        self.agent_pos[agente.id] = (x,y)

        terminated = False
        if (x,y) == self.farol:
            reward = 1.0
            self.done_agents.add(agente.id)
            terminated = True

        return reward, terminated

    def atualizacao(self):
        self.step += 1

    def is_episode_done(self):
        # termina se todos os agentes chegaram ou max_steps
        return self.step >= self.max_steps or len(self.done_agents) == self.n_agents

    def render(self):
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.viewer.assign_colors(self.agent_pos)
        self.viewer.draw_grid(
            resources={},
            agents=self.agent_pos,
            target=self.farol
        )