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
        self.agent_pos = {}
        for i in range(self.n_agents):
            self.agent_pos[f'a{i}'] = (random.randint(0, self.size-1), random.randint(0, self.size-1))
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
        x,y = self.agent_pos[agente.id]
        if acao == 'UP': y = max(0, y-1)
        elif acao == 'DOWN': y = min(self.size-1, y+1)
        elif acao == 'LEFT': x = max(0, x-1)
        elif acao == 'RIGHT': x = min(self.size-1, x+1)
        self.agent_pos[agente.id] = (x,y)
        self.step += 1
        reward = 0
        terminated = False
        if (x,y) == self.farol:
            reward = 1.0
            terminated = True
        if self.step >= self.max_steps:
            terminated = True
        return reward, terminated

    def atualizacao(self):
        pass

    def is_episode_done(self):
        return False

    def render(self):
        # pygame events (necessário para não crashar)
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.viewer.draw_grid(
            resources={},                 # farol não tem recursos
            agents=self.agent_pos,        # posições dos agentes
            target=self.farol             # desenha o farol
        )