import random
from simulador.ambiente_base import AmbienteBase
from simulador.visualizador import Visualizador

class ForagingEnv(AmbienteBase):
    def __init__(self, width=10, height=10, n_agents=2, n_resources=10, nest=(0,0), max_steps=200):
        self.w = width
        self.h = height
        self.n_agents = n_agents
        self.n_resources = n_resources
        self.nest = nest
        self.max_steps = max_steps
        self.step = 0
        self.viewer = Visualizador(width, height, title="Foraging")

    def reset(self):
        self.step = 0
        self.resources = {}
        for _ in range(self.n_resources):
            x = random.randint(0, self.w-1)
            y = random.randint(0, self.h-1)
            self.resources[(x,y)] = self.resources.get((x,y), 0) + 1
        self.agent_pos = {f'a{i}': self.nest for i in range(self.n_agents)}
        self.carrying = {f'a{i}': 0 for i in range(self.n_agents)}
        self.total_delivered = 0
        return self._state()

    def _state(self):
        return {'resources': dict(self.resources), 'agents': dict(self.agent_pos), 'nest': self.nest}

    def observacaoPara(self, agente):
        pos = self.agent_pos[agente.id]
        x, y = pos
        neighbours = {}
        for dx, dy, name in [(-1, 0, 'L'), (1, 0, 'R'), (0, -1, 'U'), (0, 1, 'D')]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                neighbours[name] = self.resources.get((nx, ny), 0)
            else:
                neighbours[name] = -1
        # recurso na própria célula
        neighbours['C'] = self.resources.get((x, y), 0)
        return {'pos': pos, 'neigh': neighbours, 'carrying': self.carrying[agente.id], 'nest': self.nest}

    def agir(self, acao, agente):
        x,y = self.agent_pos[agente.id]
        reward = 0.0
        terminated = False

        if acao == 'UP': y = max(0, y-1)
        elif acao == 'DOWN': y = min(self.h-1, y+1)
        elif acao == 'LEFT': x = max(0, x-1)
        elif acao == 'RIGHT': x = min(self.w-1, x+1)
        elif acao == 'PICK':
            if self.resources.get((x,y),0) > 0 and self.carrying[agente.id]==0:
                self.resources[(x,y)] -= 1
                if self.resources[(x,y)] == 0:
                    del self.resources[(x,y)]
                self.carrying[agente.id] = 1
                reward = 0.5
        elif acao == 'DROP':
            if (x,y) == self.nest and self.carrying[agente.id]==1:
                self.carrying[agente.id] = 0
                self.total_delivered += 1
                reward = 1.0

        self.agent_pos[agente.id] = (x,y)
        return reward, terminated

    def atualizacao(self):
        self.step += 1

    def is_episode_done(self):
        # termina quando max_steps atingido
        return self.step >= self.max_steps

    def render(self):
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.viewer.assign_colors(self.agent_pos)
        self.viewer.draw_grid(
            resources=self.resources,
            agents=self.agent_pos,
            nest=self.nest
        )