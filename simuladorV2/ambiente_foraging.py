#  AMBIENTE DE FORAGING (ForagingEnv)
#  - Agentes recolhem recursos espalhados na grelha.
#  - Podem carregar 1 recurso de cada vez.
#  - Ganham recompensa ao entregar recursos no ninho.

import random

class ForagingEnv:
    def __init__(self, width=10, height=10,
                 n_resources=10, nest=(0, 0), max_steps=200):

        self.width = width                   # Largura da grelha
        self.height = height                 # Altura da grelha
        self.n_resources = n_resources       # Nº inicial de recursos
        self.nest = nest                     # Posição do ninho
        self.max_steps = max_steps

        self.step = 0
        self.agent_ids = []
        self.agent_pos = {}                 # Posição dos agentes
        self.carrying = {}                  # 1 ou 0 (se carrega recurso)
        self.resources = {}                 # (x,y) → quantidade no tile

    # Registar agentes
    def registar_agentes(self, agentes):
        self.agent_ids = [ag.id for ag in agentes]

    # Reiniciar episódio
    def reset(self):
        self.step = 0
        self.resources = {}

        # Espalhar recursos aleatoriamente
        for _ in range(self.n_resources):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            self.resources[(x, y)] = self.resources.get((x, y), 0) + 1

        # Colocar agentes no ninho
        for aid in self.agent_ids:
            self.agent_pos[aid] = self.nest
            self.carrying[aid] = 0

        self.total_delivered = 0  # Contagem global de entregas

        return self._state()

    # Estado global do ambiente
    def _state(self):
        return {
            'resources': dict(self.resources),
            'agents': dict(self.agent_pos),
            'nest': self.nest
        }

    # Observação para agente
    def observacaoPara(self, agente):
        x, y = self.agent_pos[agente.id]

        # Visão local em 4 direções + posição atual
        vis = {}
        direcoes = [
            (-1, 0, 'L'),
            (1, 0, 'R'),
            (0, -1, 'U'),
            (0, 1, 'D')
        ]

        for dx, dy, k in direcoes:
            nx, ny = x + dx, y + dy

            if 0 <= nx < self.width and 0 <= ny < self.height:
                vis[k] = self.resources.get((nx, ny), 0)
            else:
                vis[k] = -1  # parede

        vis['C'] = self.resources.get((x, y), 0)  # recurso no tile atual

        return {
            'pos': (x, y),
            'visao': vis,
            'carrying': self.carrying[agente.id],
            'nest': self.nest
        }

    # Executar ação do agente
    def agir(self, acao, agente):
        x, y = self.agent_pos[agente.id]
        recompensa = 0.0
        terminou = False

        # Movimentos
        if acao == 'UP':
            y = max(0, y - 1)
        elif acao == 'DOWN':
            y = min(self.height - 1, y + 1)
        elif acao == 'LEFT':
            x = max(0, x - 1)
        elif acao == 'RIGHT':
            x = min(self.width - 1, x + 1)

        # Recolher recurso
        elif acao == 'PICK':
            if self.resources.get((x, y), 0) > 0 and self.carrying[agente.id] == 0:
                self.resources[(x, y)] -= 1
                if self.resources[(x, y)] == 0:
                    del self.resources[(x, y)]
                self.carrying[agente.id] = 1
                recompensa = 0.5

        # Largar recurso no ninho
        elif acao == 'DROP':
            if (x, y) == self.nest and self.carrying[agente.id] == 1:
                self.carrying[agente.id] = 0
                self.total_delivered += 1
                recompensa = 1.0

        # Atualizar posição
        self.agent_pos[agente.id] = (x, y)

        # Custo de movimento
        if acao in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']:
            recompensa = -0.01

        return recompensa, terminou

    # Atualizar contador global
    def atualizacao(self):
        self.step += 1

    # Fim do episódio
    def is_episode_done(self):
        return (
            self.step >= self.max_steps or
            len(self.resources) == 0
        )
