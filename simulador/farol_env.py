# farol_env.py
import random
from simulador.ambiente_base import AmbienteBase
from simulador.visualizador import Visualizador
from simulador.sensor import TipoDirecao


class FarolEnv(AmbienteBase):
    def __init__(self, size=10, max_steps=200, farol_fixo=None):
        self.size = size
        self.max_steps = max_steps
        self.step = 0
        self.viewer = Visualizador(size, size, title="Farol", fps=5)
        self.farol_fixo = farol_fixo or (size // 2, size // 2)
        self.farol = self.farol_fixo
        self.agent_ids = []
        self.done_agents = set()  # ⬅️ INICIALIZAR AQUI TAMBÉM

    def registar_agentes(self, agentes):
        """Regista os IDs dos agentes no ambiente"""
        self.agent_ids = [ag.id for ag in agentes]

    def reset(self):
        self.step = 0
        self.farol = self.farol_fixo
        self.done_agents = set()  # ⬅️ RESETAR A CADA EPISÓDIO

        # Agentes em posições aleatórias (mas não em cima do farol)
        self.agent_pos = {}
        for agent_id in self.agent_ids:
            while True:
                pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
                if pos != self.farol:
                    self.agent_pos[agent_id] = pos
                    break

        print(f"🎯 Farol FIXO na posição: {self.farol}")
        print(f"🤖 Agentes registados: {list(self.agent_pos.keys())}")
        return self._state()

    def _state(self):
        return {'farol': self.farol, 'agent3s': dict(self.agent_pos)}

    def observacaoPara(self, agente):
        """Retorna observação baseada nos sensores do agente"""
        if agente.id not in self.agent_pos:
            return {}

        pos_agente = self.agent_pos[agente.id]
        observacao = {}

        # Verificar cada sensor do agente
        for sensor in agente.sensores:
            if sensor.tipo == 'farol':
                # Sensor Farol: retorna direção N/S/E/O
                direcao = self._calcular_direcao_farol(pos_agente)
                observacao['direcao_farol'] = direcao.value

            elif sensor.tipo == 'visao':
                # Sensor Visão: retorna o que vê em cada direção
                visao = self._calcular_visao(pos_agente, sensor.alcance)
                observacao['visao'] = visao

        # Sempre incluir posição atual
        observacao['posicao'] = pos_agente

        return observacao

    def _calcular_direcao_farol(self, pos_agente):
        """Calcula direção do farol (N/S/E/O)"""
        x_a, y_a = pos_agente
        x_f, y_f = self.farol

        if x_f > x_a:
            return TipoDirecao.ESTE
        elif x_f < x_a:
            return TipoDirecao.OESTE
        elif y_f > y_a:
            return TipoDirecao.SUL
        elif y_f < y_a:
            return TipoDirecao.NORTE
        else:
            return TipoDirecao.NENHUMA  # Está em cima do farol

    def _calcular_visao(self, pos_agente, alcance):
        """Retorna o que o agente vê em cada direção"""
        x, y = pos_agente
        visao = {
            'N': self._conteudo_celula(x, y - 1),
            'S': self._conteudo_celula(x, y + 1),
            'E': self._conteudo_celula(x + 1, y),
            'W': self._conteudo_celula(x - 1, y)
        }

        # Para alcance > 1, adicionar diagonais
        if alcance > 1:
            visao.update({
                'NE': self._conteudo_celula(x + 1, y - 1),
                'NW': self._conteudo_celula(x - 1, y - 1),
                'SE': self._conteudo_celula(x + 1, y + 1),
                'SW': self._conteudo_celula(x - 1, y + 1)
            })

        return visao

    def _conteudo_celula(self, x, y):
        """Determina o conteúdo de uma célula"""
        # Verificar limites
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 'PAREDE'

        # Verificar farol
        if (x, y) == self.farol:
            return 'FAROL'

        # Verificar outros agentes
        for ag_id, pos in self.agent_pos.items():
            if pos == (x, y):
                return f'AGENTE_{ag_id}'

        return 'VAZIO'

    def agir(self, acao, agente):
        if agente.id in self.done_agents:
            return 0.0, True  # Agente já terminou

        x, y = self.agent_pos[agente.id]
        reward = -0.01  # Custo por passo
        terminated = False

        # Executar ação
        if acao == 'UP' and y > 0:
            y -= 1
        elif acao == 'DOWN' and y < self.size - 1:
            y += 1
        elif acao == 'LEFT' and x > 0:
            x -= 1
        elif acao == 'RIGHT' and x < self.size - 1:
            x += 1
        elif acao == 'STAY':
            pass  # Fica no mesmo lugar
        else:
            # Ação inválida
            reward = -0.1

        # Atualizar posição
        self.agent_pos[agente.id] = (x, y)

        # Verificar se alcançou o farol
        if (x, y) == self.farol:
            reward = 10.0  # Grande recompensa
            self.done_agents.add(agente.id)
            terminated = True
            if agente.verbose:
                print(f"🎉 [{agente.id}] ALCANÇOU O FAROL! +10.0 pontos")

        return reward, terminated

    def atualizacao(self):
        self.step += 1

    def is_episode_done(self):
        # ⬇️ CORREÇÃO: Usar len(self.agent_ids) em vez de self.n_agents
        return self.step >= self.max_steps or len(self.done_agents) == len(self.agent_ids)

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