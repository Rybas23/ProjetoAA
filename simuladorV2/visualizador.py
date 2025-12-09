try:
    import pygame, sys
    PYGAME_AVAILABLE = True
except Exception:
    # Caso o pygame não esteja disponível, a simulação ainda funciona mas sem visualização
    PYGAME_AVAILABLE = False


# Classe Visualizador
# Renderiza ambientes de Foraging e Farol usando pygame (quando disponível)
class Visualizador:
    def __init__(self, grid_width, grid_height, title="Ambiente", fps=5, cell_size=40):
        # Dimensões da grelha
        self.width = grid_width
        self.height = grid_height
        self.cell_size = cell_size

        # Velocidade de renderização
        self.fps = fps

        # Estado de execução
        self.running = True

        # Dicionário de cores por agente
        self.colors = {}

        # Conjunto de cores pré-definidas para agentes
        self.default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 128, 128), (255, 165, 0)
        ]

        # Inicialização do pygame (se disponível)
        if PYGAME_AVAILABLE:
            if not pygame.get_init():
                pygame.init()

            # Criar janela
            janela_largura = self.width * self.cell_size
            janela_altura = self.height * self.cell_size
            self.screen = pygame.display.set_mode((janela_largura, janela_altura))
            pygame.display.set_caption(title)

            # Relógio para controlar FPS
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

    # Atribui uma cor distinta a cada agente
    def assign_colors(self, agents):
        for index, agent_id in enumerate(agents.keys()):
            self.colors[agent_id] = self.default_colors[index % len(self.default_colors)]

    # Processa eventos do pygame (fechar janela, etc.)
    def check_events(self):
        if not PYGAME_AVAILABLE:
            return True  # Sem pygame → nada para tratar

        for event in __import__('pygame').event.get():
            if event.type == __import__('pygame').QUIT:
                self.running = False
                __import__('pygame').quit()

                # Tentar encerrar execução sem crash
                try:
                    sys.exit()
                except SystemExit:
                    pass

        return self.running

    # Função base para desenhar ambiente (recursos, agentes, alvos)
    def draw_grid(self, resources, agents, ninho=None, farol=None):
        if not PYGAME_AVAILABLE:
            return True  # Sem pygame → ignora desenho

        import pygame

        if not self.check_events():
            return False

        # Fundo branco
        self.screen.fill((255, 255, 255))

        # Desenhar recursos (foraging)
        for (x, y), quantidade in resources.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                pygame.draw.rect(
                    self.screen,
                    (139, 69, 19),  # castanho (terra)
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

        # Desenhar ninho (foraging)
        if ninho:
            nx, ny = ninho
            pygame.draw.rect(
                self.screen,
                (0, 128, 0),  # verde escuro
                (nx * self.cell_size, ny * self.cell_size, self.cell_size, self.cell_size),
                2
            )

        # Desenhar farol (ambiente Farol)
        if farol:
            tx, ty = farol
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),  # vermelho
                (tx * self.cell_size, ty * self.cell_size, self.cell_size, self.cell_size),
                2
            )

        # Desenhar agentes (círculos coloridos)
        for agent_id, (x, y) in agents.items():
            color = self.colors.get(agent_id, (0, 0, 0))
            pygame.draw.circle(
                self.screen,
                color,
                (x * self.cell_size + self.cell_size // 2,
                 y * self.cell_size + self.cell_size // 2),
                self.cell_size // 3
            )

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    # Desenhar de acordo com o tipo de ambiente
    def draw(self, ambiente):
        # Ambiente de foraging → tem resources
        if hasattr(ambiente, 'resources'):
            recursos = ambiente.resources
            agentes = ambiente.agent_pos
            ninho = ambiente.ninho

            if agentes and not self.colors:
                self.assign_colors(agentes)

            return self.draw_grid(recursos, agentes, ninho=ninho)

        # Ambiente Farol → tem farol (farol)
        else:
            recursos = {}
            agentes = ambiente.agent_pos
            farol = ambiente.farol

            if agentes and not self.colors:
                self.assign_colors(agentes)

            return self.draw_grid(recursos, agentes, farol=farol)

    # Limpeza final do pygame
    def cleanup(self):
        if PYGAME_AVAILABLE:
            import pygame
            if pygame.get_init():
                pygame.quit()
