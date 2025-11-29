import pygame
import sys


class Visualizador:
    def __init__(self, width, height, title="Ambiente", fps=5):
        # Inicializar pygame apenas uma vez
        if not pygame.get_init():
            pygame.init()

        self.width = width
        self.height = height
        self.cell_size = 40
        self.screen = pygame.display.set_mode((width * self.cell_size, height * self.cell_size))
        pygame.display.set_caption(title)
        self.colors = {}  # cores atribuídas aos agentes
        self.default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 165, 0)
        ]
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.running = True

    # Atribui cores únicas para cada agente
    def assign_colors(self, agents):
        for i, ag_id in enumerate(agents.keys()):
            self.colors[ag_id] = self.default_colors[i % len(self.default_colors)]

    # Verifica eventos e se a janela foi fechada
    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
        return self.running

    # Desenha o grid com recursos, agentes, ninho ou farol
    def draw_grid(self, resources, agents, nest=None, target=None):
        # Verificar se ainda está running
        if not self.check_events():
            return False

        # Limpar ecrã
        self.screen.fill((255, 255, 255))

        # Desenhar recursos
        for (x, y), amt in resources.items():
            # Verificar limites
            if 0 <= x < self.width and 0 <= y < self.height:
                pygame.draw.rect(
                    self.screen, (139, 69, 19),
                    (x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                )

        # Desenhar agentes
        for ag_id, (x, y) in agents.items():
            # Verificar limites
            if 0 <= x < self.width and 0 <= y < self.height:
                color = self.colors.get(ag_id, (0, 0, 0))
                pygame.draw.circle(
                    self.screen, color,
                    (x * self.cell_size + self.cell_size // 2,
                     y * self.cell_size + self.cell_size // 2),
                    self.cell_size // 3
                )

        # Desenhar ninho
        if nest:
            nx, ny = nest
            if 0 <= nx < self.width and 0 <= ny < self.height:
                pygame.draw.rect(
                    self.screen, (0, 128, 0),
                    (nx * self.cell_size, ny * self.cell_size, self.cell_size, self.cell_size), 2
                )

        # Desenhar farol ou target
        if target:
            tx, ty = target
            if 0 <= tx < self.width and 0 <= ty < self.height:
                pygame.draw.rect(
                    self.screen, (255, 0, 0),
                    (tx * self.cell_size, ty * self.cell_size, self.cell_size, self.cell_size), 2
                )

        # Atualizar display
        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def cleanup(self):
        """Limpeza segura do Pygame"""
        if pygame.get_init():
            pygame.quit()