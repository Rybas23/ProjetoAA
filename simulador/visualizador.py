import pygame

class Visualizador:
    def __init__(self, width, height, title="Ambiente"):
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = 40
        self.screen = pygame.display.set_mode((width*self.cell_size, height*self.cell_size))
        pygame.display.set_caption(title)
        self.colors = {}  # cores atribuídas aos agentes
        self.default_colors = [
            (255,0,0), (0,255,0), (0,0,255), (255,255,0),
            (255,0,255), (0,255,255), (128,128,128), (255,165,0)
        ]

    # Atribui cores únicas para cada agente
    def assign_colors(self, agents):
        for i, ag_id in enumerate(agents.keys()):
            self.colors[ag_id] = self.default_colors[i % len(self.default_colors)]

    # Desenha o grid com recursos, agentes, ninho ou farol
    def draw_grid(self, resources, agents, nest=None, target=None):
        self.screen.fill((255,255,255))

        # Eventos pygame (necessário para não travar)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Desenhar recursos
        for (x,y), amt in resources.items():
            pygame.draw.rect(
                self.screen, (139,69,19),
                (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size)
            )

        # Desenhar agentes
        for ag_id, (x,y) in agents.items():
            color = self.colors.get(ag_id, (0,0,0))
            pygame.draw.circle(
                self.screen, color,
                (x*self.cell_size + self.cell_size//2, y*self.cell_size + self.cell_size//2),
                self.cell_size//3
            )

        # Desenhar ninho
        if nest:
            nx, ny = nest
            pygame.draw.rect(
                self.screen, (0,128,0),
                (nx*self.cell_size, ny*self.cell_size, self.cell_size, self.cell_size), 2
            )

        # Desenhar farol ou target
        if target:
            tx, ty = target
            pygame.draw.rect(
                self.screen, (255,0,0),
                (tx*self.cell_size, ty*self.cell_size, self.cell_size, self.cell_size), 2
            )

        pygame.display.flip()