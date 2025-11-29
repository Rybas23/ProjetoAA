try:
    import pygame, sys
    PYGAME_AVAILABLE = True
except Exception:
    PYGAME_AVAILABLE = False

class Visualizador:
    def __init__(self, width, height, title="Ambiente", fps=5, cell_size=40):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.fps = fps
        self.running = True
        self.colors = {}
        self.default_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 165, 0)
        ]
        if PYGAME_AVAILABLE:
            if not pygame.get_init():
                pygame.init()
            self.screen = pygame.display.set_mode((self.width * self.cell_size, self.height * self.cell_size))
            pygame.display.set_caption(title)
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

    def assign_colors(self, agents):
        for i, ag_id in enumerate(agents.keys()):
            self.colors[ag_id] = self.default_colors[i % len(self.default_colors)]

    def check_events(self):
        if not PYGAME_AVAILABLE:
            return True
        for event in __import__('pygame').event.get():
            if event.type == __import__('pygame').QUIT:
                self.running = False
                __import__('pygame').quit()
                try:
                    sys.exit()
                except SystemExit:
                    pass
        return self.running

    def draw_grid(self, resources, agents, nest=None, target=None):
        if not PYGAME_AVAILABLE:
            return True
        import pygame
        if not self.check_events():
            return False
        self.screen.fill((255,255,255))
        for (x,y), amt in resources.items():
            if 0 <= x < self.width and 0 <= y < self.height:
                pygame.draw.rect(self.screen, (139,69,19), (x*self.cell_size, y*self.cell_size, self.cell_size, self.cell_size))
        if nest:
            nx, ny = nest
            pygame.draw.rect(self.screen, (0,128,0), (nx*self.cell_size, ny*self.cell_size, self.cell_size, self.cell_size), 2)
        if target:
            tx, ty = target
            pygame.draw.rect(self.screen, (255,0,0), (tx*self.cell_size, ty*self.cell_size, self.cell_size, self.cell_size), 2)
        for ag_id, (x,y) in agents.items():
            color = self.colors.get(ag_id, (0,0,0))
            pygame.draw.circle(self.screen, color, (x*self.cell_size + self.cell_size//2, y*self.cell_size + self.cell_size//2), self.cell_size//3)
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def draw(self, ambiente):
        if hasattr(ambiente, 'resources'):
            resources = getattr(ambiente, 'resources', {})
            agents = getattr(ambiente, 'agent_pos', {})
            nest = getattr(ambiente, 'nest', None)
            if agents and not self.colors:
                self.assign_colors(agents)
            return self.draw_grid(resources, agents, nest=nest)
        else:
            resources = {}
            agents = getattr(ambiente, 'agent_pos', {})
            target = getattr(ambiente, 'farol', None)
            if agents and not self.colors:
                self.assign_colors(agents)
            return self.draw_grid(resources, agents, target=target)

    def cleanup(self):
        if PYGAME_AVAILABLE:
            import pygame
            if pygame.get_init():
                pygame.quit()
