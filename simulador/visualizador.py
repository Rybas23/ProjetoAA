# Visualizador simples usando pygame
import pygame

CELL = 30

class Visualizador:
    def __init__(self, width, height, title='simulador SMA'):
        pygame.init()
        self.w = width
        self.h = height
        self.screen = pygame.display.set_mode((width*CELL, height*CELL))
        pygame.display.set_caption(title)

    def draw_grid(self, resources, agents, nest=None, target=None):
        self.screen.fill((255,255,255))
        for x in range(self.w):
            for y in range(self.h):
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                pygame.draw.rect(self.screen, (200,200,200), rect, 1)
        for (x,y), val in resources.items():
            rect = pygame.Rect(x*CELL+4, y*CELL+4, CELL-8, CELL-8)
            pygame.draw.rect(self.screen, (255,200,0), rect)
        if nest:
            x,y = nest
            rect = pygame.Rect(x*CELL+6, y*CELL+6, CELL-12, CELL-12)
            pygame.draw.rect(self.screen, (150,150,255), rect)
        if target:
            x,y = target
            pygame.draw.circle(self.screen, (255,0,0), (x*CELL+CELL//2, y*CELL+CELL//2), CELL//3)
        for aid, (x,y) in agents.items():
            rect = pygame.Rect(x*CELL+8, y*CELL+8, CELL-16, CELL-16)
            pygame.draw.rect(self.screen, (0,180,0), rect)
        pygame.display.flip()

    def quit(self):
        pygame.quit()
