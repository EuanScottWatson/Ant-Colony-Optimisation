import pygame, os
from pygame.locals import *
from random import random, choice
from copy import deepcopy
import numpy as np


class ACO:
    def __init__(self, w, h):
        # self.points = [(w * random(), h * random()) for _ in range(10)]
        self.points = [(563.393724661267, 288.75083844227373), (345.92947657945706, 176.83411403693992), (381.3848691179929, 413.1509190241147), (155.71624127162113, 143.99370393070978), (516.3181894639458, 634.4476353193571), (622.9478205975814, 2.4471244449268568), (619.2984836361079, 217.09557804709794), (401.83421393343883, 172.67997267144972), (117.79096238078942, 148.79478942445724), (230.06685631248374, 23.860267697892812)]

        self.path = [choice(self.points)]
        self.to_visit = deepcopy(self.points)
        self.to_visit.remove(self.path[0])

    def display(self, screen):
        for p in self.points:
            pygame.draw.circle(screen, (0, 0, 0), (p[0], p[1]), 5, width=2)
        
        if len(self.path) > 2:
            pygame.draw.lines(screen, (0, 0, 0), False, self.path, 2)

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return True

    def display_screen(self, screen):
        screen.fill((255, 255, 255))

        self.display(screen)

        pygame.display.update()
        pygame.display.flip()

    def run_logic(self):
        self.traverse()

    def generate_roulette(self, curr, remaining):
        weights = list(map(lambda x: 1 / np.linalg.norm(np.array(x) - np.array(curr)), remaining))
        total = sum(weights)
        probabilities = list(map(lambda x: x / total, weights))
        return probabilities


    def traverse(self):
        if len(self.to_visit) > 0:
            next_index = np.random.choice(range(len(self.to_visit)), 1, p=self.generate_roulette(self.path[-1], self.to_visit))[0]
            next = self.to_visit[next_index]
            self.to_visit.remove(next)
            self.path.append(next)
        if len(self.path) == len(self.points):
            self.path.append(self.path[0])


def main():
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Ant Colony Optimisation")

    os.environ['SDL_VIDEO_CENTERED'] = "True"

    width, height = 650, 650

    screen = pygame.display.set_mode((width, height))

    done = False
    clock = pygame.time.Clock()
    aco = ACO(width, height)

    while not done:
        done = aco.events()
        aco.run_logic()
        aco.display_screen(screen)

        clock.tick(2)


if __name__ == "__main__":
    main()
