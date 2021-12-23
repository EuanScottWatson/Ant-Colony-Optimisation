import pygame, os
from pygame.locals import *
from random import random, choice
from copy import deepcopy
import numpy as np


class ACO:
    def __init__(self, w, h, ants=1, points=25):
        self.ants = ants
        self.distances = [0 for _ in range(ants)]
        self.points = [(w * random(), h * random()) for _ in range(points)]
        self.pheremones = [[0 for _ in range(points)] for _ in range(points)]
        print(self.points)

        # self.points = [(563.393724661267, 288.75083844227373), (345.92947657945706, 176.83411403693992), (381.3848691179929, 413.1509190241147), (155.71624127162113, 143.99370393070978), (516.3181894639458, 634.4476353193571), (622.9478205975814, 2.4471244449268568), (619.2984836361079, 217.09557804709794), (401.83421393343883, 172.67997267144972), (117.79096238078942, 148.79478942445724), (230.06685631248374, 23.860267697892812)]

        self.path = [[choice(self.points)] for _ in range(ants)]
        self.to_visit = [deepcopy(self.points) for _ in range(ants)]
        for a in range(ants):
            self.to_visit[a].remove(self.path[a][0])

    def display(self, screen):
        for p in self.points:
            pygame.draw.circle(screen, (0, 0, 0), (p[0], p[1]), 5, width=2)
        
        for i, path in enumerate(self.path):
            if len(path) > 2 and (i == np.argmin(self.distances) or not all(self.distances)):
                pygame.draw.lines(screen, (0, 0, 0), False, path, 2)

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
        if not all(self.distances):
            for a in range(self.ants):
                self.traverse(a)
        else:
            print("Ant no. {0} found a distance of {1} km".format(np.argmin(self.distances), round(min(self.distances), 2)))
            self.generate_pheremones()

    def generate_pheremones(self):
        pass

    def generate_roulette(self, curr, remaining):
        dist_weights = list(map(lambda x: 1 / self.distance(x, curr), remaining))
        dist_norm = [float(x) / max(dist_weights) for x in dist_weights]

        pheremone_weights = list(map(lambda x: self.pheremones[self.points.index(curr)][self.points.index(x)], remaining))

        weights = [a + b for (a, b) in zip(dist_norm, pheremone_weights)]

        total = sum(weights)
        probabilities = list(map(lambda x: x / total, weights))

        return probabilities

    def distance(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def traverse(self, ant):
        if len(self.to_visit[ant]) > 0:
            next_index = np.random.choice(range(len(self.to_visit[ant])), 1, p=self.generate_roulette(self.path[ant][-1], self.to_visit[ant]))[0]
            next = self.to_visit[ant][next_index]
            self.to_visit[ant].remove(next)
            self.path[ant].append(next)
        if len(self.path[ant]) == len(self.points):
            self.path[ant].append(self.path[ant][0])
            distance = sum(map(lambda x: self.distance(x[0], x[1]), zip(self.path[ant], self.path[ant][1:])))
            self.distances[ant] = distance


def main():
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Ant Colony Optimisation")

    os.environ['SDL_VIDEO_CENTERED'] = "True"

    width, height = 650, 650

    screen = pygame.display.set_mode((width, height))

    done = False
    clock = pygame.time.Clock()
    aco = ACO(width, height, ants=15, points=5)

    while not done:
        done = aco.events()
        aco.run_logic()
        aco.display_screen(screen)

        clock.tick(5)


if __name__ == "__main__":
    main()
