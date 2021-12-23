import pygame, os
from pygame.locals import *
from random import random, choice
from copy import deepcopy
import numpy as np


class ACO:
    def __init__(self, w, h, ants=1, points=25):
        self.alpha = 4
        self.beta = 2

        self.ants = ants
        self.distances = [0 for _ in range(ants)]
        self.points = [(w * random(), h * random()) for _ in range(points)]
        # self.points = [(568.7634782891314, 364.6439100211969), (422.5507894165482, 322.08125997417864), (479.1514906752393, 414.6436176612771), (600.2284341277754, 213.75552764739527), (469.08730651810333, 506.309430537065), (552.0324098368213, 298.6652695991528), (410.7647007057513, 6.157219002521025), (138.83212959054836, 89.38420309002717), (67.86017862377804, 244.71224029775425), (605.5312359070574, 233.1545470297325)]
        self.pheremones = np.zeros((points, points))

        self.path = [[choice(self.points)] for _ in range(ants)]
        self.to_visit = [deepcopy(self.points) for _ in range(ants)]
        for a in range(ants):
            self.to_visit[a].remove(self.path[a][0])

        self.best_path = None
        self.best_distance = float('inf')
        self.generation = 1

    def display(self, screen):
        for p in self.points:
            pygame.draw.circle(screen, (0, 0, 0), (p[0], p[1]), 5, width=2)
        
        # for i, path in enumerate(self.path):
        #     if len(path) > 2 and (i == np.argmin(self.distances) or not all(self.distances)):
        #         pygame.draw.lines(screen, (0, 0, 0), False, path, 2)

        if self.best_path:
            pygame.draw.lines(screen, (0, 0, 0), False, self.best_path, 2)


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
            self.generate_pheremones()
            self.reset()

    def reset(self):
        if (d := np.argmin(self.distances)) < self.best_distance:
            self.best_distance = d
            self.best_path = self.path[d]
            print("Generation: {0}:\n\tAnt no. {1} found a path of distance of {2} km"
                .format(self.generation, np.argmin(self.distances), round(min(self.distances), 2))
            )

        self.distances = [0 for _ in range(self.ants)]
        self.path = [[choice(self.points)] for _ in range(self.ants)]
        self.to_visit = [deepcopy(self.points) for _ in range(self.ants)]
        for a in range(self.ants):
            self.to_visit[a].remove(self.path[a][0])
        self.generation += 1

    def generate_pheremones(self):
        sorted_dists = sorted(self.distances, reverse=True)
        for ant in range(self.ants):
            path = zip(self.path[ant], self.path[ant][1:])
            for (a, b) in path:
                a_index = self.points.index(a)
                b_index = self.points.index(b)
                p = sorted_dists.index(self.distances[ant]) / self.ants
                self.pheremones[a_index][b_index] += p
                self.pheremones[b_index][a_index] += p

    def generate_roulette(self, curr, remaining):
        dist_weights = list(map(lambda x: (1 / self.distance(x, curr) ** self.alpha), remaining))
        dist_norm = [float(x) / max(dist_weights) for x in dist_weights]

        pheremone_weights = list(map(lambda x: self.pheremones[self.points.index(curr)][self.points.index(x)] ** self.beta, remaining))

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
    aco = ACO(width, height, ants=50, points=25)

    while not done:
        done = aco.events()
        aco.run_logic()
        aco.display_screen(screen)

        clock.tick(60)


if __name__ == "__main__":
    main()
