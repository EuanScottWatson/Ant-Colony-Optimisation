import pygame, os
from pygame.locals import *
from random import random, choice
from copy import deepcopy
import numpy as np
from datetime import date, datetime
import sys
import matplotlib.pyplot as plt


class ACO:
    def __init__(self, w, h, ants=10, points=25, alpha=4, beta=2, diffusion_rate=0.1):
        self.alpha = alpha
        self.beta = beta
        self.diffusion_rate = diffusion_rate

        self.ants = ants
        self.distances = [0 for _ in range(ants)]
        # Generate points number of random points on the map, leaving space for statistics on top
        self.points = [(w * random(), (h - 50) * random() + 50) for _ in range(int(points))]
        self.pheremones = np.zeros((int(points), int(points)))

        # Pick a random starting point for each ant and start their path lists
        self.path = [[choice(self.points)] for _ in range(ants)]
        self.to_visit = [deepcopy(self.points) for _ in range(ants)]
        for a in range(ants):
            self.to_visit[a].remove(self.path[a][0])

        # Data for best seen so far
        self.best_path = None
        self.best_distance = float('inf')
        self.generation = 1
        self.time = datetime.now()

        self.font = pygame.font.SysFont('Comic Sans MS', 20)
        self.show_all = False
        self.show_pheremones = False

        self.stats = []

    def display(self, screen):
        # Draw all the points and routes connecting them
        delta = round((datetime.now() - self.time).total_seconds(), 3)

        gen = self.font.render('Generation {0}'.format(self.generation), False, (0, 0, 0))
        dist = self.font.render('Best Distance: {0}'.format(round(self.best_distance, 2)), False, (0, 0, 0))
        time = self.font.render('Time Elapsed (s): {0}'.format(delta), False, (0, 0, 0))

        screen.blit(gen, (25, 10))
        screen.blit(dist, (200, 10))
        screen.blit(time, (445, 10))

        for p in self.points:
            pygame.draw.circle(screen, (0, 0, 0), (p[0], p[1]), 5, width=2)

        if self.show_all:
            for path in self.path:
                if len(path) >= 2:
                    pygame.draw.lines(screen, (0, 0, 0), False, path, 2)
        
        max_pheremone = np.max(self.pheremones)
        if self.show_pheremones and max_pheremone > 0:
            for x in range(len(self.pheremones)):
                for y in range(len(self.pheremones)):
                    if x == y:
                        continue

                    colour = int(205 * self.pheremones[x][y] / max_pheremone)
                    if colour > 2:
                        pygame.draw.line(screen, (50 + colour, 25, 25), self.points[x], self.points[y], 2)

        if self.best_path and not self.show_pheremones:
            pygame.draw.lines(screen, (14, 17, 79), False, self.best_path, 2)


    def events(self):
        # Handles user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return True
                if event.key == K_a:
                    self.show_all = not self.show_all
                if event.key == K_p:
                    self.show_pheremones = not self.show_pheremones

    def display_screen(self, screen):
        screen.fill((160, 160, 169))

        self.display(screen)

        pygame.display.update()
        pygame.display.flip()

    def run_logic(self):
        # If gen in progress, do 1 step for each ant, otherwise update pheremones
        if not all(self.distances):
            for a in range(self.ants):
                self.traverse(a)
        else:
            self.generate_pheremones()
            self.reset()

    def reset(self):
        # Reset all the ant paths and check to see if a new better solution was found
        if (min(self.distances)) < self.best_distance:
            self.best_distance = min(self.distances)
            self.best_path = self.path[np.argmin(self.distances)]
            print("Generation: {0}:\n\tAnt no. {1} found a path of distance of {2} km"
                .format(self.generation, np.argmin(self.distances), round(min(self.distances), 2))
            )

        self.distances = [0 for _ in range(self.ants)]
        self.path = [[choice(self.points)] for _ in range(self.ants)]
        self.to_visit = [deepcopy(self.points) for _ in range(self.ants)]
        for a in range(self.ants):
            self.to_visit[a].remove(self.path[a][0])
        self.generation += 1

        # Store stats for graphings
        self.stats.append(self.best_distance)

    # Diffuses and updates pheremones for each path
    def generate_pheremones(self):
        for x in range(len(self.pheremones)):
            for y in range(len(self.pheremones)):
                self.pheremones[x][y] *= (1 - self.diffusion_rate)

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
        # Pick which path to take based on a roulette wheel
        dist_weights = list(map(lambda x: (1 / self.distance(x, curr) ** self.alpha), remaining))
        dist_norm = [float(x) / max(dist_weights) for x in dist_weights]

        pheremone_weights = list(map(lambda x: self.pheremones[self.points.index(curr)][self.points.index(x)] ** self.beta, remaining))

        # Weight depends on the pheremone strength and the distance
        weights = [a + b for (a, b) in zip(dist_norm, pheremone_weights)]

        total = sum(weights)
        probabilities = list(map(lambda x: x / total, weights))

        return probabilities

    def distance(self, a, b):
        # Calculate distance between two points, L2-norm
        return np.linalg.norm(np.array(a) / 10 - np.array(b) / 10)

    def traverse(self, ant):
        # If there are still points to visit, determine which to go to next
        if len(self.to_visit[ant]) > 0:
            next_index = np.random.choice(range(len(self.to_visit[ant])), 1, p=self.generate_roulette(self.path[ant][-1], self.to_visit[ant]))[0]
            next = self.to_visit[ant][next_index]
            self.to_visit[ant].remove(next)
            self.path[ant].append(next)

        # If all points found, determine length of path
        if len(self.path[ant]) == len(self.points):
            self.path[ant].append(self.path[ant][0])
            distance = sum(map(lambda x: self.distance(x[0], x[1]), zip(self.path[ant], self.path[ant][1:])))
            self.distances[ant] = distance


def main(ants, points, alpha, beta, diffusion_rate):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("Ant Colony Optimisation")

    os.environ['SDL_VIDEO_CENTERED'] = "True"

    width, height = 1000, 800

    screen = pygame.display.set_mode((width, height))

    done = False
    clock = pygame.time.Clock()
    aco = ACO(width, height, ants=ants, points=points, alpha=alpha, beta=beta, diffusion_rate=diffusion_rate)

    while not done:
        done = aco.events()
        aco.run_logic()
        aco.display_screen(screen)

        clock.tick(60)

    plot(aco.stats)

def plot(distances):
    # Plot the graph of best distance vs generation
    plt.plot(distances)
    plt.xlabel("Generation Number")
    plt.ylabel("Best Distance Seen")
    plt.title("Best Path's Distance per Generation")
    plt.show()


if __name__ == "__main__":
    args = {'ants': 10, 'points': 25, 'alpha': 4, 'beta': 1, 'diffusion_rate': 0.1}

    # Accepts arguments in the form of: <arg_name>=<value>
    for arg in sys.argv[1:]:
        try:
            var, val = arg.split('=')
            if var in args.keys():
                args[var] = float(val)
        except:
            print("Incorrect argument: %s" % arg)

    main(**args)
