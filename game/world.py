import random
import copy

from game.life import *

class World:
    """randomly generate and populate the world"""
    def __init__(self, size, plants, critters):
        positions = []
        for pos in random.sample(range(size ** 2), plants + critters):
            positions.append([pos//size, pos%size])

        self.size = size
        self.init_plants = [Plant(positions.pop()) for _ in range(plants)]
        self.init_critters = [Critter(positions.pop()) for _ in range(critters)]
        self.reinit()

    def reinit(self):
        self.plants = copy.deepcopy(self.init_plants)
        self.critters = copy.deepcopy(self.init_critters)
        self.year = 0

    def step(self):
        for plant in self.plants:
            plant.step()

        for critter in self.critters:
            for plant in self.plants:
                if critter.position == plant.position:
                    critter.eat(plant)
            critter.step()

        self.plants[:] = [plant for plant in self.plants if plant.energy > 0]
        self.critters[:] = [critter for critter in self.critters if critter.energy > 0]
        self.year += 1

    def chunk(self, pos, radius):
        dim = radius * 2 + 1
        chunk = [[0 for _ in range(dim)] for _ in range(dim)]
        for p in self.plants:
            x = p.position[0] - pos[0]
            if abs(x) > radius:
                x =  p.position[0] + self.size - pos[0]
            if abs(x) > radius:
                x = p.position[0] - self.size - pos[0]
            x += radius

            y = p.position[1] - pos[1]
            if abs(y) > radius:
                y = p.position[1] + self.size - pos[1]
            if abs(y) > radius:
                y = p.position[1] - self.size - pos[1]
            y += radius

            if x < dim and y < dim and x >= 0 and y >= 0:
                chunk[x][y] = 50

        return chunk

    def __str__(self):
        data = [['.' for i in range(self.size)] for i in range(self.size)]
        for plant in self.plants:
            data[plant.position[0]][plant.position[1]] = plant.rep
        for critter in self.critters:
            data[critter.position[0]][critter.position[1]] = critter.rep

        rep = ''

        for row in data:
            for e in row:
                rep += e + ' '
            rep += '\n'

        return rep
