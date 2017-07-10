class Plant:
    rep = 'w'

    def __init__(self, position):
        self.position = position
        self.energy = 3

    def step(self):
        self.energy += 0

class Critter:
    rep = 'o'
    vision_radius = 2

    def __init__(self, position):
        self.position = position
        self.energy = 7

    def step(self):
        self.energy -= 1

    def eat(self, plant):
        self.energy += plant.energy
        plant.energy = 0

    def observe(self, world):
        return world.chunk(self.position, self.vision_radius)

    def move(self, move):
        self.position[0] = (self.position[0] + move[0]) % 5
        self.position[1] = (self.position[1] + move[1]) % 5
