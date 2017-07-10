import heapq
import time

import numpy as np

from game.world import World
import net.net as netutils

def print_run(net, world):
    c = world.critters[0]
    while c.energy > 0:
        chunk = np.array(c.observe(world)).flatten()
        o = net.feed_forward(chunk)

        print(world)
        print(normalize(o))

        movement = [0] * 4
        movement[np.argmax(o)] = 1
        c.move([movement[0] - movement[1], movement[2] - movement[3]])
        world.step()
        time.sleep(0.3)
    print(world.year)
    world.reinit()

def normalize(elems):
    normalized = []
    s = sum(elems)
    for e in elems:
        normalized.append(e/s)
    return normalized

def weighted_random(elems):
    rand = np.random.uniform(0, sum(elems))
    for i in range(len(elems)):
        if rand <= elems[i]:
            return i
        rand -= elems[i]

if __name__ == '__main__':
    init_size = 20
    gen_size = 5
    n_generations = 20

    shape = [25, 14, 4]
    afuncs = [netutils.sigmoid] * 2
    dafuncs = [netutils.dsigmoid] * 2
    mu = 0.1

    w_size = 5
    n_plants = 5
    n_critters = 1

    train_fitness = []
    train_data = []
    train_values = []

    world = World(w_size, n_plants, n_critters)
    for generation in range(n_generations):
        nets = []
        size = gen_size if generation > 0 else init_size
        for _ in range(size):
            nets.append(netutils.Net(shape, mu, afuncs, dafuncs))

        net_data = []
        for net in nets:
            inp = []
            outp = []
            fitness = 0

            if len(train_data) != 0:
                for _ in range(3):
                    net.train(train_data, train_values)

            c = world.critters[0]
            while c.energy > 0:
                chunk = np.array(c.observe(world)).flatten()
                movement = [0] * 4
                o = net.feed_forward(chunk)
                movement[weighted_random(o)] = 1

                c.move([movement[0] - movement[1], movement[2] - movement[3]])
                world.step()

                inp.append(chunk)
                outp.append(movement)

            fitness += world.year
            world.reinit()
            net_data.append((inp, outp, fitness))

        best = heapq.nlargest(2, net_data, key=lambda x: x[2])
        print('Generation ' + str(generation) + ' Fitness: ' + str(best[0][2]) + ', ' + str(best[1][2]))
        for i in reversed(best):
            fitness = i[2]
            index = 0
            for j in reversed(range(len(train_fitness))):
                if fitness > train_fitness[j]:
                    index = j
                    break
            train_fitness = train_fitness[:index] + [fitness]*len(i[0]) + train_fitness[index:]
            train_data = train_data[:index] + i[0] + train_data[index:]
            train_values = train_values[:index] + i[1] + train_values[index:]
            if len(train_data) > 8000:
                train_fitness = train_fitness[len(train_data)-8000:]
                train_data = train_data[len(train_data)-8000:]
                train_values = train_values[len(train_values)-8000:]

    # train a final net to be observed in action
    net = netutils.Net(shape, mu, afuncs, dafuncs)
    for _ in range(5):
        net.train(train_data, train_values)
    print_run(net, world)
