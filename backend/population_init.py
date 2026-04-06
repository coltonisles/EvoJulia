
import random
import config

class Genotype:
    def __init__(self, layers):
       self.layers = layers

def generate_random_genotype():
    layers = []
    for _ in range(config.NUM_LAYERS):
        layer = {
            'c_real': random.uniform(-1.0, 1.0),
            'c_imag': random.uniform(-1.0, 1.0),
            'x_offset': random.uniform(-0.5, 0.5),
            'y_offset': random.uniform(-0.5, 0.5),
            'zoom': random.uniform(0.5, 2.5)
        }
        layers.append(layer)
    return Genotype(layers)

population = []
for i in range(config.POPULATION_SIZE):
    population.append(generate_random_genotype())

#print(len(population))
#print(population[0].c_real)
#print(population[0].c_imag)
