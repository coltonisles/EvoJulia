
import random
import config

#Defines a singular genotype used to generate one fractal
class Genotype:
    def __init__(self, layers):
        #list of dictionaries, each containing a layer's genotype'
       self.layers = layers

#Generates a random genotype
def generate_random_genotype():
    layers = []

    #generate random values for each layer; '_' is used when we do not care about the iteration number
    for _ in range(config.NUM_LAYERS):
        #dictionary for a single layer; random.uniform(x, y) generates a random number between the given range x, y
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
#loops through the population size and generates a random genotype for each
for i in range(config.POPULATION_SIZE):
    population.append(generate_random_genotype())

#print(len(population))
#print(population[0].c_real)
#print(population[0].c_imag)
