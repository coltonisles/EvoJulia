
import random
import config

class Genotype:
    def __init__(self, c_real, c_imag, x_offset, y_offset, zoom):
        self.c_real = c_real
        self.c_imag = c_imag
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.zoom = zoom

def generate_random_genotype():
    c_real = random.uniform(-1.0, 1.0)
    c_imag = random.uniform(-1.0, 1.0)
    x_offset = random.uniform(-0.5, 0.5)
    y_offset = random.uniform(-0.5, 0.5)
    zoom = random.uniform(0.5, 2.5)
    return Genotype(c_real, c_imag, x_offset, y_offset, zoom)

population = []
for i in range(config.POPULATION_SIZE):
    population.append(generate_random_genotype())

#print(len(population))
#print(population[0].c_real)
#print(population[0].c_imag)
