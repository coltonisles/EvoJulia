
import random
from config import GAConfig as ga
from config import FractalConfig as fractal

class Genotype:
    def __init__(self, c_real, c_imag, x_offset, y_offset, zoom):
        self.c_real = c_real
        self.c_imag = c_imag
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.zoom = zoom
        

    
def generate_random_genotype():
    c_real = random.uniform(*fractal.c_real_range)
    c_imag = random.uniform(*fractal.c_imag_range)
    x_offset = random.uniform(*fractal.x_offset_range)
    y_offset = random.uniform(*fractal.y_offset_range)
    zoom = random.uniform(*fractal.zoom_range)
    
    return Genotype(c_real, c_imag, x_offset, y_offset, zoom)

#population = []
#for i in range(ga.population_size):
#    population.append(generate_random_genotype())
#print(len(population))
#print(population[0].c_real)
#print(population[0].c_imag)

    
def let_there_be_life():
    population = []
    for i in range(ga.population_size):
        population.append(generate_random_genotype())
    return population
    



