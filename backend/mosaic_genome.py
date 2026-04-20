""" 
genome: [fractal_id, crop_cx, crop_cy, crop_scale, brightness]

  - fractal_id:         (index from sample fractals)
  - crop_cx:            (0-1)
  - crop_cy:            (0-1)
  - crop_scale:         (min - max)
  - brightness_shift:   (-0.5 - 0.5)
"""


import random
from config import SampleFractalConfig as samples
import image_preprocessor as grid
from config import FractalConfig as fractal, MosaicConfig as mosaic, GAConfig as ga

class MosaicGenome:
    def __init__(self, fractal_id, crop_cx, crop_cy, crop_scale, brightness_shift):
        self.fractal_id = fractal_id
        self.crop_cx = crop_cx
        self.crop_cy = crop_cy
        self.crop_scale = crop_scale
        self.brightness_shift = brightness_shift

# Create a [List of those^ genes]
def get_genome(genome):
    return [
        genome.fractal_id,
        genome.crop_cx,
        genome.crop_cy,
        genome.crop_scale,
        genome.brightness_shift
    ]
    
def generate_random_individual(num_fractal_samples=len(samples.JULIA_SETS)):
    fractal_id = random.randint(0,num_fractal_samples - 1)
    crop_cx = random.uniform(0.0,1.0)
    crop_cy = random.uniform(0.0,1.0)
    crop_scale = random.uniform(mosaic.min_crop_scale, mosaic.max_crop_scale)
    brightness_shift = random.uniform(mosaic.min_brightness_scale, mosaic.max_brightness_scale)
    
    return MosaicGenome(fractal_id, crop_cx, crop_cy, crop_scale, brightness_shift)



#population = []
#for i in range(ga.population_size):
#    population.append(generate_random_genotype())
#print(len(population))
#print(population[0].c_real)
#print(population[0].c_imag)

    
def let_there_be_life():
    population = []
    for i in range(ga.population_size):
        population.append(generate_random_individual())
    return population
    
#def populate_all_tiles(img):
#    all_tiles = grid.get_all_tiles(img)
    
#    for iter, tile_data in enumerate(all_tiles):
#        tile_population = let_there_be_life()