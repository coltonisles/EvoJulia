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



# == INIT FRACTAL_ID ACCORDING TO TILE BRIGHTNESS (give it a starting advantage) == #
def generate_appropriate_individual(fractal_brightnesses, tile_brightness):
    num_fractal_samples = len(fractal_brightnesses)
    
    # Find the fractal with the closest brightness to THIS tile's brightness
    best_match_index = 0
    best_diff = abs(fractal_brightnesses[0] - tile_brightness)
    for i in range(1, num_fractal_samples):
        diff = abs(fractal_brightnesses[i] - tile_brightness)
        if diff < best_diff:
            best_diff = diff
            best_match_index = i
    
    # Choose a fractal within 2 of the best match (SOME variety )
    low = max(0, best_match_index - 2)
    high = min(num_fractal_samples - 1, best_match_index + 2)
    fractal_id = random.randint(low, high)
    
    # Other genes still fully random
    crop_cx = random.uniform(0.0, 1.0)
    crop_cy = random.uniform(0.0, 1.0)
    crop_scale = random.uniform(mosaic.min_crop_scale, mosaic.max_crop_scale)
    brightness_shift = random.uniform(mosaic.min_brightness_scale, mosaic.max_brightness_scale)
    
    return MosaicGenome(fractal_id, crop_cx, crop_cy, crop_scale, brightness_shift)


# let_there_be_life(), but with the tile-fractal brightness matching
def let_there_be_life_with_help(fractal_brightnesses, tile_brightness):
    population = []
    for i in range(ga.population_size):
        population.append(generate_appropriate_individual(fractal_brightnesses, tile_brightness))
    return population