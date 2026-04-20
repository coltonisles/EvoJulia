""" 
Run the GA for EACH tile

genome: [fractal_id, crop_cx, crop_cy, crop_scale, brightness_shift]

  - fractal_id:         discrete (for crossover / mutation)
  - crop_cx:            continuous (0-1)
  - crop_cy:            continuous (0-1)
  - crop_scale:         continuous (min - max))
  - brightness_shift:   continuous (-0.4 - 0.4)
"""
from config import GAConfig as ga, MosaicConfig as mosaic, SampleFractalConfig
import numpy as np
import cv2
from config import Config
import config
import random

## =================== ##
## ===== FITNESS ===== ##

def evaluation(genome, samples, tile):
    """ 
    genome == [fractal_id, crop_cx, crop_cy, crop_scale, brightness]
    """
    fractal_id = int(genome[0])   
    crop_cx = float(genome[1])
    crop_cy = float(genome[2])
    crop_scale = float(genome[3])    
    brightness_shift = float(genome[4])
    
    # Construct the tile's image
    fractal_image = samples[fractal_id] 
    
    # Crop the fractal to match tile size
    cropped_fractal = crop_fractal(fractal_image, crop_cx, crop_cy, crop_scale, tile.shape[0], tile.shape[1])

    # brightness shift
    fractal_shifted = cropped_fractal + brightness_shift
    fractal_shifted = np.clip(fractal_shifted, 0.0, 1.0)    # lock in values to valid range (0-1)
    
    
    # Tile Edges
    # must first go from float32[0-1] to uint8[0-255] for Canny == float32[]*255
    edged_tile = cv2.Canny((tile*255).astype(np.uint8), Config.canny_low, Config.canny_high)
    edged_fractal = cv2.Canny((fractal_shifted * 255).astype(np.uint8), Config.canny_low, Config.canny_high)
    
    edged_diff = edged_fractal.astype(float) - edged_tile.astype(float)
    edge_loss = np.mean(edged_diff * edged_diff)
    
    
    # calculate MSE
    diff = fractal_shifted - tile
    mse = np.mean(diff * diff)
    
    return mse + (ga.weight_edge * edge_loss)



## =================== ##
## ===== MUTATION ===== ##

def mutate(genome, num_samples, mutation_rate=ga.mutation_rate, intensity=ga.mutation_intensity):
    """
    Mutate a single genome: [fractal_id, cx, cy, scale, brightness]
    """
    new_genome = genome.copy()
    
    # Mutate fractal_id (discrete)
    if random.random() < mutation_rate:
        new_genome[0] = random.randint(0, num_samples - 1)
    
    # Mutate continuous parameters
    for i in [1, 2, 3, 4]:
        if random.random() < mutation_rate:
            noise = random.gauss(0, intensity)
            new_genome[i] += noise
            
            # Clamp to valid ranges
            if i in [1, 2]:  # cx, cy: 0-1
                new_genome[i] = np.clip(new_genome[i], 0.0, 1.0)
            elif i == 3:  # scale: min_crop_scale to max_crop_scale
                new_genome[i] = np.clip(new_genome[i], mosaic.min_crop_scale, mosaic.max_crop_scale)
            elif i == 4:  # brightness: min_brightness_scale to max_brightness_scale
                new_genome[i] = np.clip(new_genome[i], mosaic.min_brightness_scale, mosaic.max_brightness_scale)
    
    return new_genome

def mutate_mosaic(full_genome, num_samples):
    """
    Mutate a full mosaic genome (list of tile genomes)
    """
    return [mutate(g, num_samples) for g in full_genome]



## ==================== ##
## ===== CROSSOVER ===== ##

def crossover(parent1, parent2, crossover_rate=ga.crossover_rate):
    """
    Crossover two single genomes
    """
    if random.random() < crossover_rate:
        child = []
        for i in range(len(parent1)):
            child.append(random.choice([parent1[i], parent2[i]]))
        return child
    else:
        return random.choice([parent1, parent2]).copy()

def crossover_mosaic(parent1, parent2):
    """
    Crossover two full mosaic genomes
    """
    return [crossover(g1, g2) for g1, g2 in zip(parent1, parent2)]



## ===================== ##
## ===== SELECTION ===== ##

def selection(population, fitness_scores, selection_size=ga.selection_size):
    
    # Pair each genotype with its fitness score
    paired = []
    for i in range(len(population)):
        paired.append([population[i], fitness_scores[i]])
        
    # Must be able to GET the score 
    def get_score(instance):
        return instance[1]      #[0,1] == [pop_id, score]
    
    # Sort by fitness score (lowest to highest, lowest == best)
    paired = sorted(paired, key=get_score)
    
    # SELECTION
    best = []
    for i in range(selection_size):
        best.append(paired[i][0])      # paired = [ [pop_id, score], [pop_id, score], ... ]
        
    return best
    
    
    
def crop_fractal(fractal, cx, cy, scale, tile_h, tile_w):
    H, W = fractal.shape
    
    # Cropped size
    crop_h = int(H * scale)
    crop_w = int(W * scale)
    
    # Center oriantation
    center_x = int(cx * W)
    center_y = int(cy * H)
    
    # Top-left corner
    y0 = center_y - crop_h // 2     # force integer division
    x0 = center_x - crop_w // 2
    
    # Lock in boundaries
    y0 = max(0, min(H - crop_h, y0))
    x0 = max(0, min(W - crop_w, x0))
    
    crop = fractal[y0:(y0+crop_h), x0:(x0+crop_w)]
    
    # NOWWW resize crop to tile size
    cropped = cv2.resize(crop, (tile_w, tile_h))

    return cropped






# entire_genome = [ [tile(0,0) genome], [tile(0,1) genome], [tile(0,2) genome], ... 
#                   [tile(1,0) genome], [tile(1,1) genome], [tile(1,2) genome], ...]

def fitness_full_mosaic(genome, samples, input_image, grid_n=config.MosaicConfig.grid_n, tile_size=config.MosaicConfig.tile_size):
    
    tiles_per_row = grid_n
    tile_size = tile_size
    
    output = np.zeros_like(input_image) #_like() auto forms matching np.shape
    
    iter = 0
    for row in range(tiles_per_row):
        for col in range(tiles_per_row):
            
            params = genome[iter]
            iter += 1
            
            fractal_id, cx, cy, scale, brightness = params
            fractal_image = samples[int(fractal_id)]

            tile = crop_fractal(fractal_image, cx, cy, scale, tile_size, tile_size)
            
            tile = np.clip((tile + brightness), 0.0, 1.0)
            
            y0 = row * tile_size
            x0 = col * tile_size
            
            output[y0:(y0+tile_size), x0:(x0+tile_size)] = tile
            
    # NOW, ready to compare FULL MOSAIC IMAGE against input image
    diff = output - input_image
    mse = np.mean(diff * diff)
    
    return mse 
            