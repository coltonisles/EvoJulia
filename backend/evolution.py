""" 
Run the GA for EACH tile

genome: [fractal_id, crop_cx, crop_cy, crop_scale, brightness_shift]

  - fractal_id:         discrete (for crossover / mutation)
  - crop_cx:            continuous (0-1)
  - crop_cy:            continuous (0-1)
  - crop_scale:         continuous (min - max))
  - brightness_shift:   continuous (-0.4 - 0.4)
"""
from config import GAConfig as ga
import numpy as np
import cv2
from config import Config
import config

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
    
    # mayyyyybe resize fractal to match tile? Or include this as a GA-evolvable trait?
    #fractal_resized = cv2.resize(fractal_image, (tile.shape[0]))

    # brightness shift
    fractal_shifted = fractal_image + brightness_shift
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
            