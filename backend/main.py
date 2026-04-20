## MULTIPROCESSING ##
import multiprocessing as mp
from config import GAConfig as ga, FractalConfig as fractal, MosaicConfig as mosaic, Config, SampleFractalConfig
import image_preprocessor as img
#import population_init as init
import mosaic_genome as init
import sample_fractals_generator as sampleFractals
import cv2
import numpy as np
import evolution
import argparse
import os
import time
import random


def main():
    num_workers = max(1, mp.cpu_count() - 1)
    
    ## TAKE USER ARGS FOR PARAMETER ADJUSTMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=Config.image_path)
    parser.add_argument("--output", default=Config.output_path)
    parser.add_argument("--population", type=int, default=ga.population_size)
    parser.add_argument("--generations", type=int, default=ga.num_generations)
    #
    # '--mosaic' == if used, GA will use fitness_full_mosaic() to evaluate entire image as one gigantic genome
    # (default is to judge each tile separately))
    parser.add_argument("--mosaic", action="store_true", default=False)
    
    args = parser.parse_args()
    
    ## 1. LOAD IMAGE (let there be light)
    #image_path = args.image    
    ##image_array, weights = img.load_and_process(image_path)
    


    #"""CANNY ISSUES?? https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
    #img = cv.imread('messi5.jpg', cv.IMREAD_GRAYSCALE)
    #assert img is not None, "file could not be read, check with os.path.exists()"
    #edges = cv.Canny(img,100,200)
    #"""


    ## EDGE DETECTION
    #edges = cv2.Canny(img_grey_uint8, 100, 200)
    #weights = edges.astype(float)
    #weights = (weights / 255.0) * 9.0 + 1.0
    
    #https://www.kaggle.com/code/sitinuradilla/psd-praktikum-image-analysis



    ## 1. LOAD IMAGE
    #image_path = args.image
    
    #    # resize
    #image_resized = cv2.resize(image_path, (mosaic.mosaic_size, mosaic.mosaic_size))

    #    # -> greyscale
    #img_grey = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    
    #    # Normalize 0-1
    #img_norm = img_grey / 255.0
    
    #    # Gaussian blur to reduce noice before finding edges
    ##img_blur = cv2.GaussianBlur(img_norm, (5, 5), 0)
    #img_blur = cv2.GaussianBlur(img_norm, (5, 5), 1)
    
    #    ## FEATURE EXTRACTION
    #edges = cv2.Canny()



    # 1. LOAD IMAGE    ## unit8 for Canny Edges, float32 for fitnessing
    image_path = args.image
    img_grey_float32, img_grey_uint8 = img.prepare_GreyImage_float32_and_unit8(image_path)

    # 1a. CANNY
        # Gaussian blur to reduce noice before finding edges
    #img_blur = cv2.GaussianBlur(img_grey_uint8, (5, 5), 0)
    img_blur = cv2.GaussianBlur(img_grey_uint8, (5, 5), 1.4)
    
        # NOW Canny
    img_edged = cv2.Canny(img_blur, Config.canny_low, Config.canny_high)
        # Add dilation to strengthen edges
    kernel = np.ones((Config.canny_dilation, Config.canny_dilation), np.uint8)
    img_dilated = cv2.dilate(img_edged, kernel)




    # Generate sample fractals, ordered by brightness (let there be wind)
    sample_fractals = sampleFractals.generate_multiple_julias()
    
    
    ## ====================================== ##
    ## ==== MOSAIC MODE (v tile by tile) ==== ##
    if args.mosaic:
        
        print(f"Mosaic Mode Initialized. Generating base population of {args.population} mosaic-sized genomes...")
        
        num_tiles = mosaic.grid_n * mosaic.grid_n
        
        # init population
        full_population = []
        for i in range(args.population):
            full_mosaic_genome = []
            for tile in range(num_tiles):
                
                individual = init.generate_random_individual()
                full_mosaic_genome.append(init.get_genome(individual))
            
            full_population.append(full_mosaic_genome)
            # That completes ONE member of the population 
            
        
        # ============= #
        # == FITNESS == #
        print(f"Evaluating {len(full_population)} mosaic-sized genomes...")
        
        for generation in range(args.generations):
            full_scores = []
            for i, full_genome in enumerate(full_population):
                score = evolution.fitness_full_mosaic(full_genome, sample_fractals, img_grey_float32)
                full_scores.append(score)
                
                if generation == 0 or (i+1) % 10 == 0:
                    print(f"Gen {generation+1}, Genome {i+1}/{len(full_population)}: score == {score:.5f}")
            
            # =============== #
            # == SELECTION == #
            selected_population = evolution.selection(full_population, full_scores)
            
            # =============== #
            # == ELITISM == #
            num_elites = int(ga.elitism_rate * args.population)
            new_population = selected_population[:num_elites]
            
            # ================= #
            # == CROSSOVER & MUTATION == #
            while len(new_population) < args.population:
                p1 = random.choice(selected_population)
                p2 = random.choice(selected_population)
                child = evolution.crossover_mosaic(p1, p2)
                child = evolution.mutate_mosaic(child, len(sample_fractals))
                new_population.append(child)
            
            full_population = new_population
        
        # After all generations, get the best
        best_full_genome = full_population[0]  # since selection sorts, but here we can evaluate again or assume
        
        # "Tiles! ... ASSEMBLE"
        tiles_assemble(best_full_genome, sample_fractals, args.output)
    
    ## ================================================== ##
    ## ==== INDIVIDUALS MODE (v full mosaic genomes) ==== ##
    else:
        
        
    #2. PREPARE INITAIL POPULATION (let there be life)
    #active_population = init.let_there_be_life()        # population[] of [genome]s
    
    #2a. let there be a LOT of life (for each tile)
    #num_samples = len(sample_fractals)
    #num_tiles = mosaic.grid_n
    #tile_size = mosaic.tile_size
    
    #H, W = img_grey_float32.shape       #[float32] instead of [uint8]
    #for x in range(num_tiles):
    #    for y in range(num_tiles):
    #        tile_data = img_grey_float32[
    #            x * tile_size: (x+1) * tile_size,
    #            y * tile_size: (y+1) * tile_size
    #        ]
    #        tile_population = [
    #            init.let_there_be_life()
    #        ]
    # ^^^^ = get_all_tiles()
    
        all_tiles = img.get_all_tiles(img_grey_float32)
        
        ## == GENETIC ALGORITHM TIME == ## 
        best_genome_per_tile = []       # getting ready
        
        for iter, tile_data in enumerate(all_tiles):
            print(f"GA Initializing for {iter+1}/{len(all_tiles)}...")
            
            # Populate every tile with randos
            tile_population = init.let_there_be_life()
            
            # ======================================= #
            # == STILL NEED MUTATION AND CROSSOVER == #
            
            for generation in range(args.generations):
                
                # ============= #
                # == FITNESS == #
                # judge every genome in THIS tile's population
                tile_scores = []
                for genome in tile_population:
                    score = evolution.evaluation(init.get_genome(genome), sample_fractals, tile_data)
                    tile_scores.append(score)
                
                
                # =============== #
                # == SELECTION == #
                # sorted as best up front
                selected_population = evolution.selection(tile_population, tile_scores)
                
                # =============== #
                # == ELITISM == #
                num_elites = int(ga.elitism_rate * args.population)
                new_population = selected_population[:num_elites]
                
                # ================= #
                # == CROSSOVER & MUTATION == #
                while len(new_population) < args.population:
                    p1 = random.choice(selected_population)
                    p2 = random.choice(selected_population)
                    child_genome = evolution.crossover(init.get_genome(p1), init.get_genome(p2))
                    child_genome = evolution.mutate(child_genome, len(sample_fractals))
                    child_individual = init.MosaicGenome(*child_genome)
                    new_population.append(child_individual)
                
                tile_population = new_population
            
        
            # == ALL GENERATIONS COMPLETE FOR THIS TILE == #
            best_genome_this_tile = init.get_genome(tile_population[0])
            #print(f"Best genome for tile {iter+1} = {best_genome_this_tile}.")
            best_genome_per_tile.append(best_genome_this_tile)
            
        # == ALL TILES COMPLETE == #
        tiles_assemble(best_genome_per_tile, sample_fractals, args.output)   
            
            
            
            
            
            
            #tile_scores = evolution.fitness(tile_population)
            
            
            ## RUN THE GA, get the best genome (for THIS tile)
            #best_genome = evolution.selection(tile_population, tile_scores)
            ## Store the best -- iter will be used later to reconstruct the mosaic grid
            #best_genome_per_tile.append(best_genome)
        
    


def tiles_assemble(best_genome_per_tile, sample_fractals, output_path): 
    tile_size = mosaic.tile_size    # 32px  ( == 32x32 px tile sizes)
    grid_n = mosaic.grid_n          # 12    ( == 12x12 grid of tiles)

    # Init output image as solid black wall
    output_image = np.zeros(( (grid_n * tile_size), (grid_n * tile_size) ), dtype=np.float32)
    
    iter = 0
    for row in range(grid_n):
        for col in range(grid_n):
            genome_list = best_genome_per_tile[iter]    # best genome for ITER tile, then for ITER+1 tile, etc...
            iter += 1
    
            # Unpack the genome to its params
            # genome: [fractal_id, crop_x, crop_y, crop_scale, brightness]
            fractal_id = int(genome_list[0])
            cx = float(genome_list[1])
            cy = float(genome_list[2])
            scale = float(genome_list[3])
            brightness = float(genome_list[4])
    
            # CROP the fractal image to the tile_size 
            fractal_img = sample_fractals[fractal_id]
            tile = evolution.crop_fractal(fractal_img, cx, cy, scale, tile_size, tile_size)
            
            # SHIFT BRIGHTNESS
            tile = np.clip((tile + brightness), 0.0, 1.0)       # clip() to lock in valid range of brightness (0-1)
            
            # POSITION that tile on the output grid
            y0 = row * tile_size
            x0 = col * tile_size
            output_image[y0:(y0 + tile_size), x0:(x0 + tile_size)] = tile
            
    # MOSAIC IMAGE IS CREATED! must convert to uint8 for cv2 saving
    output_uint8 = (output_image*255).astype(np.uint8)
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"output-{timestamp}.png"
    full_path = os.path.join(output_path, filename)
    success = cv2.imwrite(full_path, output_uint8)
    if success:
        print(f"SAVED mosaic to: {full_path}, go check it out!")
    else:
        print(f"FAILED to save mosaic to: {full_path}") 


if __name__ == "__main__":
    main()

# CONSTANTS AND VARIABLES TO TWEAK
#CONFIG = {
#    "image_path": "target.png" or "target.jpg" or "target.jpeg",
#    "output_path": "mosaic_output.png",
    
#    "grid_n": 12,
#    "tile_size": 32,
    
#    "min_crop_scale": 0.03,
#    "max_crop_scale": 1.0,
    
#    # SAMPLE FRACTALS
#    "sample_fractal_count": 10,     # NUmber of fractals to generate
#    "sample_fractal_resolution": 512,   # resolution (px * px) of each sample fractal
#    "max_iterations": 80,               # GNERATIONS OF GA for SAMPLE fractals
    
#    # ================================================= #
#    ## == GA PARAMS == ##
#    "population_size": 60,
#    NUM_GENERATIONS: 80,
#    ELITISM_RATE_FLOAT: 0.1,
#    MUTATION_RATE_FLOAT: 0.3,
#    MUTATION_INTENSITY_FLOAT: 0.08,     # How much a mutation can change a parameter
    
#    TOURNAMENT_SIZE: 5,
#    CROSSOVER_RATE_FLOAT: 0.2,          # Possibility of crossover between 2 parents (else a clone)
    
#}
#USE @DATACLASSES INSTEAD