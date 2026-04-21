## MULTIPROCESSING ##
import multiprocessing as mp
#from config import GAConfig as ga, FractalConfig as fractal, MosaicConfig as mosaic, Config, SampleFractalConfig
import config
from config import GAConfig as ga, FractalConfig as fractal, Config, SampleFractalConfig
mosaic = config.MosaicConfig()
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
    parser.add_argument("--tile_size", type=int, default=mosaic.tile_size)
    parser.add_argument("--grid_n", type=int, default=mosaic.grid_n)
    #
    # '--mosaic' == if used, GA will use fitness_full_mosaic() to evaluate entire image as one gigantic genome
    # (default is to judge each tile separately))
    parser.add_argument("--mosaic", action="store_true", default=False)
    
    # == SUPERTILE ARGS == #
    # '--merge' == if used, similar adjacent tiles merge into 2x2 supertiles AFTER the GA finishes
    # '--merge_tolerance' == brightness range threshold; higher == more merges
    parser.add_argument("--merge", action="store_true", default=False)
    parser.add_argument("--merge_tolerance", type=float, default=mosaic.merge_tolerance)
    
    args = parser.parse_args()
    
    # update existing valuesbased on new args
    #if args.grid_n is not None:
    #    mosaic.grid_n = args.grid_n
    #if args.tile_size is not None:
    #    mosaic.tile_size = args.tile_size
        
    if args.grid_n != parser.get_default("grid_n"):
        config.MosaicConfig.grid_n = args.grid_n
        mosaic.grid_n = args.grid_n

    if args.tile_size != parser.get_default("tile_size"):
        config.MosaicConfig.tile_size = args.tile_size
        mosaic.tile_size = args.tile_size
        
        
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
    # Get the brightnesses for each for matchmaking
    fractal_brightnesses = [float(frac.mean()) for frac in sample_fractals]
    
    all_tiles = img.get_all_tiles(img_grey_float32)
    
    
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
        #best_full_genome = full_population[0]  # since selection sorts already 
        # oops that was actually just pulling randoms, not sorted = not the best
        
        # After the last generation, re-evaluate and pick the actual best
        final_scores = [evolution.fitness_full_mosaic(g, sample_fractals, img_grey_float32) for g in full_population]
        best_id = final_scores.index(min(final_scores))
        best_full_genome = full_population[best_id]
        
                # ============== #
        # == SUPERTILE MERGE == #
        # If --merge, check for groups of 4 similar tiles (2x2) and mark them as supertiles
        supertile_map = {}      # { tile_index : "skip" or "supertile_2x2" }
        if args.merge:
            #all_tiles = img.get_all_tiles(img_grey_float32)
            best_full_genome, supertile_map = merge_similar_tiles_2x2(
                best_full_genome, all_tiles, sample_fractals,
                grid_n=mosaic.grid_n,
                brightness_tolerance=args.merge_tolerance
            )
        
        #final_scores = [evolution.fitness_full_mosaic(g, sample_fractals, img_grey_float32) for g in full_population]
        #best_score = min(final_scores)
        
        #tile_best_scores = []

        #for g in best_genome_per_tile:
        #    # score each tile genome individually
        #    tile_id = best_genome_per_tile.index(g)
        #    tile_data = all_tiles[tile_id]
        #    score = evolution.evaluation(g, sample_fractals, tile_data)
        #    tile_best_scores.append(score)

        #best_score = min(tile_best_scores)
        #avg_best_score = sum(tile_best_scores) / len(tile_best_scores)
        
        final_score = evolution.fitness_full_mosaic(best_full_genome, sample_fractals, img_grey_float32)
        

        # "Tiles! ... ASSEMBLE"
        #tiles_assemble(best_full_genome, sample_fractals, args.output, args)
        tiles_assemble(
            best_full_genome,
            sample_fractals,
            args.output,
            args,
            supertile_map,
            final_score,
            mosaic.tile_size,
            mosaic.grid_n,
            ga.selection_size,
            ga.crossover_rate,
            ga.weight_edge,
            args.merge_tolerance
        )
        #tiles_assemble(best_full_genome, sample_fractals, args.output, args, supertile_map)
        
    
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
    
        #all_tiles = img.get_all_tiles(img_grey_float32)
        
        ## == GENETIC ALGORITHM TIME == ## 
        best_genome_per_tile = []       # getting ready
        
        for iter, tile_data in enumerate(all_tiles):
            
            # Populate every tile with randos
            tile_population = init.let_there_be_life()
            
            # ======================================= #
            # == STILL NEED MUTATION AND CROSSOVER == #
            
            for generation in range(args.generations):
                
                # == ADAPTIVE SELECTION == #                
                progress = generation / max(args.generations - 1, 1)    # 0.0 -> 1.0
                current_intensity = ga.initial_mutation_intensity + (ga.mutation_intensity - ga.initial_mutation_intensity) * progress

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
                #selected_population = evolution.selection(tile_population, tile_scores)
                selected_population = evolution.tournament_selection(tile_population, tile_scores)
                
                
                # =============== #
                # == ELITISM == #
                num_elites = int(ga.elitism_rate * args.population)
                new_population = selected_population[:num_elites]
                
                # ================= #
                # == CROSSOVER & MUTATION == #
                while len(new_population) < args.population:
                    p1 = random.choice(selected_population)
                    p2 = random.choice(selected_population)
                    #child_genome = evolution.crossover(init.get_genome(p1), init.get_genome(p2))
                    child_genome = evolution.single_point_crossover(init.get_genome(p1), init.get_genome(p2))
                    
                    #child_genome = evolution.mutate(child_genome, len(sample_fractals))
                    child_genome = evolution.mutate(child_genome, len(sample_fractals), intensity=current_intensity)
                    
                    child_individual = init.MosaicGenome(*child_genome)
                    new_population.append(child_individual)
                
                tile_population = new_population
                print(f"Tile {iter+1}/{len(all_tiles)}, Gen {generation+1}/{args.generations}, Best Score == {tile_scores[0]:.5f}")
                
        
            # == ALL GENERATIONS COMPLETE FOR THIS TILE == #
            best_genome_this_tile = init.get_genome(tile_population[0])
            #print(f"Best genome for tile {iter+1} = {best_genome_this_tile}.")
            best_genome_per_tile.append(best_genome_this_tile)
            #print(f"Tile {iter+1}/{len(all_tiles)}, Gen {generation+1}/{args.generations}, Best Score == {tile_scores[0]:.5f}")
            
            
        # == ALL TILES COMPLETE == #
        
        # ===================== #
        # == SUPERTILE MERGE == #
        # If --merge, check for groups of 4 similar tiles (2x2) and mark them as supertiles
        supertile_map = {}      # { tile_index : "skip" or "supertile_2x2" }
        if args.merge:
            best_genome_per_tile, supertile_map = merge_similar_tiles_2x2(
                best_genome_per_tile, all_tiles, sample_fractals,
                grid_n=mosaic.grid_n,
                brightness_tolerance=args.merge_tolerance
            )
        
        #tiles_assemble(best_genome_per_tile, sample_fractals, args.output, args, supertile_map)   
        
        #final_scores = [evolution.fitness_full_mosaic(g, sample_fractals, img_grey_float32) for g in full_population]
        #best_score = min(final_scores)
        

        
        tile_best_scores = []
        for g in best_genome_per_tile:
            tile_id = best_genome_per_tile.index(g)
            tile_data = all_tiles[tile_id]
            score = evolution.evaluation(g, sample_fractals, tile_data)
            tile_best_scores.append(score)
        
        best_score = min(tile_best_scores)
        avg_best_score = sum(tile_best_scores) / len(tile_best_scores)
        
        tiles_assemble(
            best_genome_per_tile,
            sample_fractals,
            args.output,
            args,
            supertile_map,
            best_score,
            mosaic.tile_size,
            mosaic.grid_n,
            ga.selection_size,
            ga.crossover_rate,
            ga.weight_edge,
            args.merge_tolerance
        )
            
            
            
                        
            
            
            
            
            
            #tile_scores = evolution.fitness(tile_population)
            
            
            ## RUN THE GA, get the best genome (for THIS tile)
            #best_genome = evolution.selection(tile_population, tile_scores)
            ## Store the best -- iter will be used later to reconstruct the mosaic grid
            #best_genome_per_tile.append(best_genome)
        
    

# ================================================================ #
# == MERGE 4 SIMILAR TILES INTO 1 BIG SUPERTILE IF CLOSE ENOUGH == #
#def merge_similar_tiles_2x2(best_genomes, all_tiles, sample_fractals, grid_n=mosaic.grid_n, brightness_tolerance=mosaic.merge_tolerance):
def merge_similar_tiles_2x2(best_genomes, all_tiles, sample_fractals, grid_n=None, brightness_tolerance=None):
    
    grid_n = grid_n or mosaic.grid_n
    brightness_tolerance = brightness_tolerance or mosaic.merge_tolerance
    
    # First, score each tile's FINAL genome (we'll use the best of the 4 of the 2x2 supertile)
    tile_scores = []
    for i in range(len(all_tiles)):
        score = evolution.evaluation(best_genomes[i], sample_fractals, all_tiles[i])
        tile_scores.append(score)
    
    # Will return a mutated copy of best_genomes + a supertile_map
    merged_genomes = []
    for genome in best_genomes:
        merged_genomes.append(list(genome))     # copy so we can mutate
    
    supertile_map = {}
    
    # Track number of merged supertiles (fyi)
    num_supertiles = 0
    
    # Check every SECOND tile x SECOND tile (the 2x2 blocks that tile the grid)
    for row in range(0, grid_n - 1, 2):
        for col in range(0, grid_n - 1, 2):
            # specific indeces of each tile composing the 2x2 supertile
            top_left = row * grid_n + col
            top_right = row * grid_n + col + 1
            bottom_left = (row + 1) * grid_n + col
            bottom_right = (row + 1) * grid_n + col + 1
            
            # Check if all indices are within bounds
            if top_right >= len(all_tiles) or bottom_left >= len(all_tiles) or bottom_right >= len(all_tiles):
                continue
            
            supertile_indeces = [top_left, top_right, bottom_left, bottom_right]
            
            # mean brightnesses of each tile in the supertile
            supertile_brightnesses = []
            for tile_index in supertile_indeces:
                supertile_brightnesses.append(float(all_tiles[tile_index].mean()))
            
            # Check for similarity == are all 4 brightnesses within the CONFIG tolerance?
            brightnesses_range = max(supertile_brightnesses) - min(supertile_brightnesses)
            
            # if so:
            if brightnesses_range < brightness_tolerance:
                # which subtile is the best
                winner_index = supertile_indeces[0]
                winner_score = tile_scores[winner_index]
                
                for tile_index in supertile_indeces[1:]:
                    if tile_scores[tile_index] < winner_score:
                        winner_score = tile_scores[tile_index]
                        winner_index = tile_index
                
                # Winner's genome applied to the top-left of this supertile (the one that will be rendered big)
                winner_genome = list(best_genomes[winner_index])
                merged_genomes[top_left] = winner_genome
                
                # Mark supertile positions in the sidecar map
                supertile_map[top_left] = "supertile_2x2"   # render as 2x2
                supertile_map[top_right] = "skip"
                supertile_map[bottom_left] = "skip"
                supertile_map[bottom_right] = "skip"
                
                num_supertiles += 1
    
    print(f"Created {num_supertiles} supertiles! (merge_tolerance = {brightness_tolerance})")
    return merged_genomes, supertile_map


# ======================================================== #
# == ASSEMBLE FINAL IMAGE (including supertile support) == #
# supertile_map == OPTIONAL (defaults to empty {} so old everything should still work)
def tiles_assemble(
    best_genome_per_tile,
    sample_fractals,
    output_path,
    args,
    supertile_map=None,
    best_score=None,
    tile_size=None,
    grid_n=None,
    selection_size=None,
    crossover_rate=None,
    weight_edge=None,
    merge_tolerance=None
):
    if supertile_map is None:
        supertile_map = {}
    
    #tile_size = mosaic.tile_size    # 32px  ( == 32x32 px tile sizes)
    #grid_n = mosaic.grid_n          # 12    ( == 12x12 grid of tiles)

    tile_size = tile_size or mosaic.tile_size
    grid_n = grid_n or mosaic.grid_n

    # Init output image as solid black wall
    output_image = np.zeros(( (grid_n * tile_size), (grid_n * tile_size) ), dtype=np.float32)
    
    iter = 0
    for row in range(grid_n):
        for col in range(grid_n):
            if iter >= len(best_genome_per_tile):
                break
            genome_list = best_genome_per_tile[iter]    # best genome for ITER tile, then for ITER+1 tile, etc...
            tile_index = iter
            iter += 1
            
            # == SUPERTILE CHECK == #
            # If this tile is part of a supertile (and NOT the top-left), skip
            role = supertile_map.get(tile_index, None)
            if role == "skip":
                continue
    
            # Unpack the genome to its params
            # genome: [fractal_id, crop_x, crop_y, crop_scale, brightness]
            fractal_id = int(genome_list[0])
            cx = float(genome_list[1])
            cy = float(genome_list[2])
            scale = float(genome_list[3])
            brightness = float(genome_list[4])
            
            # == RENDER SIZE == #
            # Normal tile = tile_size x tile_size
            # 2x2 supertile = (2*tile_size) x (2*tile_size), as ONE image (not 4 copies)
            if role == "supertile_2x2":
                render_size = tile_size * 2
            else:
                render_size = tile_size
    
            # CROP the fractal image to the tile_size (or supertile_size if --merge) 
            # SAME cx/cy/scale, just rendered to a larger output grid
            fractal_img = sample_fractals[fractal_id]
            tile = evolution.crop_fractal(fractal_img, cx, cy, scale, render_size, render_size)
            
            # SHIFT BRIGHTNESS
            tile = np.clip((tile + brightness), 0.0, 1.0)       # clip() to lock in valid range of brightness (0-1)
            
            # POSITION that tile on the output grid
            y0 = row * tile_size
            x0 = col * tile_size
            if y0 + render_size <= output_image.shape[0] and x0 + render_size <= output_image.shape[1]:
                output_image[y0:(y0 + render_size), x0:(x0 + render_size)] = tile
            
    # MOSAIC IMAGE IS CREATED! must convert to uint8 for cv2 saving
    output_uint8 = (output_image*255).astype(np.uint8)
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")[-6:]
    

    
    # build filename
    base = (
        f"score{best_score:.5f}"
        f"_TileSize{tile_size}"
        f"_GridN{grid_n}"
        f"_SelSize{ga.selection_size}"
        f"_CrossRate{ga.crossover_rate}"
        f"_WeightEdge{ga.weight_edge}"
        f"_pop{args.population}"
        f"_gens{args.generations}"
    )
    # add "-merged" to filename if supertiles were used
    merge_tag = f"-merged_MergeTolerance{merge_tolerance}" if supertile_map else ""
    mosaic_tag = "-mosaic" if args.mosaic else ""

    filename = f"{base}{merge_tag}{mosaic_tag}.png"
    
    #if args.mosaic:
    #    mosa
    #    filename = f"output-mosaic{merge_tag}-pop{args.population}-gens{args.generations}-{timestamp}.png"
    #else:
    #    filename = f"output-tiled{merge_tag}-pop{args.population}-gens{args.generations}-{timestamp}.png"
    
    
    
    
    
    full_path = os.path.join(output_path, filename)
    success = cv2.imwrite(full_path, output_uint8)
    if success:
        print(f"SAVED mosaic as: {filename}, go check it out!")
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