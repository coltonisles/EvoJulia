## MULTIPROCESSING ##
import multiprocessing as mp
from config import GAConfig as ga, FractalConfig as fractal, MosaicConfig as mosaic, Config, SampleFractalConfig
import image_preprocessor as img
import population_init as initPop
import sample_fractals_generator as sampleFractals
import cv2
import numpy as np


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


import argparse


def canny_edging(grey_uint8)

def main():
    num_workers = max(1, mp.cpu_count() - 1)
    
    ## TAKE USER ARGS FOR PARAMETER ADJUSTMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=Config.image_path)
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--population", type=int, default=ga.population_size)
    parser.add_argument("--generations", type=int, default=ga.num_generations)
    
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


    #2. PREPARE INITAIL POPULATION (let there be life)
    active_population = initPop.population
    
    
    