


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

    
from dataclasses import dataclass, field
import multiprocessing as mp
from math import pi

@dataclass 
class FractalConfig:
    c_real_range: tuple[float, float] = (-1.5, 1.5)     # (-1.0, 1.0)
    c_real: float = -0.8
    #
    c_imag_range: tuple[float, float] = (-1.5, 1.5)     # (-1.0, 1.0)
    c_imag: float = 0.17
    #
    x_offset_range: tuple[float, float] = (-1.0, 1.0)   # (-0.5, 0.5)
    x_offset: float = 0
    #
    y_offset_range: tuple[float, float] = (-1.0, 1.0)   # (-0.5, 0.5)
    y_offset: float = 0
    #
    zoom_range: tuple[float, float] = (0.5, 3.0)        # (0.5, 2.5)
    zoom: float = 1.0
    #
    x_min: float = -1.5
    x_max: float = 1.5
    y_min: float = -1.5
    y_max: float = 1.5

@dataclass
class MosaicConfig:
    grid_n: int = 12
    tile_size: int = 32
    mosaic_size: int = 256
    
    min_crop_scale: float = 0.03
    max_crop_scale: float = 1.0
    min_brightness_scale: float = -0.5
    max_brightness_scale: float = 0.5
    
@dataclass
class GAConfig:
    population_size: int = 60
    num_generations: int = 80
    elitism_rate: float = 0.05
    mutation_rate: float = 0.3
    mutation_intensity: float = 0.1    # How much a mutation can change a parameter
    
    tournament_size: int = 5
    selection_size: int = 20
    crossover_rate: float = 0.75
    
    # WEIGHTS for fitess
    weight_edge: float = 0.5    # (0.1 – 1.0)
    
    # ADAPTIVE MUTATION (decreases to the above rate, starting at the below rate)
    initial_mutation_intensity: float = 0.3    # intensity at gen 0
    # quit early if wasting time
    stagnation_limit: int = 15
    
    
@dataclass
class SampleFractalConfig:    
    
    # VALUES:
    #https://www.mintlify.com/ibon-ira/Fractol-42/fractals/julia#connected-vs-disconnected-sets
    #https://paulbourke.net/fractals/juliaset/
    #https://paulbourke.net/fractals/juliaset/julia_set.py
    
    # (c_real, c_imag, x_offset, y_offset, zoom)
    JULIA_SETS = [
        # Dark (low zoom)
        ( 0.36,  0.36,  0.0, 0.0, 0.8),
        (-1.77,  0.0,   0.0, 0.0, 0.6),
        (-0.12,  0.75,  0.0, 0.0, 0.8),

        # Mid
        (-0.7,   0.27,  0.0, 0.0, 1.0),
        (-0.8,   0.156, 0.0, 0.0, 1.0),
        ( 0.355, 0.355, 0.0, 0.0, 1.2),

        # Bright (higher zoom)
        (-0.75,  0.0,   0.0, 0.0, 1.5),
        (-1.25,  0.0,   0.0, 0.0, 1.0),
        (-0.75,  0.0,   0.0, 0.0, 3.0),
        
        
        # from https://paulbourke.net/fractals/juliaset/ and https://paulbourke.net/fractals/juliaset/julia_set.py
        ( 1.0,     0.0, -1.201171875, -0.9635417, 1.0),
        ( 0.0,     1.0, -0.5390625, -1.4296875, 1.0),
        ( 1.0,     0.2,    0.0, 0.0, 1.0),
        ( 1.0,     0.3,    0.0, 0.0, 1.0),
        ( 1.0,     0.4,    0.0, 0.0, 1.0),
        ( 0.0,     1.5,    0.0, 0.0, 1.0),   
        ( 1.0,     1.0,    0.0, 0.0, 1.0),
        ( 0.985,   0.174,  0.0, 0.0, 1.0),
        (-1.299,  -0.75,   0.0, 0.0, 1.0),
        ( 1.175,   0.428,  0.0, 0.0, 1.0),
        ( 1.879,   0.684,  0.0, 0.0, 1.0),
        (-0.2,     1.0,    0.0, 0.0, 1.0),
        ( 0.0,     1.0,    0.0, 0.0, 1.0),   
        (-0.123,   0.745,  0.0, 0.0, 1.0),   # douady rabbit
        (-0.75,    0.0,    0.0, 0.0, 1.0),   # san marco
        (-0.391,  -0.587,  0.0, 0.0, 1.0),   # siegel disk   
        (-0.54,   0.54,    0.0, 0.0, 0.9), 
        ( 0.45,   0.143,   0.0, 0.0, 1.2),
        (-0.7,    -0.3,    0.0, 0.0, 1.0),
        (-0.75,   -0.2,    0.0, 0.0, 1.0),
        (-0.75,    0.15,   0.0, 0.0, 1.0),
        (-0.7,     0.35,   0.0, 0.0, 1.0),
        ( 0.285,   0.01,   0.0, 0.0, 1.0),   
        (-0.4,     -0.6,   0.0, 0.0, 1.0),
    ] 
    sample_size: int = len(JULIA_SETS)         # NUmber of fractals to generate
    sample_resolution: int = 512              # resolution (px * px) of each sample fractal
    max_iterations: int = 80                  # GNERATIONS OF GA for SAMPLE fractals
    
# IF TIME ALLOWS, WE'LL GA OVER THOSE SAMPLE-FRACTALS TOO
@dataclass
class JuliaParams:
    c_real: float
    c_imag: float
    x_offset: float
    y_offset: float
    zoom: float
    

    
@dataclass
class Config:
    # ValueError: mutable default <class 'config.MosaicConfig'> for field mosaic is not allowed: use default_factory    
    mosaic: MosaicConfig = field(default_factory=MosaicConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    sampleFractals: SampleFractalConfig = field(default_factory=SampleFractalConfig)
    fractal: FractalConfig = field(default_factory=FractalConfig)
    
    image_path: str = "../IMG_6363.jpeg"
    output_path: str = "../output_images/"
    
    # == IMAGE PROCESSING == #
    canny_low:  float = 50.0
    canny_high: float = 150.0
    canny_dilation: int = 5
    
    
    ## == MULTIPROCESSING == ##
    #from dataclasses import field
    #num_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    num_workers: int = max(1, mp.cpu_count() - 1)
    