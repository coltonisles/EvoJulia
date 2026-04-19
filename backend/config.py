


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

    
from dataclasses import dataclass

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
    
@dataclass
class GAConfig:
    population_size: int = 60
    num_generations: int = 80
    elitism_rate: float = 0.1
    mutation_rate: float = 0.3
    mutation_intensity: float = 0.08     # How much a mutation can change a parameter
    
    tournament_size: int = 5
    selection_size: int = 40
    crossover_rate: float = 0.2
    
@dataclass
class SampleFractalConfig:
    sample_size: int = 10                     # NUmber of fractals to generate
    sample_resolution: int = 512              # resolution (px * px) of each sample fractal
    max_iterations: int = 80                  # GNERATIONS OF GA for SAMPLE fractals
    
    
    # VALUES:
    #https://www.mintlify.com/ibon-ira/Fractol-42/fractals/julia#connected-vs-disconnected-sets
    # (c_real, c_imag, x_offset, y_offset, zoom)
    JULIA_SETS: list[tuple[float, float, float, float, float]] = [
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
    ] 
    
@dataclass
class Config:
    mosaic: MosaicConfig = MosaicConfig()
    ga: GAConfig = GAConfig()
    sampleFractals: SampleFractalConfig = SampleFractalConfig()
    fractal: FractalConfig = FractalConfig()
    
    image_path: str = "../IMG_6363.jpeg"
    output_path: str = "../output_images/mosaic_output.png"
    
    # == IMAGE PROCESSING == #
    canny_low:  float = 50.0
    canny_high: float = 150.0
    canny_dilation: int = 3
    
    
    ## == MULTIPROCESSING == ##
    #from dataclasses import field
    import multiprocessing as mp    
    #num_workers: int = field(default_factory=lambda: max(1, multiprocessing.cpu_count() - 1))
    num_workers: int = max(1, mp.cpu_count() - 1)
    