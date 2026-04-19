


# CONSTANTS AND VARIABLES TO TWEAK
CONFIG = {
    "image_path": "target.png" or "target.jpg" or "target.jpeg",
    "output_path": "mosaic_output.png",
    
    "grid_n": 12,
    "tile_size": 32,
    
    "min_crop_scale": 0.03,
    "max_crop_scale": 1.0,
    
    # SAMPLE FRACTALS
    "sample_fractal_count": 10,     # NUmber of fractals to generate
    "sample_fractal_resolution": 512,   # resolution (px * px) of each sample fractal
    "max_iterations": 80,               # GNERATIONS OF GA for SAMPLE fractals
    
    # ================================================= #
    ## == GA PARAMS == ##
    "population_size": 60,
    NUM_GENERATIONS: 80,
    ELITISM_RATE_FLOAT: 0.1,
    MUTATION_RATE_FLOAT: 0.3,
    MUTATION_INTENSITY_FLOAT: 0.08,     # How much a mutation can change a parameter
    
    TOURNAMENT_SIZE: 5,
    CROSSOVER_RATE_FLOAT: 0.2,          # Possibility of crossover between 2 parents (else a clone)
    
    
    
    
}


from dataclasses import dataclass

@dataclass
class MosaicConfig:
    grid_n: int = 12
    tile_size: int = 32
    
    min_crop_scale: 0.03
    max_crop_scale: 1.0
    
@dataclass
class GAConfig:
    population_size: 60
    num_generations: 80
    elitism_rate: 0.1
    mutation_rate: 0.3
    mutation_intensity: 0.08     # How much a mutation can change a parameter
    
    tournament_size: 5
    selection_size: 40
    crossover_rate: 0.2
    
@dataclass
class SampleFractalConfig:
    sample_size: 10                     # NUmber of fractals to generate
    sample_resolution: 512              # resolution (px * px) of each sample fractal
    max_iterations: 80                  # GNERATIONS OF GA for SAMPLE fractals
    
    
@dataclass
class Config:
    mosaic: MosaicConfig = MosaicConfig()
    ga: GAConfig = GAConfig()
    sampleFractals: SampleFractalConfig = SampleFractalConfig()
    
    output_path: "mosaic_output.png"
    
    ## == MULTIPROCESSING == ##
    from dataclasses import field
    import multiprocessing    
    n_workers: int = field(
        default_factory=lambda: max(1, multiprocessing.cpu_count() - 1)
    )
    