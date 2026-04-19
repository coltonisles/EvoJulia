## MULTIPROCESSING ##
import multiprocessing as mp
n_workers = args.workers if args.workers > 0 else max(1, mp.cpu_count() - 1)

# CONSTANTS AND VARIABLES TO TWEAK
CONFIG = {
    "image_path": "target.png" or "target.jpg" or "target.jpeg",
    "output_path": "mosaic_output.png",
    
    "grid_n": 12,
    "tile_size": 32,
    
    
    
}