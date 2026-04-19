import numpy as np
from single_fractal_generator import generate_julia
from PIL import Image
from config import SampleFractalConfig as SampleFractals


#RESOLUTION = 256   # tiny for testing
#MAX_ITER   = 40    # tiny for testing
MAX_ITER   = SampleFractals.max_iterations    
RESOLUTION = SampleFractals.sample_resolution



## VALUES:
##https://www.mintlify.com/ibon-ira/Fractol-42/fractals/julia#connected-vs-disconnected-sets
## (c_real, c_imag, x_offset, y_offset, zoom)
#JULIA_SETS = [
#    # Dark (low zoom)
#    ( 0.36,  0.36,  0.0, 0.0, 0.8),
#    (-1.77,  0.0,   0.0, 0.0, 0.6),
#    (-0.12,  0.75,  0.0, 0.0, 0.8),

#    # Mid
#    (-0.7,   0.27,  0.0, 0.0, 1.0),
#    (-0.8,   0.156, 0.0, 0.0, 1.0),
#    ( 0.355, 0.355, 0.0, 0.0, 1.2),

#    # Bright (higher zoom)
#    (-0.75,  0.0,   0.0, 0.0, 1.5),
#    (-1.25,  0.0,   0.0, 0.0, 1.0),
#    (-0.75,  0.0,   0.0, 0.0, 3.0),
#] 
# use SampleFractals.JULIA_SETS instead

images = []
means  = []

def generate_multiple_julias():
    for i, (c_real, c_imag, x_offset, y_offset, zoom) in enumerate(SampleFractals.JULIA_SETS):
        image = generate_julia(c_real, c_imag, x_offset, y_offset, zoom)
        images.append(image)                    # float32[]
        means.append(float(image.mean()))       # Each pixel is a 'greyness' / brightness value, so average = mean brightness
        
        print(f"{i}: mean-brightness={image.mean():.3f}")
        
        # TO SAVE THESE PRE-BUILT FRACTALS:
        img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(f"fractal-{i}-{image.mean():.3f}.png")
    
    print("exiting generate_multiple_julias()")
    
# FOR TESTING - UNCOMMENT TO RE-GENERATE
#generate_multiple_julias()