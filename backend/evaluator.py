import numpy as np
import cupy as cp
import config

WIDTH = config.WIDTH
HEIGHT = config.HEIGHT
MAX_ITERATIONS = config.MAX_ITERATIONS

_gpu_target = None
_gpu_weights = None

def init_gpu(target, weights):
    global _gpu_target, _gpu_weights
    _gpu_target = cp.asarray(target, dtype=cp.float32)
    _gpu_weights = cp.asarray(weights, dtype=cp.float32)

def generate_fractal_array(genotype):

    #LOOK INTO BATCHING/VECTORIZATION FOR MASS TRANSMIT TO GPU

    layered_image = cp.zeros((HEIGHT, WIDTH), dtype=float)

    for layer in genotype.layers:
        c = complex(layer['c_real'], layer['c_imag'])
        x_min = (-1.5 / layer['zoom'] + layer['x_offset'])
        y_min = (-1.5 / layer['zoom'] + layer['y_offset'])
        x_max = (1.5 / layer['zoom'] + layer['x_offset'])
        y_max = (1.5 / layer['zoom'] + layer['y_offset'])

        #linspace is used to create a linear space of values starting at start and ending at stop and
        # distributes of the num number. start and stop are the first and last values of the array
        # i.e. (0, 10, 5) creates an array of 5 values start at 0 to 10 (0, 2.5, 5, 7.5, 10)
        #linspace(start, stop, num): start <= stop, num > 0); num = length of the array
        x = cp.linspace(x_min, x_max, WIDTH)
        y = cp.linspace(y_min, y_max, HEIGHT)

        # transforms 1d coordinate vectors into 2d coordinate matrices meshgrid(one or more 1d arrays,
        # copy?,
        # sparse?(return sparse coord. saves memory and computes on time on large grids.
        # indexing = 'xy' || 'ij' == '1st arr is x and 2nd arr is y' || 'dimensions are the same order as input arr'
        X, Y = cp.meshgrid(x, y, indexing = 'xy')

        #combines X and Y into a grid of complex numbers. j == imaginary number
        z = X + 1j * Y

        #creates an array of 0's
        #zeros(shape, dtype, order) Shape = defines dimensions of an array,
        # dtype = datatype(opt),
        # order = memory layout c-style(row major) || fortran-style(column major)
        escape_times = cp.zeros((HEIGHT, WIDTH))

        #tracks the current pixels that are active (not escaped) init with 1(true) known as a mask
        active_pixels = cp.ones((HEIGHT, WIDTH), dtype=bool)

        for i in range(MAX_ITERATIONS):
            #applies the function to the pixels that are active
            z = z**2 + c
            #checks the entire grid at once for magnitude > 2
            escaped = cp.abs(z) > 2.0
            #find pixels that just escaped in this loop
            new_escape = escaped & active_pixels
            #records the current loop i as the score for the newly escaped pixels
            escape_times[new_escape] = i
            #updates the 'mask' of active pixels to exclude the newly escaped pixels
            active_pixels = active_pixels & ~escaped

        #calculates the highest escape time; ensures not 0
        max_score = cp.max(escape_times)
        if max_score == 0:
            max_score = 1

        #calc brightness based on escape time. escape_time / max_score * 255 to get the percentage (0.0 -> 1.0). 0 = black, 255 = white
        #max_score to get an average brightness to prevent very dark pixels
        pixel_values = (escape_times / max_score) * 255
        layered_image += pixel_values

    layered_image = cp.clip(layered_image, 0, 255)

    return layered_image
#need to update to do the work on the gpu too
def calculate_mse(image_a, image_b, weights):

    squared_diff = cp.square(image_a - image_b)

    weighted_mse = cp.average(squared_diff, weights = weights)

    return weighted_mse

def evaluate(genotype):
    fractal_arr = generate_fractal_array(genotype)
    return float(calculate_mse(fractal_arr, _gpu_target, weights = _gpu_weights))


def batch_evaluate(population, batch_size=10):
    total_scores = []

    for i in range (0, len(population), batch_size):
        batch = population[i: i + batch_size]
        current_size = len(batch)
        c_real = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        c_imag = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        x_min = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        y_min = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        x_max = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        y_max = np.zeros((current_size, config.NUM_LAYERS, 1, 1))

        for b, genotype in enumerate(batch):
            for l, layer in enumerate(genotype.layers):
                c_real[b, l, 0, 0] = layer['c_real']
                c_imag[b, l, 0, 0] = layer['c_imag']
                x_min[b, l, 0, 0] = -1.5 / layer['zoom'] + layer['x_offset']
                y_min[b, l, 0, 0] = -1.5 / layer['zoom'] + layer['y_offset']
                x_max[b, l, 0, 0] = 1.5 / layer['zoom'] + layer['x_offset']
                y_max[b, l, 0, 0] = 1.5 / layer['zoom'] + layer['y_offset']

        c_real_cp = cp.asarray(c_real, dtype=cp.float32)
        c_imag_cp = cp.asarray(c_imag, dtype=cp.float32)
        x_min_cp = cp.asarray(x_min, dtype=cp.float32)
        y_min_cp = cp.asarray(y_min, dtype=cp.float32)
        x_max_cp = cp.asarray(x_max, dtype=cp.float32)
        y_max_cp = cp.asarray(y_max, dtype=cp.float32)

        nx = cp.linspace(0, 1, WIDTH, dtype=cp.float32).reshape(1, 1, 1, WIDTH)
        ny = cp.linspace(0, 1, HEIGHT, dtype=cp.float32).reshape(1, 1, HEIGHT, 1)

        X = x_min_cp + nx * (x_max_cp - x_min_cp)
        Y = y_min_cp + ny * (y_max_cp - y_min_cp)

        escape_times = cp.zeros((current_size, config.NUM_LAYERS, HEIGHT, WIDTH), dtype=cp.float32)
        active_pixels = cp.ones((current_size, config.NUM_LAYERS, HEIGHT, WIDTH), dtype=bool)

        for _ in range(MAX_ITERATIONS):
            x2 = X * X
            y2 = Y * Y

            Y = 2.0 * X * Y + c_imag_cp
            X = x2 - y2 + c_real_cp

            escaped = (x2 + y2) > 4.0
            new_escape = escaped & active_pixels
            escape_times[new_escape] = _
            active_pixels = active_pixels & ~escaped

        max_scores = cp.max(escape_times, axis=(2, 3), keepdims=True)
        max_scores[max_scores == 0] = 1
        pixel_values = (escape_times / max_scores) * 255.0

        layered_images = cp.sum(pixel_values, axis=1)
        layered_images = cp.clip(layered_images, 0, 255)

        squared_diff = cp.square(layered_images - _gpu_target)
        weighted_mse = cp.sum(squared_diff * _gpu_weights,axis=(1, 2)) / cp.sum(_gpu_weights)

        batch_scores = cp.asnumpy(weighted_mse).tolist()
        total_scores.extend(batch_scores)

        del X, Y, escape_times, active_pixels, squared_diff, layered_images, pixel_values
        cp.get_default_memory_pool().free_all_blocks()

    return total_scores




