import numpy as np
import cupy as cp
import config

MAX_ITERATIONS = config.MAX_ITERATIONS

_gpu_target = None
_gpu_weights = None

def init_gpu(target, weights):
    global _gpu_target, _gpu_weights
    _gpu_target = cp.asarray(target, dtype=cp.float32)
    _gpu_weights = cp.asarray(weights, dtype=cp.float32)

#generates a fractal image based on the genotype
#THIS FUNCTION IS CALLED ONCE FOR THE FINAL RENDER
def generate_fractal_array(genotype):

    #initializes an array of 0's'
    layered_image = cp.zeros((config.FINAL_HEIGHT, config.FINAL_WIDTH), dtype=float)

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
        x = cp.linspace(x_min, x_max, config.FINAL_WIDTH)
        y = cp.linspace(y_min, y_max, config.FINAL_HEIGHT)

        # transforms 1d coordinate vectors into 2d coordinate matrices meshgrid(one or more 1d arrays,
        # copy?,
        # sparse?(return sparse coord. saves memory and computes on time on large grids.
        # indexing = 'xy' || 'ij' == '1st arr is x and 2nd arr is y' || 'dimensions are the same order as input arr'
        X, Y = cp.meshgrid(x, y, indexing = 'xy')

        #combines X and Y into a grid of complex numbers. 1j == python symbol for imaginary numbers
        z = X + 1j * Y

        #creates an array of 0's
        #zeros(shape, dtype, order) Shape = defines dimensions of an array,
        # dtype = datatype(opt),
        # order = memory layout c-style(row major) || fortran-style(column major)
        escape_times = cp.zeros((config.FINAL_HEIGHT, config.FINAL_WIDTH))

        #tracks the current pixels that are active (not escaped) init with 1(true) known as a mask
        active_pixels = cp.ones((config.FINAL_HEIGHT, config.FINAL_WIDTH), dtype=bool)

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

    #clips the pixel values to be between 0 and 255
    #cp.clip(a, a_min, a_max)
    layered_image = cp.clip(layered_image, 0, 255)

    return layered_image

#calculates the mean squared error between the generated fractal and the target image
def calculate_mse(image_a, image_b, weights):

    squared_diff = cp.square(image_a - image_b)

    weighted_mse = cp.average(squared_diff, weights = weights)

    return weighted_mse

#evaluates the genotype and returns the mean squared error
#DEPRECATED, USE batch_evaluate INSTEAD
def evaluate(genotype):
    fractal_arr = generate_fractal_array(genotype)
    return float(calculate_mse(fractal_arr, _gpu_target, weights = _gpu_weights))

#evaluates a batch of genotypes and returns a list of mean squared errors
#uses 4-dimensional matrix math to evaluate multiple genotypes at once
def batch_evaluate(population, batch_size):
    total_scores = []

    for i in range (0, len(population), batch_size):
        batch = population[i: i + batch_size]
        current_size = len(batch)
        #creates a 4-dimensional array of 0's to hold batch
        #np.zeros((batch_size, layers, height, width))
        c_real = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        c_imag = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        x_min = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        y_min = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        x_max = np.zeros((current_size, config.NUM_LAYERS, 1, 1))
        y_max = np.zeros((current_size, config.NUM_LAYERS, 1, 1))

        #populates the 4-dimensional array with the genotype parameters
        #batch[b].layers[l].c_real = genotype.layers[l]['c_real']
        #enumerate(): returns a tuple of the index and value of the iterable
        for b, genotype in enumerate(batch):
            for l, layer in enumerate(genotype.layers):
                c_real[b, l, 0, 0] = layer['c_real']
                c_imag[b, l, 0, 0] = layer['c_imag']
                x_min[b, l, 0, 0] = -1.5 / layer['zoom'] + layer['x_offset']
                y_min[b, l, 0, 0] = -1.5 / layer['zoom'] + layer['y_offset']
                x_max[b, l, 0, 0] = 1.5 / layer['zoom'] + layer['x_offset']
                y_max[b, l, 0, 0] = 1.5 / layer['zoom'] + layer['y_offset']

        #transfer the 4-dimensional array to a cupy array (GPU) as 32-bit floats
        c_real_cp = cp.asarray(c_real, dtype=cp.float32)
        c_imag_cp = cp.asarray(c_imag, dtype=cp.float32)
        x_min_cp = cp.asarray(x_min, dtype=cp.float32)
        y_min_cp = cp.asarray(y_min, dtype=cp.float32)
        x_max_cp = cp.asarray(x_max, dtype=cp.float32)
        y_max_cp = cp.asarray(y_max, dtype=cp.float32)

        #creates a grid of complex numbers and reshapes it to be 4-dimensional
        #linspace(start, stop, num): start <= stop, num > 0
        #reshape(w,x,y,z): w = width, x = height, y = depth, z = channels
        nx = cp.linspace(0, 1, config.GEN_WIDTH, dtype=cp.float32).reshape(1, 1, 1, config.GEN_WIDTH)
        ny = cp.linspace(0, 1, config.GEN_HEIGHT, dtype=cp.float32).reshape(1, 1, config.GEN_HEIGHT, 1)

        #combines the grid of complex numbers with the genotype parameters to create a grid of complex numbers
        X = x_min_cp + nx * (x_max_cp - x_min_cp)
        Y = y_min_cp + ny * (y_max_cp - y_min_cp)

        escape_times = cp.zeros((current_size, config.NUM_LAYERS, config.GEN_HEIGHT, config.GEN_WIDTH), dtype=cp.float32)
        active_pixels = cp.ones((current_size, config.NUM_LAYERS, config.GEN_HEIGHT, config.GEN_WIDTH), dtype=bool)

        for _ in range(MAX_ITERATIONS):
            x2 = X * X
            y2 = Y * Y

            #z = x + yi
            Y = 2.0 * X * Y + c_imag_cp
            #z = x - yi
            X = x2 - y2 + c_real_cp

            #checks if the pixel is escaped; identical to 'cp.abs(z) > 2.0' but much faster
            escaped = (x2 + y2) > 4.0
            new_escape = escaped & active_pixels
            escape_times[new_escape] = _
            #'~' is the bitwise NOT operator; the NOT keyword is equivalent to '~' but for single variables
            active_pixels = active_pixels & ~escaped

        #Find the max score for each individual layer to calculate percentages
        #cp.max(): returns the maximum value of the array
        #axis: axis to calculate the maximum value along
        #keepdims: if true, the result will be a 1-D array with the maximum value along the specified axis.
        max_scores = cp.max(escape_times, axis=(2, 3), keepdims=True)
        #replace 0s with 1s to prevent division by 0
        max_scores[max_scores == 0] = 1
        pixel_values = (escape_times / max_scores) * 255.0

        # Sums the pixel values across all layers to get the final image; axis = 1 sums across the columns
        layered_images = cp.sum(pixel_values, axis=1)
        #clips the pixel values to be between 0 and 255
        layered_images = cp.clip(layered_images, 0, 255)

        #calculates the weighted mean squared error between the generated fractal and the target image
        #_gpu_target: the target image on the GPU
        squared_diff = cp.square(layered_images - _gpu_target)
        # _gpu_weights: the weights of each layer on the GPU
        #axis=(1, 2): sum across the columns and rows of the array
        weighted_mse = cp.sum(squared_diff * _gpu_weights,axis=(1, 2)) / cp.sum(_gpu_weights)

        #converts the cupy array to a numpy array and returns the mean squared error
        batch_scores = cp.asnumpy(weighted_mse).tolist()
        total_scores.extend(batch_scores)

        #frees the GPU memory used by the arrays
        del X, Y, escape_times, active_pixels, squared_diff, layered_images, pixel_values
        cp.get_default_memory_pool().free_all_blocks()

    return total_scores




