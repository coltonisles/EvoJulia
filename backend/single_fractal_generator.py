import numpy as np
from PIL import Image

WIDTH = 800
LENGTH = 800

#complex(x, y): x: -2 <= x <=2, y: -2 <= y <= 2
#test value cr = -0.8, ci = 0.17 for good fractal
c_real = -0.7089246769139015
c_imag = 1.1302003595409786

x_offset = 0
y_offset = 0
zoom = 1.0

MAX_ITERATIONS = 255

c = complex(c_real, c_imag)
x_min = (-1.5/ zoom + x_offset)
y_min = (-1.5 / zoom + y_offset)
x_max = (1.5 / zoom + x_offset)
y_max = (1.5 / zoom + y_offset)

#linspace is used to create a linear space of values starting at start and ending at stop and
# distributes of the num number. start and stop are the first and last values of the array
# i.e. (0, 10, 5) creates an array of 5 values start at 0 to 10 (0, 2.5, 5, 7.5, 10)
#linspace(start, stop, num): start <= stop, num > 0 |||||||||| num = length
x = np.linspace(x_min, x_max, WIDTH)
y = np.linspace(y_min, y_max, LENGTH)

# transforms 1d coordinate vectors into 2d coordinate matrices meshgrid(one or more 1d arrays,
# copy?,
# sparse?(return sparse coord. saves memory and computes on time on large grids.
# indexing = 'xy' || 'ij' == '1st arr is x and 2nd arr is y' || 'dimensions are the same order as input arr'
X, Y = np.meshgrid(x, y, indexing = 'xy')

#combines X and Y into a grid of complex numbers. j == imaginary number
z = X + 1j * Y

#creates an array of 0's
#zeros(shape, dtype, order) Shape = defines dimensions of an array,
# dtype = datatype(opt),
# order = memory layout c-style(row major) || fortran-style(column major)
escape_times = np.zeros((LENGTH, WIDTH))

#tracks the current pixels that are active (not escaped) init with 1(true) known as a mask
active_pixels = np.ones((LENGTH, WIDTH), dtype=bool)

for i in range(MAX_ITERATIONS):
    #applies the function to the pixels that are active
    z[active_pixels] = z[active_pixels]**2 + c
    #checks the entire grid at once for magnitude > 2
    escaped = np.abs(z) > 2.0
    #find pixels that just escaped in this loop
    new_escape = escaped & active_pixels
    #records the current loop i as the score for the newly escaped pixels
    escape_times[new_escape] = i
    #updates the 'mask' of active pixels to exclude the newly escaped pixels
    active_pixels = active_pixels & ~escaped

#calculates the highest escape time; ensures not 0
max_score = np.max(escape_times)
if max_score == 0:
    max_score = 1

#calc brightness based on escape time. escape_time / max_score * 255 to get the percentage (0.0 -> 1.0). 0 = black, 255 = white
#max_score to get an average brightness to prevent very dark pixels
pixel_values = (escape_times / max_score) * 255

#Image cannot read decimal points so must convert to unsigned integers (uint8; 8bit unsigned int)
img_data = np.uint8(pixel_values)
#renders the image from the array. expects uint8 dtype, mode = 'L'(GreyScale), 'RGB', or 'RGBA' (A = alpha value)
final_image = Image.fromarray(img_data)
#displays the final image
final_image.show()

#final_image.save('fractal.png')