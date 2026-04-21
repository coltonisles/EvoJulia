import cv2
from config import MosaicConfig as Mosaic
import numpy as np
from PIL import Image


TARGET_WIDTH = 200
TARGET_HEIGHT = 200

GRID_N_BY_N = 12
TILE_SIZE = 32

def load_and_process(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find or load image at {image_path}")
    #converts a colour image into a grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resizes the image to the target dimensions
    #gray_img = cv2.resize(gray_img, (TARGET_WIDTH, TARGET_HEIGHT))
    side = grid_n_by_n * tile_size
    gray_img = cv2.resize(gray_img, (side, side))
    
    
    #returns an image with edges being white and else everything else black
    edges = cv2.Canny(gray_img, 100, 200)

    weights = edges.astype(float)
    weights = (weights / 255.0) * 9.0 + 1.0

    #return gray_img
    return gray_img, weights


#testing only
#res = load_and_process("../IMG_6363.jpeg")
#cv2.imshow("Processed Target", res)
#cv2.waitKey(0)




# ## == ALT == ## #

def prepare_GreyImage_float32_and_unit8(image_path, grid_n_by_n=None, tile_size=None):
    if grid_n_by_n is None:
        grid_n_by_n = Mosaic.grid_n
    if tile_size is None:
        tile_size = Mosaic.tile_size
    
    img = Image.open(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find or load image at {image_path}")
    
    side = grid_n_by_n * tile_size
    
    # GREYSCALE for Canny Edge Detection (uint8), and for tile merging and fitnessing (float32) -- https://www.geeksforgeeks.org/python/python-pil-image-convert-method/
    img_grey = img.convert("L").resize((side, side))
    # Image.LANCZOS might help add sharpness if edges are weak??
    #img_grey = img.convert("L").resize((side, side), Image.LANCZOS)
    
    img_grey_uint8 = np.array(img_grey, dtype=np.uint8)
    img_grey_float32 = img_grey_uint8.astype(np.float32) / 255.0    # Normalization 0-1
    #img_grey_float32 = img_grey_uint8.astype(np.float32) / 255.0 * 9.0 + 1.0
    
    return img_grey_float32, img_grey_uint8


def get_tile_xy(img, x_index, y_index, tile_size=Mosaic.tile_size):
    """ 
    Get (and build) tile (x, y) OF (if time allows) DYNAMIC SIZE in the mosaic grid
    
    Grid is n tiles wide and n tiles height.
    
    Full mosaic = H*W = (n * tile_size)(n * tile_size)
    
    Tiles can stretch to occupy a larger space on the grid instead of just tile_size*tile_size:
    h0 == STARTING ROW of tile 
    h1 == ENDING ROW of tile
    w0 == STARTING COL
    w1 == ENDING COL
    
    Returns img[h0:h1, w0:w1]
        # This is 2D NumPy array == the image. 
            # Forms an array of shape (h1 - h0, w1 - w0),
            # Taking rows from h0 to (h1 - 1)
            # and columns from w0 to (w1 - 1)
        # == 2D tile of size:
            # (h1-h0) rows
            # (w1-w0) columns
            
    example:
    h0 = x * tile_size = 4 * 32 = 128
    h1 = h0 + tile_size = 128 + 32 = 160
    w0 = y * tile_size = 7 * 32 = 224
    w1 = w0 + tile_size = 224 + 32 = 256
    --> img[128:160, 224:256]
        == 32rows x 32cols == 1 tile at (x,y) of img[]
    """
    if tile_size is None:
        tile_size = Mosaic.tile_size
    h0 = x_index * tile_size
    h1 = h0 + tile_size
    w0 = y_index * tile_size
    w1 = w0 + tile_size
    return img[h0:h1, w0:w1]

def get_all_tiles(img, num_tiles=None, tile_size=None):
    if num_tiles is None:
        num_tiles = Mosaic.grid_n
    if tile_size is None:
        tile_size = Mosaic.tile_size
    
    tiles = []
    
    #num_samples = len(sample_fractals)
    #H, W = img_grey_float32.shape       #[float32] instead of [uint8]
    
    for x in range(num_tiles):
        for y in range(num_tiles):
            tile = img[
                x * tile_size: (x+1) * tile_size,
                y * tile_size: (y+1) * tile_size
            ]
            
            tiles.append(tile)
            
    return tiles


#tile_population = [
#    init.let_there_be_life()
#]

