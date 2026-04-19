import cv2
from config import MosaicConfig as Mosaic

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
import numpy as np
from PIL import Image

def prepare_GreyImage_float32_and_unit8(image_path, grid_n_by_n=Mosaic.grid_n, tile_size=Mosaic.tile_size):
    
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