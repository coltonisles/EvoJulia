import cv2
import config

#TARGET_WIDTH = config.WIDTH
#TARGET_HEIGHT = config.HEIGHT

def load_and_process(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find or load image at {image_path}")
    #converts a colour image into a grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resizes the image to the target dimensions
    gray_img = cv2.resize(gray_img, (config.WIDTH, config.HEIGHT))

    #returns an image with edges being white and else everything else black
    edges = cv2.Canny(gray_img, 100, 200)

    weights = edges.astype(float)
    weights = (weights / 255.0) * 9.0 + 1.0

    return gray_img, weights

#testing only
#res = load_and_process("../IMG_6363.jpeg")
#cv2.imshow("Processed Target", res)
#cv2.waitKey(0)