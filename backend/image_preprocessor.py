import cv2

TARGET_WIDTH = 200
TARGET_HEIGHT = 200

def load_and_process(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not find or load image at {image_path}")
    #converts a colour image into a grayscale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #resizes the image to the target dimensions
    gray_img = cv2.resize(gray_img, (TARGET_WIDTH, TARGET_HEIGHT))
    return gray_img

#testing only
res = load_and_process("../IMG_6363.jpeg")
cv2.imshow("Processed Target", res)
cv2.waitKey(0)