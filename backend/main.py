import numpy as np
import cv2
import config
import image_preprocessor
import population_init
import fitness_evaluator

# 1. Load your target image (replace with your actual image path)
target_path = "../IMG_6363.jpeg"

# 1 & 2. Load your image AND unpack the generated weights at the same time!
target_image, weights = image_preprocessor.load_and_process(target_path)

# (You can delete the np.ones() line completely, because you are now
# using the actual edge-detection weights you built in the preprocessor!)

# 3. Grab the very first genotype from your initialized population
test_genotype = population_init.population[0]

# 4. Run the evaluation!
print("Evaluating Genotype 0...")
score = fitness_evaluator.evaluate(test_genotype, target_image, weights)

# 5. Print the results
print(f"Genotype Parameters: c=({test_genotype.c_real:.2f} + {test_genotype.c_imag:.2f}j), zoom={test_genotype.zoom:.2f}")
print(f"Fitness Score (MSE): {score:.2f}")

# --- BONUS: Let's actually look at the fractal! ---
# We can use the generator function to get the array and display it
fractal_array = fitness_evaluator.generate_fractal_array(test_genotype)

# Show the target image and the generated fractal side-by-side
cv2.imshow("Target Image", target_image)
cv2.imshow("Random Fractal", fractal_array)

print("Press any key to close the images...")
cv2.waitKey(0)
cv2.destroyAllWindows()