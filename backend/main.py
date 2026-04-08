
import cv2
import numpy as np
import psutil
import os
import config
import image_preprocessor
import population_init
import evaluator
import evolver
from tqdm import tqdm
import random
from concurrent.futures import ProcessPoolExecutor
from skimage.exposure import match_histograms


def run_evo():
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    image_path = "Images/helix_nebula.jpg"

    image_array, weights = image_preprocessor.load_and_process(image_path)

    active_population = population_init.population

    with ProcessPoolExecutor(initializer=evaluator.init_worker, initargs=(image_array, weights)) as executor:

        for generation in tqdm(range(config.GENERATION_SIZE), desc="Overall Progress", position=0):

            scores = list(executor.map(evaluator.evaluate, active_population, chunksize=10))

            best_score = min(scores)
            best_index = scores.index(best_score)
            best_genotype = active_population[best_index]

            tqdm.write(f"Generation {generation} Best MSE: {best_score:.2f}")

            parents = evolver.select_parents(active_population,scores, 20)
            new_population = [best_genotype]

            while len(new_population) < config.POPULATION_SIZE:
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)

                child = evolver.crossover(parent1, parent2)
                child = evolver.mutate(child, config.MUTATION_RATE, config.MUTATION_RANGE)
                new_population.append(child)

            active_population = new_population

    print("Evolution Complete, Generating Final Image...")

    final_fractal = evaluator.generate_fractal_array(best_genotype)

    rgb_origin = cv2.imread(image_path)
    rgb_origin = cv2.resize(rgb_origin, (config.WIDTH, config.HEIGHT))

    rgb_final = cv2.cvtColor(final_fractal, cv2.COLOR_GRAY2RGB)

    colored_fractal = match_histograms(rgb_final, rgb_origin, channel_axis=-1)

    colored_fractal = np.uint8(colored_fractal)

    combined_image = np.hstack((rgb_origin, colored_fractal))

    basePath = image_path.split('.')[0]
    file_name = f"{basePath}_{best_score:.0f}.png"
    cv2.imwrite(file_name, combined_image)
    print(f"Final Fractal saved as {file_name}")

    print("\n---Final Fractal Genotype---")
    for i, layer in enumerate(best_genotype.layers):
        print(f"Layer {i + 1}:")
        print(f"  Real: {layer['c_real']: .4f}, Imag: {layer['c_imag']: .4f}")
        print(f"  X/Y: {layer['x_offset']: .4f} / {layer['y_offset']: .4f}, Zoom: {layer['zoom']: .4f}")
    print("--------------------------------\n")

    cv2.imshow("Original VS Fractal", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_evo()