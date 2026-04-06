
import cv2
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


def run_evo():
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    image_path = "maple_leaf.jpg"

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
                child = evolver.mutate(child, mutate_rate=0.1, mutate_value=0.1)
                new_population.append(child)

            active_population = new_population

    print("Evolution Complete, Generating Final Image...")

    final_fractal = evaluator.generate_fractal_array(best_genotype)
    file_name = f"final_fractal_{best_score:.0f}.png"
    cv2.imwrite(file_name, final_fractal)
    print(f"Final Fractal saved as {file_name}")

    print("\n---Final Fractal Genotype---")
    print(f"Real: {best_genotype.c_real:.4f}")
    print(f"Imaginary: {best_genotype.c_imag:.4f}")
    print(f"X Offset: {best_genotype.x_offset:.4f}")
    print(f"Y Offset: {best_genotype.y_offset:.4f}")
    print(f"Zoom: {best_genotype.zoom:.4f}")
    print("--------------------------------\n")

    cv2.imshow("Original Image", image_array)
    cv2.imshow("Final Fractal", final_fractal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_evo()