
import cv2
import numpy as np
import cupy as cp
import psutil
import os
import config
import image_preprocessor
import population_init
import evaluator
import evolver
from tqdm import tqdm
import random
from skimage.exposure import match_histograms
import imageio


def run_evo():
    #potentially remove this line later
    p = psutil.Process(os.getpid())
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)

    image_path = config.IMAGE_PATH

    image_array, weights = image_preprocessor.load_and_process(image_path)

    active_population = population_init.population

    evaluator.init_gpu(image_array, weights)

    rgb_origin_sm = cv2.imread(image_path)
    rgb_origin_sm = cv2.resize(rgb_origin_sm, (config.FINAL_WIDTH, config.FINAL_HEIGHT))

    gif_frames = []

    snapshot_rate = config.SNAPSHOT_RATE

    gen_hist = []

    #main genetic algorithm loop
    #tqdm() is a progress bar that displays the progress of the loop
    for generation in tqdm(range(config.GENERATION_SIZE), desc="Overall Progress", position=0):

        #simulated annealing algorithm: reduces mutation rate over time
        #allows for an initial high mutation rate and gradually reduces it to provide more precision over time
        precision_factor = max(0.01, 1.0 - (generation / config.GENERATION_SIZE))

        #multiplies the mutation rate and range by the precision factor to reduce the mutation rate over time
        current_mutation_rate = config.MUTATION_RATE * precision_factor
        current_mutation_range = config.MUTATION_RANGE * precision_factor

        #evaluates the current population batch
        scores = evaluator.batch_evaluate(active_population, config.BATCH_SIZE)

        #find best scores (lowest value) and corresponding genotypes
        best_score = min(scores)
        best_index = scores.index(best_score)
        best_genotype = active_population[best_index]
        #tqdm.write(): prints a message to the console without interrupting the progress bar
        tqdm.write(f"Generation {generation} Best MSE: {best_score:.2f} || Mutation Rate: {current_mutation_rate:.2%}")

        gen_hist.append(f"Generation {generation} | MSE: {best_score:.2f} | Genotype: {best_genotype}")

        #selects the top 20 parents for the next generation
        parents = evolver.select_parents(active_population,scores, config.SELECTION_SIZE)
        #elitism: the best genotype from the previous generation is always preserved
        #ensures the score will never decrease
        new_population = [best_genotype]

        #breeding loop until the population is full
        while len(new_population) < config.POPULATION_SIZE:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            child = evolver.crossover(parent1, parent2)
            child = evolver.mutate(child, current_mutation_rate, current_mutation_range)
            new_population.append(child)

        if generation % snapshot_rate == 0:
            frame_fractal = evaluator.generate_fractal_array(best_genotype)
            frame_fractal = cp.asnumpy(frame_fractal).astype(np.uint8)

            rgb_frame = cv2.cvtColor(frame_fractal, cv2.COLOR_GRAY2RGB)
            colored_frame = match_histograms(rgb_frame, rgb_origin_sm, channel_axis=-1)
            colored_frame = np.uint8(colored_frame)

            fixed_colored_frame = cv2.cvtColor(colored_frame, cv2.COLOR_BGR2RGB)

            gif_frames.append(fixed_colored_frame)

        active_population = new_population

    print("Evolution Complete, Generating Final Image...")

    final_fractal = evaluator.generate_fractal_array(best_genotype)

    final_fractal = cp.asnumpy(final_fractal).astype(np.uint8)

    rgb_origin = cv2.imread(image_path)
    rgb_origin = cv2.resize(rgb_origin, (config.FINAL_WIDTH, config.FINAL_HEIGHT))


    rgb_final = cv2.cvtColor(final_fractal, cv2.COLOR_GRAY2RGB)

    colored_fractal = match_histograms(rgb_final, rgb_origin, channel_axis=-1)

    colored_fractal = np.uint8(colored_fractal)

    combined_image = np.hstack((rgb_origin, colored_fractal))



    base_path = os.path.basename(image_path)
    base_name = base_path.split('.')[0]
    folder_name = f"{base_name}_{best_score:.0f}"

    save_directory = os.path.join("Output", folder_name)

    os.makedirs(save_directory, exist_ok=True)
    print(f"\nCreated new save folder: {save_directory}")



    file_name = f"{base_name}_{best_score:.0f}.png"
    total_path = os.path.join(save_directory, file_name)

    print("Creating GIF...")

    gif_path = os.path.join(save_directory, f"{base_name}_timelapse.gif")

    imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
    print(f"Timelapse saved as {gif_path}")

    cv2.imwrite(total_path, combined_image)
    print(f"Final Fractal saved as {total_path}")

    log_path = os.path.join(save_directory, f"{base_name}_{best_score:0.2f}_log.txt")

    with open(log_path, "w") as file:
        file.write(f"Target Image: {base_name}\n")
        file.write(f"Final Best MSE: {best_score:.2f}\n")
        file.write("--------------------------------\n")
        for i, layer in enumerate(best_genotype.layers):
            file.write(f"Evolutionary History for {base_name}\n")
            file.write("=========================================\n")
            # .join() efficiently writes every item in the list separated by a newline
            file.write("\n".join(gen_hist))

    print(f"Genotype history saved to {log_path}")
    print("\n--------------------------------\n")

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
