import random
import config
from population_init import Genotype


def select_parents(population, scores, selection_size):

    scored_genotypes = list(zip(population, scores))
    sorted_scored_genotypes = sorted(scored_genotypes, key=lambda pair: pair[1])
    top_scored_genotypes = sorted_scored_genotypes[:selection_size]
    return [pair[0] for pair in top_scored_genotypes]

def crossover(parent1, parent2):
    new_layers = []
    for i in range(config.NUM_LAYERS):
        layer = {
            'c_real': random.choice([parent1.layers[i]['c_real'], parent2.layers[i]['c_real']]),
            'c_imag': random.choice([parent1.layers[i]['c_imag'], parent2.layers[i]['c_imag']]),
            'x_offset': random.choice([parent1.layers[i]['x_offset'], parent2.layers[i]['x_offset']]),
            'y_offset': random.choice([parent1.layers[i]['y_offset'], parent2.layers[i]['y_offset']]),
            'zoom': random.choice([parent1.layers[i]['zoom'], parent2.layers[i]['zoom']])
        }
        new_layers.append(layer)
    return Genotype(new_layers)

def mutate(genotype, mutate_rate, mutate_range):
    for i in range(config.NUM_LAYERS):
        if random.random() < mutate_rate:
            new_val = genotype.layers[i]['c_real'] + random.uniform(-mutate_range, mutate_range)
            genotype.layers[i]['c_real'] = max(-1.0, min(1.0, new_val))
        if random.random() < mutate_rate:
            new_val = genotype.layers[i]['c_imag'] + random.uniform(-mutate_range, mutate_range)
            genotype.layers[i]['c_imag'] = max(-1.0, min(1.0, new_val))
        if random.random() < mutate_rate:
            genotype.layers[i]['x_offset'] += random.uniform(-mutate_range, mutate_range)
        if random.random() < mutate_rate:
            genotype.layers[i]['y_offset'] += random.uniform(-mutate_range, mutate_range)
        if random.random() < mutate_rate:
            new_zoom = genotype.layers[i]['zoom'] + random.uniform(-mutate_range, mutate_range)
            genotype.layers[i]['zoom'] = max(0.1, new_zoom)
    return genotype