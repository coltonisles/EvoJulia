import random
import config
from population_init import Genotype


def select_parents(population, scores, selection_size):

    scored_genotypes = list(zip(population, scores))
    sorted_scored_genotypes = sorted(scored_genotypes, key=lambda pair: pair[1])
    top_scored_genotypes = sorted_scored_genotypes[:selection_size]
    selected_parents = [pair[0] for pair in top_scored_genotypes]

    return selected_parents

def crossover(parent1, parent2):
    c_real_new = random.choice([parent1.c_real, parent2.c_real])
    c_imag_new = random.choice([parent1.c_imag, parent2.c_imag])
    x_offset_new = random.choice([parent1.x_offset, parent2.x_offset])
    y_offset_new = random.choice([parent1.y_offset, parent2.y_offset])
    zoom_new = random.choice([parent1.zoom, parent2.zoom])
    return Genotype(c_real_new, c_imag_new, x_offset_new, y_offset_new, zoom_new)

def mutate(genotype, mutate_rate=0.1, mutate_value=0.1):
    c_real_chance = random.random()
    c_imag_chance = random.random()
    x_offset_chance = random.random()
    y_offset_chance = random.random()
    zoom_chance = random.random()

    if c_real_chance < mutate_rate:
        genotype.c_real += random.uniform(-mutate_value, mutate_value)
    if c_imag_chance < mutate_rate:
        genotype.c_imag += random.uniform(-mutate_value, mutate_value)
    if x_offset_chance < mutate_rate:
        genotype.x_offset += random.uniform(-mutate_value, mutate_value)
    if y_offset_chance < mutate_rate:
        genotype.y_offset += random.uniform(-mutate_value, mutate_value)
    if zoom_chance < mutate_rate:
        genotype.zoom += random.uniform(-mutate_value, mutate_value)

    return genotype