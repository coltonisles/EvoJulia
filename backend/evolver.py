import random
import config
from population_init import Genotype

#finds the best genotypes in the current generation; lower score = better
def select_parents(population, scores, selection_size):

    #list() converts a generator into a list
    #zip() combines two or more iterables into a tuple, where the i-th element comes from the i-th iterable argument
    scored_genotypes = list(zip(population, scores))
    #sorted() sorts the list from lowest to highest score
    #key=lambda pair: pair[1] tells sorted to look at the second element of the tuple
    sorted_scored_genotypes = sorted(scored_genotypes, key=lambda pair: pair[1])
    #returns the top-scored genotypes based on the selection size and scores
    top_scored_genotypes = sorted_scored_genotypes[:selection_size]
    #returns only the genotypes from the pairs
    return [pair[0] for pair in top_scored_genotypes]

#creates a new genotype by random selection of the genotype parameters from parents and swapping
def crossover(parent1, parent2):
    new_layers = []
    #flip a coin to decide which parent to use for each parameter of child
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

#adds a random amount of mutation(mutate range) to each parameter of the genotype based on the mutate rate
def mutate(genotype, mutate_rate, mutate_range):

    #ruels for each parameter of the genotype to mutate (trait name, min limit, max limit)
    mutate_rules = [
        ('c_real', -1.0, 1.0),
        ('c_imag', -1.0, 1.0),
        ('x_offset', None, None), #offsets can drift infinitely
        ('y_offset', None, None),
        ('zoom', 0.1, None) #zoom has no max limit but can only go as far back as 0.1
    ]
    #loops through each layer of the genotype
    for i in range(config.NUM_LAYERS):
        #loop through each rule and mutate the parameter if the random number is less than the mutate rate
        for trait, min_limit,  max_limit in mutate_rules:
            #mutate_rate is the probability of mutating a parameter
            #random.random() generates a random number between 0 and 1
            if random.random() < mutate_rate:
                #mutate_range is the amount to mutate the parameter by
                #random.uniform(x, y) generates a random number between the given range x, y
                new_val = genotype.layers[i][trait] + random.uniform(-mutate_range, mutate_range)
                #limits the new value to the min and max limits if they exist
                if min_limit is not None:
                    new_val = max(min_limit, new_val)
                if max_limit is not None:
                    new_val = min(max_limit, new_val)

                genotype.layers[i][trait] = new_val



        '''
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
            '''
    return genotype