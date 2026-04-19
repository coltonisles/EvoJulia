# So I don't forget during lunch... 

""" BASIC STRUCTURE
population init
loop generations:
    evaluate fitness
    select
    crossover
    mutate
    replace
return best

"""

# Basically... 
def evolve_population():
    population = [...]
    
    for gen in range(num_generations):
        fitnesses = batch_fitness(population)

        new_population = elites

        while len(new_population) < pop_size:
            p1 = select(...)
            p2 = select(...)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    return best_individual