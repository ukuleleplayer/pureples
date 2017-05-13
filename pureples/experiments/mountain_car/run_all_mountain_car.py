import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neat_mountain_car
import hyperneat_mountain_car
import es_hyperneat_mountain_car_small
import es_hyperneat_mountain_car_medium
import es_hyperneat_mountain_car_large
import gym
import multiprocessing as multi
from multiprocessing import Manager

# Initialize lists to keep track during run.
manager = Manager()

neat_stats, hyperneat_stats, es_hyperneat_small_stats = manager.list([]), manager.list([]), manager.list([])
es_hyperneat_medium_stats, es_hyperneat_large_stats = manager.list([]), manager.list([])

neat_run_one_fitnesses, hyperneat_run_one_fitnesses, es_hyperneat_small_run_one_fitnesses = [], [], []
es_hyperneat_medium_run_one_fitnesses, es_hyperneat_large_run_one_fitnesses = [], []

neat_run_ten_fitnesses, hyperneat_run_ten_fitnesses, es_hyperneat_small_run_ten_fitnesses = [], [], [] 
es_hyperneat_medium_run_ten_fitnesses, es_hyperneat_large_run_ten_fitnesses = [], []

neat_one_solved, hyperneat_one_solved, es_hyperneat_small_one_solved = 0, 0, 0
es_hyperneat_medium_one_solved, es_hyperneat_large_one_solved = 0, 0

neat_ten_solved, hyperneat_ten_solved, es_hyperneat_small_ten_solved = 0, 0, 0 
es_hyperneat_medium_ten_solved, es_hyperneat_large_ten_solved = 0, 0

runs = 16
inputs = range(runs)
gens = 200
fit_threshold = -110
max_fit = -110
env = gym.make("MountainCar-v0")


# Run the experiments.
def run(i):
    print("This is run #" + str(i))
    neat_stats.append(neat_mountain_car.run(gens, env)[1])
    hyperneat_stats.append(hyperneat_mountain_car.run(gens, env)[1])
    es_hyperneat_small_stats.append(es_hyperneat_mountain_car_small.run(gens, env)[1])
    es_hyperneat_medium_stats.append(es_hyperneat_mountain_car_medium.run(gens, env)[1])
    es_hyperneat_large_stats.append(es_hyperneat_mountain_car_large.run(gens, env)[1])

p = multi.Pool(multi.cpu_count())
p.map(run,range(runs))

# Average the NEAT runs.
temp_fit_one = [0.0] * gens
temp_fit_ten = [0.0] * gens
for (stat_one, stat_ten) in neat_stats:
    if stat_one.best_genome().fitness > max_fit:
        neat_run_one_fitnesses.append(max_fit)
    else:
        neat_run_one_fitnesses.append(stat_one.best_genome().fitness)
    if stat_ten.best_genome().fitness > max_fit:
        neat_run_ten_fitnesses.append(max_fit)
    else:
        neat_run_ten_fitnesses.append(stat_one.best_genome().fitness)
        
    if stat_one.best_genome().fitness >= fit_threshold:
        neat_one_solved += 1
    if stat_ten.best_genome().fitness >= fit_threshold:
        neat_ten_solved += 1
    for i in range(gens):
        if i < len(stat_one.most_fit_genomes):
            if stat_one.most_fit_genomes[i].fitness > max_fit:
                temp_fit_one[i] += max_fit
            else:
                temp_fit_one[i] += stat_one.most_fit_genomes[i].fitness
        else:
            temp_fit_one[i] += max_fit
        if i < len(stat_ten.most_fit_genomes):
            if stat_ten.most_fit_genomes[i].fitness > max_fit:
                temp_fit_ten[i] += max_fit
            else:
                temp_fit_ten[i] += stat_ten.most_fit_genomes[i].fitness
        else:
            temp_fit_ten[i] += max_fit

neat_one_average_fit = [x / runs for x in temp_fit_one]
neat_ten_average_fit = [x / runs for x in temp_fit_ten]

# Average the HyperNEAT runs.
temp_fit_one = [0.0] * gens
temp_fit_ten = [0.0] * gens
for (stat_one, stat_ten) in hyperneat_stats:
    if stat_one.best_genome().fitness > max_fit:
        hyperneat_run_one_fitnesses.append(max_fit)
    else:
        hyperneat_run_one_fitnesses.append(stat_one.best_genome().fitness)
    if stat_ten.best_genome().fitness > max_fit:
        hyperneat_run_ten_fitnesses.append(max_fit)
    else:
        hyperneat_run_ten_fitnesses.append(stat_one.best_genome().fitness)

    if stat_one.best_genome().fitness >= fit_threshold:
        hyperneat_one_solved += 1
    if stat_ten.best_genome().fitness >= fit_threshold:
        hyperneat_ten_solved += 1
    for i in range(gens):
        if i < len(stat_one.most_fit_genomes):
            if stat_one.most_fit_genomes[i].fitness > max_fit:
                temp_fit_one[i] += max_fit
            else:
                temp_fit_one[i] += stat_one.most_fit_genomes[i].fitness
        else:
            temp_fit_one[i] += max_fit
        if i < len(stat_ten.most_fit_genomes):
            if stat_ten.most_fit_genomes[i].fitness > max_fit:
                temp_fit_ten[i] += max_fit
            else:
                temp_fit_ten[i] += stat_ten.most_fit_genomes[i].fitness
        else:
            temp_fit_ten[i] += max_fit

hyperneat_one_average_fit = [x / runs for x in temp_fit_one]
hyperneat_ten_average_fit = [x / runs for x in temp_fit_ten]

# Average the small ES-HyperNEAT runs.
temp_fit_one = [0.0] * gens
temp_fit_ten = [0.0] * gens
for (stat_one, stat_ten) in es_hyperneat_small_stats:
    if stat_one.best_genome().fitness > max_fit:
        es_hyperneat_small_run_one_fitnesses.append(max_fit)
    else:
        es_hyperneat_small_run_one_fitnesses.append(stat_one.best_genome().fitness)
    if stat_ten.best_genome().fitness > max_fit:
        es_hyperneat_small_run_ten_fitnesses.append(max_fit)
    else:
        es_hyperneat_small_run_ten_fitnesses.append(stat_one.best_genome().fitness)

    if stat_one.best_genome().fitness >= fit_threshold:
        es_hyperneat_small_one_solved += 1
    if stat_ten.best_genome().fitness >= fit_threshold:
        es_hyperneat_small_ten_solved += 1
    for i in range(gens):
        if i < len(stat_one.most_fit_genomes):
            if stat_one.most_fit_genomes[i].fitness > max_fit:
                temp_fit_one[i] += max_fit
            else:
                temp_fit_one[i] += stat_one.most_fit_genomes[i].fitness
        else:
            temp_fit_one[i] += max_fit
        if i < len(stat_ten.most_fit_genomes):
            if stat_ten.most_fit_genomes[i].fitness > max_fit:
                temp_fit_ten[i] += max_fit
            else:
                temp_fit_ten[i] += stat_ten.most_fit_genomes[i].fitness
        else:
            temp_fit_ten[i] += max_fit

es_hyperneat_small_one_average_fit = [x / runs for x in temp_fit_one]
es_hyperneat_small_ten_average_fit = [x / runs for x in temp_fit_ten]

# Average the medium ES-HyperNEAT runs.
temp_fit_one = [0.0] * gens
temp_fit_ten = [0.0] * gens
for (stat_one, stat_ten) in es_hyperneat_medium_stats:
    if stat_one.best_genome().fitness > max_fit:
        es_hyperneat_medium_run_one_fitnesses.append(max_fit)
    else:
        es_hyperneat_medium_run_one_fitnesses.append(stat_one.best_genome().fitness)
    if stat_ten.best_genome().fitness > max_fit:
        es_hyperneat_medium_run_ten_fitnesses.append(max_fit)
    else:
        es_hyperneat_medium_run_ten_fitnesses.append(stat_one.best_genome().fitness)

    if stat_one.best_genome().fitness >= fit_threshold:
        es_hyperneat_medium_one_solved += 1
    if stat_ten.best_genome().fitness >= fit_threshold:
        es_hyperneat_medium_ten_solved += 1
    for i in range(gens):
        if i < len(stat_one.most_fit_genomes):
            if stat_one.most_fit_genomes[i].fitness > max_fit:
                temp_fit_one[i] += max_fit
            else:
                temp_fit_one[i] += stat_one.most_fit_genomes[i].fitness
        else:
            temp_fit_one[i] += max_fit
        if i < len(stat_ten.most_fit_genomes):
            if stat_ten.most_fit_genomes[i].fitness > max_fit:
                temp_fit_ten[i] += max_fit
            else:
                temp_fit_ten[i] += stat_ten.most_fit_genomes[i].fitness
        else:
            temp_fit_ten[i] += max_fit

es_hyperneat_medium_one_average_fit = [x / runs for x in temp_fit_one]
es_hyperneat_medium_ten_average_fit = [x / runs for x in temp_fit_ten]

# Average the large ES-HyperNEAT runs.
temp_fit_one = [0.0] * gens
temp_fit_ten = [0.0] * gens
for (stat_one, stat_ten) in es_hyperneat_large_stats:
    if stat_one.best_genome().fitness > max_fit:
        es_hyperneat_large_run_one_fitnesses.append(max_fit)
    else:
        es_hyperneat_large_run_one_fitnesses.append(stat_one.best_genome().fitness)
    if stat_ten.best_genome().fitness > max_fit:
        es_hyperneat_large_run_ten_fitnesses.append(max_fit)
    else:
        es_hyperneat_large_run_ten_fitnesses.append(stat_one.best_genome().fitness)

    if stat_one.best_genome().fitness >= fit_threshold:
        es_hyperneat_large_one_solved += 1
    if stat_ten.best_genome().fitness >= fit_threshold:
        es_hyperneat_large_ten_solved += 1
    for i in range(gens):
        if i < len(stat_one.most_fit_genomes):
            if stat_one.most_fit_genomes[i].fitness > max_fit:
                temp_fit_one[i] += max_fit
            else:
                temp_fit_one[i] += stat_one.most_fit_genomes[i].fitness
        else:
            temp_fit_one[i] += max_fit
        if i < len(stat_ten.most_fit_genomes):
            if stat_ten.most_fit_genomes[i].fitness > max_fit:
                temp_fit_ten[i] += max_fit
            else:
                temp_fit_ten[i] += stat_ten.most_fit_genomes[i].fitness
        else:
            temp_fit_ten[i] += max_fit

es_hyperneat_large_one_average_fit = [x / runs for x in temp_fit_one]
es_hyperneat_large_ten_average_fit = [x / runs for x in temp_fit_ten]

# Write fitnesses to files.
# NEAT.
thefile = open('neat_mountain_car_run_fitnesses.txt', 'w+')
thefile.write("NEAT one\n")
for item in neat_run_one_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in neat_one_average_fit:
    thefile.write("NEAT one solves mountain_car at generation: " + str(neat_one_average_fit.index(max_fit)))
else:
    thefile.write("NEAT one does not solve mountain_car with best fitness: " + str(neat_one_average_fit[gens-1]))
thefile.write("\nNEAT one solves mountain_car in " + str(neat_one_solved) + " out of " + str(runs) + " runs.\n")
thefile.write("NEAT ten\n")
for item in neat_run_ten_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in neat_ten_average_fit:
    thefile.write("NEAT ten solves mountain_car at generation: " + str(neat_ten_average_fit.index(max_fit)))
else:
    thefile.write("NEAT ten does not solve mountain_car with best fitness: " + str(neat_ten_average_fit[gens-1]))
thefile.write("\nNEAT ten solves mountain_car in " + str(neat_ten_solved) + " out of " + str(runs) + " runs.\n")

# HyperNEAT.
thefile = open('hyperneat_mountain_car_run_fitnesses.txt', 'w+')
thefile.write("HyperNEAT one\n")
for item in hyperneat_run_one_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in hyperneat_one_average_fit:
    thefile.write("HyperNEAT one solves mountain_car at generation: " + str(hyperneat_one_average_fit.index(max_fit)))
else:
    thefile.write("HyperNEAT one does not solve mountain_car with best fitness: " + str(hyperneat_one_average_fit[gens-1]))
thefile.write("\nHyperNEAT one solves mountain_car in " + str(hyperneat_one_solved) + " out of " + str(runs) + " runs.\n")
thefile.write("HyperNEAT ten\n")
for item in hyperneat_run_ten_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in hyperneat_ten_average_fit:
    thefile.write("HyperNEAT ten solves mountain_car at generation: " + str(hyperneat_ten_average_fit.index(max_fit)))
else:
    thefile.write("HyperNEAT ten does not solve mountain_car with best fitness: " + str(hyperneat_ten_average_fit[gens-1]))
thefile.write("\nHyperNEAT ten solves mountain_car in " + str(hyperneat_ten_solved) + " out of " + str(runs) + " runs.\n")

# ES-HyperNEAT small.
thefile = open('es_hyperneat_mountain_car_small_run_fitnesses.txt', 'w+')
thefile.write("ES-HyperNEAT small one\n")
for item in es_hyperneat_small_run_one_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in es_hyperneat_small_one_average_fit:
    thefile.write("ES-HyperNEAT small one solves mountain_car at generation: " + str(es_hyperneat_small_one_average_fit.index(max_fit)))
else:
    thefile.write("ES-HyperNEAT small one does not solve mountain_car with best fitness: " + str(es_hyperneat_small_one_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT small one solves mountain_car in " + str(es_hyperneat_small_one_solved) + " out of " + str(runs) + " runs.\n")
thefile.write("ES-HyperNEAT small ten\n")
for item in es_hyperneat_small_run_ten_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in es_hyperneat_small_ten_average_fit:
    thefile.write("ES-HyperNEAT small ten solves mountain_car at generation: " + str(es_hyperneat_small_ten_average_fit.index(max_fit)))
else:
    thefile.write("ES-HyperNEAT small ten does not solve mountain_car with best fitness: " + str(es_hyperneat_small_ten_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT small ten solves mountain_car in " + str(es_hyperneat_small_ten_solved) + " out of " + str(runs) + " runs.\n")

# ES-HyperNEAT medium.
thefile = open('es_hyperneat_mountain_car_medium_run_fitnesses.txt', 'w+')
thefile.write("ES-HyperNEAT medium one\n")
for item in es_hyperneat_medium_run_one_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in es_hyperneat_medium_one_average_fit:
    thefile.write("ES-HyperNEAT medium one solves mountain_car at generation: " + str(es_hyperneat_medium_one_average_fit.index(max_fit)))
else:
    thefile.write("ES-HyperNEAT medium one does not solve mountain_car with best fitness: " + str(es_hyperneat_medium_one_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT medium one solves mountain_car in " + str(es_hyperneat_medium_one_solved) + " out of " + str(runs) + " runs.\n")
thefile.write("ES-HyperNEAT medium ten\n")
for item in es_hyperneat_medium_run_ten_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in es_hyperneat_medium_ten_average_fit:
    thefile.write("ES-HyperNEAT medium ten solves mountain_car at generation: " + str(es_hyperneat_medium_ten_average_fit.index(max_fit)))
else:
    thefile.write("ES-HyperNEAT medium ten does not solve mountain_car with best fitness: " + str(es_hyperneat_medium_ten_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT medium ten solves mountain_car in " + str(es_hyperneat_medium_ten_solved) + " out of " + str(runs) + " runs.\n")

# ES-HyperNEAT large.
thefile = open('es_hyperneat_mountain_car_large_run_fitnesses.txt', 'w+')
thefile.write("ES-HyperNEAT large one\n")
for item in es_hyperneat_large_run_one_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in es_hyperneat_large_one_average_fit:
    thefile.write("ES-HyperNEAT large one solves mountain_car at generation: " + str(es_hyperneat_large_one_average_fit.index(max_fit)))
else:
    thefile.write("ES-HyperNEAT large one does not solve mountain_car with best fitness: " + str(es_hyperneat_large_one_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT large one solves mountain_car in " + str(es_hyperneat_large_one_solved) + " out of " + str(runs) + " runs.\n")
thefile.write("ES-HyperNEAT large ten\n")
for item in es_hyperneat_large_run_ten_fitnesses:
    thefile.write("%s\n" % item)
if max_fit in es_hyperneat_large_ten_average_fit:
    thefile.write("ES-HyperNEAT large ten solves mountain_car at generation: " + str(es_hyperneat_large_ten_average_fit.index(max_fit)))
else:
    thefile.write("ES-HyperNEAT large ten does not solve mountain_car with best fitness: " + str(es_hyperneat_large_ten_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT large ten solves mountain_car in " + str(es_hyperneat_large_ten_solved) + " out of " + str(runs) + " runs.\n")

# Plot one fitnesses.
plt.plot(range(gens), neat_one_average_fit, 'r-', label="NEAT")
plt.plot(range(gens), hyperneat_one_average_fit, 'g--', label="HyperNEAT")
plt.plot(range(gens), es_hyperneat_small_one_average_fit, 'b-.', label="ES-HyperNEAT small")
plt.plot(range(gens), es_hyperneat_medium_one_average_fit, 'c-.', label="ES-HyperNEAT medium")
plt.plot(range(gens), es_hyperneat_large_one_average_fit, 'm-.', label="ES-HyperNEAT large")

plt.title("Average mountain_car fitnesses one episode")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.grid()
plt.legend(loc="best")

plt.savefig('mountain_car_one_fitnesses.svg')

plt.close()

# Plot ten fitnesses.
plt.plot(range(gens), neat_ten_average_fit, 'r-', label="NEAT")
plt.plot(range(gens), hyperneat_ten_average_fit, 'g--', label="HyperNEAT")
plt.plot(range(gens), es_hyperneat_small_ten_average_fit, 'b-.', label="ES-HyperNEAT small")
plt.plot(range(gens), es_hyperneat_medium_ten_average_fit, 'c-.', label="ES-HyperNEAT medium")
plt.plot(range(gens), es_hyperneat_large_ten_average_fit, 'm-.', label="ES-HyperNEAT large")

plt.title("Average mountain_car fitnesses ten episodes")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.grid()
plt.legend(loc="best")

plt.savefig('mountain_car_ten_fitnesses.svg')

plt.close()

