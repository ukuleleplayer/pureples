import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neat_xor
import hyperneat_xor
import es_hyperneat_xor_small
import es_hyperneat_xor_medium
import es_hyperneat_xor_large
import multiprocessing as multi
from multiprocessing import Manager

manager = Manager()

neat_stats, hyperneat_stats, es_hyperneat_small_stats = manager.list([]), manager.list([]), manager.list([])
es_hyperneat_medium_stats, es_hyperneat_large_stats = manager.list([]), manager.list([])
neat_run_fitnesses, hyperneat_run_fitnesses, es_hyperneat_small_run_fitnesses, es_hyperneat_medium_run_fitnesses, es_hyperneat_large_run_fitnesses = [], [], [], [], []
neat_solved, hyperneat_solved, es_hyperneat_small_solved, es_hyperneat_medium_solved, es_hyperneat_large_solved = 0, 0, 0, 0, 0
runs = 20
inputs = range(runs)
gens = 300
fit_threshold = 0.975
max_fit = 1.0


# Run the experiments.
def run(i):
    print("This is run #" + str(i))
    neat_stats.append(neat_xor.run(gens)[1])
    hyperneat_stats.append(hyperneat_xor.run(gens)[1])
    es_hyperneat_small_stats.append(es_hyperneat_xor_small.run(gens)[1])
    es_hyperneat_medium_stats.append(es_hyperneat_xor_medium.run(gens)[1])
    es_hyperneat_large_stats.append(es_hyperneat_xor_large.run(gens)[1])    

p = multi.Pool(multi.cpu_count())
p.map(run,range(runs))

# Average the NEAT runs.
temp_fit = [0.0] * gens
for stat in neat_stats:
    neat_run_fitnesses.append(stat.best_genome().fitness)
    if stat.best_genome().fitness >= fit_threshold:
        neat_solved += 1
    for i in range(gens):
        if i < len(stat.most_fit_genomes):
            temp_fit[i] += stat.most_fit_genomes[i].fitness
        else:
            temp_fit[i] += max_fit

neat_average_fit = [x / runs for x in temp_fit]

# Average the HyperNEAT runs.
temp_fit = [0.0] * gens
for stat in hyperneat_stats:
    hyperneat_run_fitnesses.append(stat.best_genome().fitness)
    if stat.best_genome().fitness >= fit_threshold:
        hyperneat_solved += 1
    for i in range(gens):
        if i < len(stat.most_fit_genomes):
            temp_fit[i] += stat.most_fit_genomes[i].fitness
        else:
            temp_fit[i] += max_fit

hyperneat_average_fit = [x / runs for x in temp_fit]

# Average the small ES-HyperNEAT runs.
temp_fit = [0.0] * gens
for stat in es_hyperneat_small_stats:
    es_hyperneat_small_run_fitnesses.append(stat.best_genome().fitness)
    if stat.best_genome().fitness >= fit_threshold:
        es_hyperneat_small_solved += 1
    for i in range(gens):
        if i < len(stat.most_fit_genomes):
            temp_fit[i] += stat.most_fit_genomes[i].fitness
        else:
            temp_fit[i] += max_fit

es_hyperneat_small_average_fit = [x / runs for x in temp_fit]

# Average the medium ES-HyperNEAT runs.
temp_fit = [0.0] * gens
for stat in es_hyperneat_medium_stats:
    es_hyperneat_medium_run_fitnesses.append(stat.best_genome().fitness)
    if stat.best_genome().fitness >= fit_threshold:
        es_hyperneat_medium_solved += 1
    for i in range(gens):
        if i < len(stat.most_fit_genomes):
            temp_fit[i] += stat.most_fit_genomes[i].fitness
        else:
            temp_fit[i] += max_fit

es_hyperneat_medium_average_fit = [x / runs for x in temp_fit]

# Average the large ES-HyperNEAT runs.
temp_fit = [0.0] * gens
for stat in es_hyperneat_large_stats:
    es_hyperneat_large_run_fitnesses.append(stat.best_genome().fitness)
    if stat.best_genome().fitness >= fit_threshold:
        es_hyperneat_large_solved += 1
    for i in range(gens):
        if i < len(stat.most_fit_genomes):
            temp_fit[i] += stat.most_fit_genomes[i].fitness
        else:
            temp_fit[i] += max_fit

es_hyperneat_large_average_fit = [x / runs for x in temp_fit]

# Write fitnesses to files.
thefile = open('neat_xor_run_fitnesses.txt', 'w+')
for item in neat_run_fitnesses:
    thefile.write("%s\n" % item)
if 1.0 in neat_average_fit:
    thefile.write("NEAT solves XOR at generation: " + str(neat_average_fit.index(1.0)-1))
else:
    thefile.write("NEAT does not solve XOR with best fitness: " + str(neat_average_fit[gens-1]))
thefile.write("\nNEAT solves XOR in " + str(neat_solved) + " out of " + str(runs) + " runs.")

thefile = open('hyperneat_xor_run_fitnesses.txt', 'w+')
for item in hyperneat_run_fitnesses:
    thefile.write("%s\n" % item)
if 1.0 in hyperneat_average_fit:
    thefile.write("HyperNEAT solves XOR at generation: " + str(hyperneat_average_fit.index(1.0)-1))
else:
    thefile.write("HyperNEAT does not solve XOR with best fitness: " + str(hyperneat_average_fit[gens-1]))
thefile.write("\nHyperEAT solves XOR in " + str(hyperneat_solved) + " out of " + str(runs) + " runs.")

thefile = open('es_hyperneat_xor_small_run_fitnesses.txt', 'w+')
for item in es_hyperneat_small_run_fitnesses:
    thefile.write("%s\n" % item)
if 1.0 in es_hyperneat_small_average_fit:
    thefile.write("ESHyperNEAT small solves XOR at generation: " + str(es_hyperneat_small_average_fit.index(1.0)-1))
else:
    thefile.write("ES-HyperNEAT small does not solve XOR with best fitness: " + str(es_hyperneat_small_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT small solves XOR in " + str(es_hyperneat_small_solved) + " out of " + str(runs) + " runs.")

thefile = open('es_hyperneat_xor_medium_run_fitnesses.txt', 'w+')
for item in es_hyperneat_medium_run_fitnesses:
    thefile.write("%s\n" % item)
if 1.0 in es_hyperneat_medium_average_fit:
    thefile.write("ESHyperNEAT medium solves XOR at generation: " + str(es_hyperneat_medium_average_fit.index(1.0)-1))
else:
    thefile.write("ES-HyperNEAT medium does not solve XOR with best fitness: " + str(es_hyperneat_medium_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT medium solves XOR in " + str(es_hyperneat_medium_solved) + " out of " + str(runs) + " runs.")

thefile = open('es_hyperneat_xor_large_run_fitnesses.txt', 'w+')
for item in es_hyperneat_large_run_fitnesses:
    thefile.write("%s\n" % item)
if 1.0 in es_hyperneat_large_average_fit:
    thefile.write("ESHyperNEAT large solves XOR at generation: " + str(es_hyperneat_large_average_fit.index(1.0)-1))
else:
    thefile.write("ES-HyperNEAT large does not solve XOR with best fitness: " + str(es_hyperneat_large_average_fit[gens-1]))
thefile.write("\nES-HyperNEAT large solves XOR in " + str(es_hyperneat_large_solved) + " out of " + str(runs) + " runs.")

# Plot the fitnesses.
plt.plot(range(gens), neat_average_fit, 'r-', label="NEAT")
plt.plot(range(gens), hyperneat_average_fit, 'g--', label="HyperNEAT")
plt.plot(range(gens), es_hyperneat_small_average_fit, 'b-.', label="ES-HyperNEAT small")
plt.plot(range(gens), es_hyperneat_medium_average_fit, 'c-.', label="ES-HyperNEAT medium")
plt.plot(range(gens), es_hyperneat_large_average_fit, 'm-.', label="ES-HyperNEAT large")

plt.title("Average XOR fitnesses")
plt.xlabel("Generations")
plt.ylabel("Fitness")
plt.grid()
plt.legend(loc="best")

plt.savefig('xor_fitnesses.svg')

plt.close()
