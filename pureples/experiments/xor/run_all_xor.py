"""
Runs ALL XOR tasks using ES-HyperNEAT, HyperNEAT and NEAT.
Reports everything to text files.
"""


from multiprocessing import Manager
import multiprocessing as multi
from itertools import repeat
import matplotlib.pyplot as plt
import matplotlib
import es_hyperneat_xor
import hyperneat_xor
import neat_xor
matplotlib.use('Agg')


def run(number, gens, neat_stats, hyperneat_stats, es_hyperneat_small_stats,
        es_hyperneat_medium_stats, es_hyperneat_large_stats):
    """
    Run the experiments.
    """
    print(f"This is run #{str(number)}")
    neat_stats.append(neat_xor.run(gens)[1])
    hyperneat_stats.append(hyperneat_xor.run(gens)[1])
    es_hyperneat_small_stats.append(es_hyperneat_xor.run(gens, "S")[1])
    es_hyperneat_medium_stats.append(es_hyperneat_xor.run(gens, "M")[1])
    es_hyperneat_large_stats.append(es_hyperneat_xor.run(gens, "L")[1])


if __name__ == '__main__':
    MANAGER = Manager()

    NEAT_STATS, HYPERNEAT_STATS, ES_HYPERNEAT_SMALL_STATS = MANAGER.list(
        []), MANAGER.list([]), MANAGER.list([])
    ES_HYPERNEAT_MEDIUM_STATS, ES_HYPERNEAT_LARGE_STATS = MANAGER.list(
        []), MANAGER.list([])
    NEAT_RUN_FITNESS, HYPERNEAT_RUN_FITNESSES, ES_HYPERNEAT_SMALL_RUN_FITNESSES = [], [], []
    ES_HYPERNEAT_MEDIUM_RUN_FITNESSES, ES_HYPERNEAT_LARGE_RUN_FITNESSES = [], []
    NEAT_SOLVED, HYPERNEAT_SOLVED, ES_HYPERNEAT_SMALL_SOLVED = 0, 0, 0
    ES_HYPERNEAT_MEDIUM_SOLVED, ES_HYPERNEAT_LARGE_SOLVED = 0, 0
    RUNS = 20
    INPUTS = range(RUNS)
    GENS = 300
    FIT_THRESHOLD = 0.975
    MAX_FIT = 1.0

    P = multi.Pool(multi.cpu_count())
    P.starmap(run, zip(range(RUNS), repeat(GENS), repeat(NEAT_STATS),
                       repeat(HYPERNEAT_STATS), repeat(
                           ES_HYPERNEAT_SMALL_STATS), repeat(ES_HYPERNEAT_MEDIUM_STATS),
                       repeat(ES_HYPERNEAT_LARGE_STATS)))

    # Average the NEAT runs.
    TEMP_FIT = [0.0] * GENS

    for stat in NEAT_STATS:
        NEAT_RUN_FITNESS.append(stat.best_genome().fitness)
        if stat.best_genome().fitness >= FIT_THRESHOLD:
            NEAT_SOLVED += 1

        for i in range(GENS):
            if i < len(stat.most_fit_genomes):
                TEMP_FIT[i] += stat.most_fit_genomes[i].fitness
            else:
                TEMP_FIT[i] += MAX_FIT

    NEAT_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT]

    # Average the HyperNEAT runs.
    TEMP_FIT = [0.0] * GENS

    for stat in HYPERNEAT_STATS:
        HYPERNEAT_RUN_FITNESSES.append(stat.best_genome().fitness)
        if stat.best_genome().fitness >= FIT_THRESHOLD:
            HYPERNEAT_SOLVED += 1

        for i in range(GENS):
            if i < len(stat.most_fit_genomes):
                TEMP_FIT[i] += stat.most_fit_genomes[i].fitness
            else:
                TEMP_FIT[i] += MAX_FIT

    HYPERNEAY_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT]

    # Average the small ES-HyperNEAT runs.
    TEMP_FIT = [0.0] * GENS

    for stat in ES_HYPERNEAT_SMALL_STATS:
        ES_HYPERNEAT_SMALL_RUN_FITNESSES.append(stat.best_genome().fitness)
        if stat.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_SMALL_SOLVED += 1

        for i in range(GENS):
            if i < len(stat.most_fit_genomes):
                TEMP_FIT[i] += stat.most_fit_genomes[i].fitness
            else:
                TEMP_FIT[i] += MAX_FIT

    ES_HYPERNEAT_SMALL_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT]

    # Average the medium ES-HyperNEAT runs.
    TEMP_FIT = [0.0] * GENS

    for stat in ES_HYPERNEAT_MEDIUM_STATS:
        ES_HYPERNEAT_MEDIUM_RUN_FITNESSES.append(stat.best_genome().fitness)
        if stat.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_MEDIUM_SOLVED += 1

        for i in range(GENS):
            if i < len(stat.most_fit_genomes):
                TEMP_FIT[i] += stat.most_fit_genomes[i].fitness
            else:
                TEMP_FIT[i] += MAX_FIT

    ES_HYPERNEAT_MEDIUM_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT]

    # Average the large ES-HyperNEAT runs.
    TEMP_FIT = [0.0] * GENS

    for stat in ES_HYPERNEAT_LARGE_STATS:
        ES_HYPERNEAT_LARGE_RUN_FITNESSES.append(stat.best_genome().fitness)
        if stat.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_LARGE_SOLVED += 1

        for i in range(GENS):
            if i < len(stat.most_fit_genomes):
                TEMP_FIT[i] += stat.most_fit_genomes[i].fitness
            else:
                TEMP_FIT[i] += MAX_FIT

    ES_HYPERNEAT_LARGE_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT]

    # Write fitnesses to files.
    THEFILE = open('neat_xor_run_fitnesses.txt', 'w+')

    for item in NEAT_RUN_FITNESS:
        THEFILE.write("%s\n" % item)
    if 1.0 in NEAT_AVERAGE_FIT:
        THEFILE.write("NEAT solves XOR at generation: " +
                      str(NEAT_AVERAGE_FIT.index(1.0)-1))
    else:
        THEFILE.write("NEAT does not solve XOR with best fitness: " +
                      str(NEAT_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nNEAT solves XOR in " + str(NEAT_SOLVED) +
                  " out of " + str(RUNS) + " runs.")

    THEFILE = open('hyperneat_xor_run_fitnesses.txt', 'w+')

    for item in HYPERNEAT_RUN_FITNESSES:
        THEFILE.write("%s\n" % item)
    if 1.0 in HYPERNEAY_AVERAGE_FIT:
        THEFILE.write("HyperNEAT solves XOR at generation: " +
                      str(HYPERNEAY_AVERAGE_FIT.index(1.0)-1))
    else:
        THEFILE.write("HyperNEAT does not solve XOR with best fitness: " +
                      str(HYPERNEAY_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nHyperEAT solves XOR in " +
                  str(HYPERNEAT_SOLVED) + " out of " + str(RUNS) + " runs.")

    THEFILE = open('es_hyperneat_xor_small_run_fitnesses.txt', 'w+')

    for item in ES_HYPERNEAT_SMALL_RUN_FITNESSES:
        THEFILE.write("%s\n" % item)
    if 1.0 in ES_HYPERNEAT_SMALL_AVERAGE_FIT:
        THEFILE.write("ESHyperNEAT small solves XOR at generation: " +
                      str(ES_HYPERNEAT_SMALL_AVERAGE_FIT.index(1.0)-1))
    else:
        THEFILE.write("ES-HyperNEAT small does not solve XOR with best fitness: " +
                      str(ES_HYPERNEAT_SMALL_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT small solves XOR in " +
                  str(ES_HYPERNEAT_SMALL_SOLVED) + " out of " + str(RUNS) + " runs.")

    THEFILE = open('es_hyperneat_xor_medium_run_fitnesses.txt', 'w+')

    for item in ES_HYPERNEAT_MEDIUM_RUN_FITNESSES:
        THEFILE.write("%s\n" % item)
    if 1.0 in ES_HYPERNEAT_MEDIUM_AVERAGE_FIT:
        THEFILE.write("ESHyperNEAT medium solves XOR at generation: " +
                      str(ES_HYPERNEAT_MEDIUM_AVERAGE_FIT.index(1.0)-1))
    else:
        THEFILE.write("ES-HyperNEAT medium does not solve XOR with best fitness: " +
                      str(ES_HYPERNEAT_MEDIUM_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT medium solves XOR in " +
                  str(ES_HYPERNEAT_MEDIUM_SOLVED) + " out of " + str(RUNS) + " runs.")

    THEFILE = open('es_hyperneat_xor_large_run_fitnesses.txt', 'w+')

    for item in ES_HYPERNEAT_LARGE_RUN_FITNESSES:
        THEFILE.write("%s\n" % item)
    if 1.0 in ES_HYPERNEAT_LARGE_AVERAGE_FIT:
        THEFILE.write("ESHyperNEAT large solves XOR at generation: " +
                      str(ES_HYPERNEAT_LARGE_AVERAGE_FIT.index(1.0)-1))
    else:
        THEFILE.write("ES-HyperNEAT large does not solve XOR with best fitness: " +
                      str(ES_HYPERNEAT_LARGE_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT large solves XOR in " +
                  str(ES_HYPERNEAT_LARGE_SOLVED) + " out of " + str(RUNS) + " runs.")

    # Plot the fitnesses.
    plt.plot(range(GENS), NEAT_AVERAGE_FIT, 'r-', label="NEAT")
    plt.plot(range(GENS), HYPERNEAY_AVERAGE_FIT, 'g--', label="HyperNEAT")
    plt.plot(range(GENS), ES_HYPERNEAT_SMALL_AVERAGE_FIT,
             'b-.', label="ES-HyperNEAT small")
    plt.plot(range(GENS), ES_HYPERNEAT_MEDIUM_AVERAGE_FIT,
             'c-.', label="ES-HyperNEAT medium")
    plt.plot(range(GENS), ES_HYPERNEAT_LARGE_AVERAGE_FIT,
             'm-.', label="ES-HyperNEAT large")

    plt.title("Average XOR fitnesses")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig('xor_fitnesses.svg')

    plt.close()
