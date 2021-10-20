"""
Runs ALL pole balancing tasks using ES-HyperNEAT, HyperNEAT and NEAT.
Reports everything to text files.
"""

from multiprocessing import Manager
from itertools import repeat
import multiprocessing as multi
import gym
import matplotlib.pyplot as plt
import matplotlib
import es_hyperneat_pole_balancing
import hyperneat_pole_balancing
import neat_pole_balancing
matplotlib.use('Agg')


def run(number, gens, env, neat_stats, hyperneat_stats,
        es_hyperneat_small_stats, es_hyperneat_medium_stats, es_hyperneat_large_stats):
    """
    Run the experiments.
    """
    print(f"This is run #{str(number)}")
    neat_stats.append(neat_pole_balancing.run(gens, env)[1])
    hyperneat_stats.append(hyperneat_pole_balancing.run(gens, env)[1])
    es_hyperneat_small_stats.append(
        es_hyperneat_pole_balancing.run(gens, env, "S")[1])
    es_hyperneat_medium_stats.append(
        es_hyperneat_pole_balancing.run(gens, env, "M")[1])
    es_hyperneat_large_stats.append(
        es_hyperneat_pole_balancing.run(gens, env, "L")[1])


if __name__ == '__main__':
    # Initialize lists to keep track during run.
    MANAGER = Manager()

    NEAT_STATS, HYPERNEAT_STATS, ES_HYPERNEAT_SMALL_STATS = MANAGER.list(
        []), MANAGER.list([]), MANAGER.list([])
    ES_HYPERNEAT_MEDIUM_STATS, ES_HYPERNEAT_LARGE_STATS = MANAGER.list(
        []), MANAGER.list([])
    NEAT_RUN_ONE_FITNESS, HYPERNEAT_RUN_ONE_FITNESSES = [], []
    ES_HYPERNEAT_SMALL_RUN_ONE_FITNESSES, ES_HYPERNEAT_MEDIUM_RUN_ONE_FITNESSES = [], []
    ES_HYPERNEAT_LARGE_RUN_ONE_FITNESSES = []

    NEAT_RUN_TEN_FITNESSES, HYPERNEAT_RUN_TEN_FITNESSES = [], []
    ES_HYPERNEAT_SMALL_RUN_TEN_FITNESSES, ES_HYPERNEAT_MEDIUM_RUN_TEN_FITNESSES = [], []
    ES_HYPERNEAT_LARGE_RUN_TEN_FITNESSES = []

    NEAT_RUN_HUNDRED_FITNESSES, HYPERNEAT_RUN_HUNDRED_FITNESSES = [], []
    ES_HYPERNEAT_SMALL_RUN_HUNDRED_FITNESSES, ES_HYPERNEAT_MEDIUM_RUN_HUNDRED_FITNESSES = [], []
    ES_HYPERNEAT_LARGE_RUN_HUNDRED_FITNESSES = []

    NEAT_ONE_SOLVED, HYPERNEAT_ONE_SOLVED, ES_HYPERNEAT_SMALL_ONE_SOLVED = 0, 0, 0
    ES_HYPERNEAT_MEDIUM_ONE_SOLVED, ES_HYPERNEAT_LARGE_ONE_SOLVED = 0, 0

    NEAT_TEN_SOLVED, HYPERNEAT_TEN_SOLVED, ES_HYPERNEAT_SMALL_TEN_SOLVED = 0, 0, 0
    ES_HYPERNEAT_MEDIUM_TEN_SOLVED, ES_HYPERNEAT_LARGE_TEN_SOLVED = 0, 0

    NEAT_HUNDRED_SOLVED, HYPERNEAT_HUNDRED_SOLVED, ES_HYPERNEAT_SMALL_HUNDRED_SOLVED = 0, 0, 0
    ES_HYPERNEAT_MEDIUM_HUNDRED_SOLVED, ES_HYPERNEAT_LARGE_HUNDRED_SOLVED = 0, 0

    RUNS = 16
    INPUTS = range(RUNS)
    GENS = 50
    FIT_THRESHOLD = 475
    MAX_FIT = 475
    ENV = gym.make("CartPole-v1")

    P = multi.Pool(multi.cpu_count())
    P.starmap(run, zip(range(RUNS), repeat(GENS), repeat(ENV), repeat(NEAT_STATS),
                       repeat(HYPERNEAT_STATS), repeat(
                           ES_HYPERNEAT_SMALL_STATS), repeat(ES_HYPERNEAT_MEDIUM_STATS),
                       repeat(ES_HYPERNEAT_LARGE_STATS)))

    # Average the NEAT runs.
    TEMP_FIT_ONE = [0.0] * GENS
    TEMP_FIT_TEN = [0.0] * GENS
    TEMP_FIT_HUNDRED = [0.0] * GENS

    for (stat_one, stat_ten, stat_hundred) in NEAT_STATS:
        if stat_one.best_genome().fitness > MAX_FIT:
            NEAT_RUN_ONE_FITNESS.append(MAX_FIT)
        else:
            NEAT_RUN_ONE_FITNESS.append(stat_one.best_genome().fitness)
        if stat_ten.best_genome().fitness > MAX_FIT:
            NEAT_RUN_TEN_FITNESSES.append(MAX_FIT)
        else:
            NEAT_RUN_TEN_FITNESSES.append(stat_one.best_genome().fitness)
        if stat_hundred.best_genome().fitness > MAX_FIT:
            NEAT_RUN_HUNDRED_FITNESSES.append(MAX_FIT)
        else:
            NEAT_RUN_HUNDRED_FITNESSES.append(stat_one.best_genome().fitness)

        if stat_one.best_genome().fitness >= FIT_THRESHOLD:
            NEAT_ONE_SOLVED += 1
        if stat_ten.best_genome().fitness >= FIT_THRESHOLD:
            NEAT_TEN_SOLVED += 1
        if stat_hundred.best_genome().fitness >= FIT_THRESHOLD:
            NEAT_HUNDRED_SOLVED += 1

        for i in range(GENS):
            if i < len(stat_one.most_fit_genomes):
                if stat_one.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_ONE[i] += MAX_FIT
                else:
                    TEMP_FIT_ONE[i] += stat_one.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_ONE[i] += MAX_FIT
            if i < len(stat_ten.most_fit_genomes):
                if stat_ten.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_TEN[i] += MAX_FIT
                else:
                    TEMP_FIT_TEN[i] += stat_ten.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_TEN[i] += MAX_FIT
            if i < len(stat_hundred.most_fit_genomes):
                if stat_hundred.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_HUNDRED[i] += MAX_FIT
                else:
                    TEMP_FIT_HUNDRED[i] += stat_hundred.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_HUNDRED[i] += MAX_FIT

    NEAT_ONE_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_ONE]
    NEAT_TEN_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_TEN]
    NEAT_HUNDRED_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_HUNDRED]

    # Average the HyperNEAT runs.
    TEMP_FIT_ONE = [0.0] * GENS
    TEMP_FIT_TEN = [0.0] * GENS
    TEMP_FIT_HUNDRED = [0.0] * GENS

    for (stat_one, stat_ten, stat_hundred) in HYPERNEAT_STATS:
        if stat_one.best_genome().fitness > MAX_FIT:
            HYPERNEAT_RUN_ONE_FITNESSES.append(MAX_FIT)
        else:
            HYPERNEAT_RUN_ONE_FITNESSES.append(stat_one.best_genome().fitness)
        if stat_ten.best_genome().fitness > MAX_FIT:
            HYPERNEAT_RUN_TEN_FITNESSES.append(MAX_FIT)
        else:
            HYPERNEAT_RUN_TEN_FITNESSES.append(stat_one.best_genome().fitness)
        if stat_hundred.best_genome().fitness > MAX_FIT:
            HYPERNEAT_RUN_HUNDRED_FITNESSES.append(MAX_FIT)
        else:
            HYPERNEAT_RUN_HUNDRED_FITNESSES.append(
                stat_one.best_genome().fitness)

        if stat_one.best_genome().fitness >= FIT_THRESHOLD:
            HYPERNEAT_ONE_SOLVED += 1
        if stat_ten.best_genome().fitness >= FIT_THRESHOLD:
            HYPERNEAT_TEN_SOLVED += 1
        if stat_hundred.best_genome().fitness >= FIT_THRESHOLD:
            HYPERNEAT_HUNDRED_SOLVED += 1

        for i in range(GENS):
            if i < len(stat_one.most_fit_genomes):
                if stat_one.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_ONE[i] += MAX_FIT
                else:
                    TEMP_FIT_ONE[i] += stat_one.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_ONE[i] += MAX_FIT
            if i < len(stat_ten.most_fit_genomes):
                if stat_ten.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_TEN[i] += MAX_FIT
                else:
                    TEMP_FIT_TEN[i] += stat_ten.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_TEN[i] += MAX_FIT
            if i < len(stat_hundred.most_fit_genomes):
                if stat_hundred.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_HUNDRED[i] += MAX_FIT
                else:
                    TEMP_FIT_HUNDRED[i] += stat_hundred.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_HUNDRED[i] += MAX_FIT

    HYPERNEAT_ONE_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_ONE]
    HYPERNEAT_TEN_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_TEN]
    HYPERNEAT_HUNDRED_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_HUNDRED]

    # Average the small ES-HyperNEAT runs.
    TEMP_FIT_ONE = [0.0] * GENS
    TEMP_FIT_TEN = [0.0] * GENS
    TEMP_FIT_HUNDRED = [0.0] * GENS

    for (stat_one, stat_ten, stat_hundred) in ES_HYPERNEAT_SMALL_STATS:
        if stat_one.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_SMALL_RUN_ONE_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_SMALL_RUN_ONE_FITNESSES.append(
                stat_one.best_genome().fitness)
        if stat_ten.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_SMALL_RUN_TEN_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_SMALL_RUN_TEN_FITNESSES.append(
                stat_one.best_genome().fitness)
        if stat_hundred.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_SMALL_RUN_HUNDRED_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_SMALL_RUN_HUNDRED_FITNESSES.append(
                stat_one.best_genome().fitness)

        if stat_one.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_SMALL_ONE_SOLVED += 1
        if stat_ten.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_SMALL_TEN_SOLVED += 1
        if stat_hundred.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_SMALL_HUNDRED_SOLVED += 1

        for i in range(GENS):
            if i < len(stat_one.most_fit_genomes):
                if stat_one.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_ONE[i] += MAX_FIT
                else:
                    TEMP_FIT_ONE[i] += stat_one.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_ONE[i] += MAX_FIT
            if i < len(stat_ten.most_fit_genomes):
                if stat_ten.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_TEN[i] += MAX_FIT
                else:
                    TEMP_FIT_TEN[i] += stat_ten.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_TEN[i] += MAX_FIT
            if i < len(stat_hundred.most_fit_genomes):
                if stat_hundred.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_HUNDRED[i] += MAX_FIT
                else:
                    TEMP_FIT_HUNDRED[i] += stat_hundred.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_HUNDRED[i] += MAX_FIT

    ES_HYPERNEAT_SMALL_ONE_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_ONE]
    ES_HYPERNEAT_SMALL_TEN_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_TEN]
    ES_HYPERNEAT_SMALL_HUNDRED_AVERAGE_FIT = [
        x / RUNS for x in TEMP_FIT_HUNDRED]

    # Average the medium ES-HyperNEAT runs.
    TEMP_FIT_ONE = [0.0] * GENS
    TEMP_FIT_TEN = [0.0] * GENS
    TEMP_FIT_HUNDRED = [0.0] * GENS

    for (stat_one, stat_ten, stat_hundred) in ES_HYPERNEAT_MEDIUM_STATS:
        if stat_one.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_MEDIUM_RUN_ONE_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_MEDIUM_RUN_ONE_FITNESSES.append(
                stat_one.best_genome().fitness)
        if stat_ten.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_MEDIUM_RUN_TEN_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_MEDIUM_RUN_TEN_FITNESSES.append(
                stat_one.best_genome().fitness)
        if stat_hundred.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_MEDIUM_RUN_HUNDRED_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_MEDIUM_RUN_HUNDRED_FITNESSES.append(
                stat_one.best_genome().fitness)

        if stat_one.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_MEDIUM_ONE_SOLVED += 1
        if stat_ten.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_MEDIUM_TEN_SOLVED += 1
        if stat_hundred.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_MEDIUM_HUNDRED_SOLVED += 1

        for i in range(GENS):
            if i < len(stat_one.most_fit_genomes):
                if stat_one.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_ONE[i] += MAX_FIT
                else:
                    TEMP_FIT_ONE[i] += stat_one.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_ONE[i] += MAX_FIT
            if i < len(stat_ten.most_fit_genomes):
                if stat_ten.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_TEN[i] += MAX_FIT
                else:
                    TEMP_FIT_TEN[i] += stat_ten.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_TEN[i] += MAX_FIT
            if i < len(stat_hundred.most_fit_genomes):
                if stat_hundred.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_HUNDRED[i] += MAX_FIT
                else:
                    TEMP_FIT_HUNDRED[i] += stat_hundred.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_HUNDRED[i] += MAX_FIT

    ES_HYPERNEAT_MEDIUM_ONE_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_ONE]
    ES_HYPERNEAT_MEDIUM_TEN_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_TEN]
    ES_HYPERNEAT_MEDIUM_HUNDRED_AVERAGE_FIT = [
        x / RUNS for x in TEMP_FIT_HUNDRED]

    # Average the large ES-HyperNEAT runs.
    TEMP_FIT_ONE = [0.0] * GENS
    TEMP_FIT_TEN = [0.0] * GENS
    TEMP_FIT_HUNDRED = [0.0] * GENS

    for (stat_one, stat_ten, stat_hundred) in ES_HYPERNEAT_LARGE_STATS:
        if stat_one.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_LARGE_RUN_ONE_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_LARGE_RUN_ONE_FITNESSES.append(
                stat_one.best_genome().fitness)
        if stat_ten.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_LARGE_RUN_TEN_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_LARGE_RUN_TEN_FITNESSES.append(
                stat_one.best_genome().fitness)
        if stat_hundred.best_genome().fitness > MAX_FIT:
            ES_HYPERNEAT_LARGE_RUN_HUNDRED_FITNESSES.append(MAX_FIT)
        else:
            ES_HYPERNEAT_LARGE_RUN_HUNDRED_FITNESSES.append(
                stat_one.best_genome().fitness)

        if stat_one.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_LARGE_ONE_SOLVED += 1
        if stat_ten.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_LARGE_TEN_SOLVED += 1
        if stat_hundred.best_genome().fitness >= FIT_THRESHOLD:
            ES_HYPERNEAT_LARGE_HUNDRED_SOLVED += 1

        for i in range(GENS):
            if i < len(stat_one.most_fit_genomes):
                if stat_one.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_ONE[i] += MAX_FIT
                else:
                    TEMP_FIT_ONE[i] += stat_one.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_ONE[i] += MAX_FIT
            if i < len(stat_ten.most_fit_genomes):
                if stat_ten.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_TEN[i] += MAX_FIT
                else:
                    TEMP_FIT_TEN[i] += stat_ten.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_TEN[i] += MAX_FIT
            if i < len(stat_hundred.most_fit_genomes):
                if stat_hundred.most_fit_genomes[i].fitness > MAX_FIT:
                    TEMP_FIT_HUNDRED[i] += MAX_FIT
                else:
                    TEMP_FIT_HUNDRED[i] += stat_hundred.most_fit_genomes[i].fitness
            else:
                TEMP_FIT_HUNDRED[i] += MAX_FIT

    ES_HYPERNEAT_LARGE_ONE_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_ONE]
    ES_HYPERNEAT_LARGE_TEN_AVERAGE_FIT = [x / RUNS for x in TEMP_FIT_TEN]
    ES_HYPERNEAT_LARGE_HUNDRED_AVERAGE_FIT = [
        x / RUNS for x in TEMP_FIT_HUNDRED]

    # Write fitnesses to files.
    # NEAT.
    THEFILE = open('neat_pole_balancing_run_fitnesses.txt', 'w+')
    THEFILE.write("NEAT one\n")

    for item in NEAT_RUN_ONE_FITNESS:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in NEAT_ONE_AVERAGE_FIT:
        THEFILE.write("NEAT one solves pole_balancing at generation: " +
                      str(NEAT_ONE_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("NEAT one does not solve pole_balancing with best fitness: " +
                      str(NEAT_ONE_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nNEAT one solves pole_balancing in " +
                  str(NEAT_ONE_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("NEAT ten\n")

    for item in NEAT_RUN_TEN_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in NEAT_TEN_AVERAGE_FIT:
        THEFILE.write("NEAT ten solves pole_balancing at generation: " +
                      str(NEAT_TEN_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("NEAT ten does not solve pole_balancing with best fitness: " +
                      str(NEAT_TEN_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nNEAT ten solves pole_balancing in " +
                  str(NEAT_TEN_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("NEAT hundred\n")

    for item in NEAT_RUN_HUNDRED_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in NEAT_HUNDRED_AVERAGE_FIT:
        THEFILE.write("NEAT hundred solves pole_balancing at generation: " +
                      str(NEAT_HUNDRED_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("NEAT hundred does not solve pole_balancing with best fitness: " +
                      str(NEAT_HUNDRED_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nNEAT hundred solves pole_balancing in " +
                  str(NEAT_HUNDRED_SOLVED) + " out of " + str(RUNS) + " runs.\n")

    # HyperNEAT.
    THEFILE = open('hyperneat_pole_balancing_run_fitnesses.txt', 'w+')
    THEFILE.write("HyperNEAT one\n")

    for item in HYPERNEAT_RUN_ONE_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in HYPERNEAT_ONE_AVERAGE_FIT:
        THEFILE.write("HyperNEAT one solves pole_balancing at generation: " +
                      str(HYPERNEAT_ONE_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("HyperNEAT one does not solve pole_balancing with best fitness: " +
                      str(HYPERNEAT_ONE_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nHyperNEAT one solves pole_balancing in " +
                  str(HYPERNEAT_ONE_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("HyperNEAT ten\n")

    for item in HYPERNEAT_RUN_TEN_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in HYPERNEAT_TEN_AVERAGE_FIT:
        THEFILE.write("HyperNEAT ten solves pole_balancing at generation: " +
                      str(HYPERNEAT_TEN_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("HyperNEAT ten does not solve pole_balancing with best fitness: " +
                      str(HYPERNEAT_TEN_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nHyperNEAT ten solves pole_balancing in " +
                  str(HYPERNEAT_TEN_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("HyperNEAT hundred\n")

    for item in HYPERNEAT_RUN_HUNDRED_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in HYPERNEAT_HUNDRED_AVERAGE_FIT:
        THEFILE.write("HyperNEAT hundred solves pole_balancing at generation: " +
                      str(HYPERNEAT_HUNDRED_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("HyperNEAT hundred does not solve pole_balancing with best fitness: " +
                      str(HYPERNEAT_HUNDRED_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nHyperNEAT hundred solves pole_balancing in " +
                  str(HYPERNEAT_HUNDRED_SOLVED) + " out of " + str(RUNS) + " runs.\n")

    # ES-HyperNEAT small.
    THEFILE = open('es_hyperneat_pole_balancing_small_run_fitnesses.txt', 'w+')
    THEFILE.write("ES-HyperNEAT small one\n")

    for item in ES_HYPERNEAT_SMALL_RUN_ONE_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_SMALL_ONE_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT small one solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_SMALL_ONE_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("ES-HyperNEAT small one does not solve pole_balancing with best fitness: " +
                      str(ES_HYPERNEAT_SMALL_ONE_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT small one solves pole_balancing in " +
                  str(ES_HYPERNEAT_SMALL_ONE_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("ES-HyperNEAT small ten\n")

    for item in ES_HYPERNEAT_SMALL_RUN_TEN_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_SMALL_TEN_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT small ten solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_SMALL_TEN_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("ES-HyperNEAT small ten does not solve pole_balancing with best fitness: " +
                      str(ES_HYPERNEAT_SMALL_TEN_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT small ten solves pole_balancing in " +
                  str(ES_HYPERNEAT_SMALL_TEN_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("ES-HyperNEAT small hundred\n")

    for item in ES_HYPERNEAT_SMALL_RUN_HUNDRED_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_SMALL_HUNDRED_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT small hundred solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_SMALL_HUNDRED_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write(
            "ES-HyperNEAT small hundred does not solve pole_balancing with best fitness: " +
            str(ES_HYPERNEAT_SMALL_HUNDRED_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT small hundred solves pole_balancing in " +
                  str(ES_HYPERNEAT_SMALL_HUNDRED_SOLVED) + " out of " + str(RUNS) + " runs.\n")

    # ES-HyperNEAT medium.
    THEFILE = open(
        'es_hyperneat_pole_balancing_medium_run_fitnesses.txt', 'w+')
    THEFILE.write("ES-HyperNEAT medium one\n")

    for item in ES_HYPERNEAT_MEDIUM_RUN_ONE_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_MEDIUM_ONE_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT medium one solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_MEDIUM_ONE_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("ES-HyperNEAT medium one does not solve pole_balancing with best fitness: " +
                      str(ES_HYPERNEAT_MEDIUM_ONE_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT medium one solves pole_balancing in " +
                  str(ES_HYPERNEAT_MEDIUM_ONE_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("ES-HyperNEAT medium ten\n")

    for item in ES_HYPERNEAT_MEDIUM_RUN_TEN_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_MEDIUM_TEN_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT medium ten solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_MEDIUM_TEN_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("ES-HyperNEAT medium ten does not solve pole_balancing with best fitness: " +
                      str(ES_HYPERNEAT_MEDIUM_TEN_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT medium ten solves pole_balancing in " +
                  str(ES_HYPERNEAT_MEDIUM_TEN_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("ES-HyperNEAT medium hundred\n")

    for item in ES_HYPERNEAT_MEDIUM_RUN_HUNDRED_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_MEDIUM_HUNDRED_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT medium hundred solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_MEDIUM_HUNDRED_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write(
            "ES-HyperNEAT medium hundred does not solve pole_balancing with best fitness: " +
            str(ES_HYPERNEAT_MEDIUM_HUNDRED_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT medium hundred solves pole_balancing in " +
                  str(ES_HYPERNEAT_MEDIUM_HUNDRED_SOLVED) + " out of " + str(RUNS) + " runs.\n")

    # ES-HyperNEAT large.
    THEFILE = open('es_hyperneat_pole_balancing_large_run_fitnesses.txt', 'w+')
    THEFILE.write("ES-HyperNEAT large one\n")

    for item in ES_HYPERNEAT_LARGE_RUN_ONE_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_LARGE_ONE_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT large one solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_LARGE_ONE_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("ES-HyperNEAT large one does not solve pole_balancing with best fitness: " +
                      str(ES_HYPERNEAT_LARGE_ONE_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT large one solves pole_balancing in " +
                  str(ES_HYPERNEAT_LARGE_ONE_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("ES-HyperNEAT large ten\n")

    for item in ES_HYPERNEAT_LARGE_RUN_TEN_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_LARGE_TEN_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT large ten solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_LARGE_TEN_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write("ES-HyperNEAT large ten does not solve pole_balancing with best fitness: " +
                      str(ES_HYPERNEAT_LARGE_TEN_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT large ten solves pole_balancing in " +
                  str(ES_HYPERNEAT_LARGE_TEN_SOLVED) + " out of " + str(RUNS) + " runs.\n")
    THEFILE.write("ES-HyperNEAT large hundred\n")

    for item in ES_HYPERNEAT_LARGE_RUN_HUNDRED_FITNESSES:
        THEFILE.write("%s\n" % item)

    if MAX_FIT in ES_HYPERNEAT_LARGE_HUNDRED_AVERAGE_FIT:
        THEFILE.write("ES-HyperNEAT large hundred solves pole_balancing at generation: " +
                      str(ES_HYPERNEAT_LARGE_HUNDRED_AVERAGE_FIT.index(MAX_FIT)))
    else:
        THEFILE.write(
            "ES-HyperNEAT large hundred does not solve pole_balancing with best fitness: " +
            str(ES_HYPERNEAT_LARGE_HUNDRED_AVERAGE_FIT[GENS-1]))
    THEFILE.write("\nES-HyperNEAT large hundred solves pole_balancing in " +
                  str(ES_HYPERNEAT_LARGE_HUNDRED_SOLVED) + " out of " + str(RUNS) + " runs.\n")

    # Plot one fitnesses.
    plt.plot(range(GENS), NEAT_ONE_AVERAGE_FIT, 'r-', label="NEAT")
    plt.plot(range(GENS), HYPERNEAT_ONE_AVERAGE_FIT, 'g--', label="HyperNEAT")
    plt.plot(range(GENS), ES_HYPERNEAT_SMALL_ONE_AVERAGE_FIT,
             'b-.', label="ES-HyperNEAT small")
    plt.plot(range(GENS), ES_HYPERNEAT_MEDIUM_ONE_AVERAGE_FIT,
             'c-.', label="ES-HyperNEAT medium")
    plt.plot(range(GENS), ES_HYPERNEAT_LARGE_ONE_AVERAGE_FIT,
             'm-.', label="ES-HyperNEAT large")

    plt.title("Average pole_balancing fitnesses one episode")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig('pole_balancing_one_fitnesses.svg')

    plt.close()

    # Plot ten fitnesses.
    plt.plot(range(GENS), NEAT_TEN_AVERAGE_FIT, 'r-', label="NEAT")
    plt.plot(range(GENS), HYPERNEAT_TEN_AVERAGE_FIT, 'g--', label="HyperNEAT")
    plt.plot(range(GENS), ES_HYPERNEAT_SMALL_TEN_AVERAGE_FIT,
             'b-.', label="ES-HyperNEAT small")
    plt.plot(range(GENS), ES_HYPERNEAT_MEDIUM_TEN_AVERAGE_FIT,
             'c-.', label="ES-HyperNEAT medium")
    plt.plot(range(GENS), ES_HYPERNEAT_LARGE_TEN_AVERAGE_FIT,
             'm-.', label="ES-HyperNEAT large")

    plt.title("Average pole_balancing fitnesses ten episodes")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig('pole_balancing_ten_fitnesses.svg')

    plt.close()

    # Plot hundred fitnesses.
    plt.plot(range(GENS), NEAT_HUNDRED_AVERAGE_FIT, 'r-', label="NEAT")
    plt.plot(range(GENS), HYPERNEAT_HUNDRED_AVERAGE_FIT,
             'g--', label="HyperNEAT")
    plt.plot(range(GENS), ES_HYPERNEAT_SMALL_HUNDRED_AVERAGE_FIT,
             'b-.', label="ES-HyperNEAT small")
    plt.plot(range(GENS), ES_HYPERNEAT_MEDIUM_HUNDRED_AVERAGE_FIT,
             'c-.', label="ES-HyperNEAT medium")
    plt.plot(range(GENS), ES_HYPERNEAT_LARGE_HUNDRED_AVERAGE_FIT,
             'm-.', label="ES-HyperNEAT large")

    plt.title("Average pole_balancing fitnesses hundred episodes")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")

    plt.savefig('pole_balancing_hundred_fitnesses.svg')

    plt.close()
