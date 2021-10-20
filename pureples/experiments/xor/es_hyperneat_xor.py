"""
An experiment using a variable-sized ES-HyperNEAT network to perform the simple XOR task.
Fitness threshold set in config
- by default very high to show the high possible accuracy of this library.
"""

import pickle
import neat
import neat.nn
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network inputs and expected outputs.
XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
OUTPUT_COORDINATES = [(0.0, 1.0)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 5.0,
            "activation": "sigmoid"}


DYNAMIC_PARAMS = params(VERSION)

# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/xor/config_cppn_xor')


def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness, in this case, as the mean squared error.
    """
    for _, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(SUBSTRATE, cppn, DYNAMIC_PARAMS)
        net = network.create_phenotype_network()

        sum_square_error = 0.0

        for xor_inputs, xor_expected in zip(XOR_INPUTS, XOR_OUTPUTS):
            new_xor_input = xor_inputs + (1.0,)
            net.reset()

            for _ in range(network.activations):
                xor_output = net.activate(new_xor_input)

            sum_square_error += ((xor_output[0] - xor_expected[0])**2.0)/4.0

        genome.fitness = 1 - sum_square_error


def run(gens, version):
    """
    Create the population and run the XOR task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    global DYNAMIC_PARAMS
    DYNAMIC_PARAMS = params(version)

    winner = pop.run(eval_fitness, gens)
    print(f"es_hyperneat_xor_{VERSION_TEXT} done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    WINNER = run(300, VERSION)[0]  # Only relevant to look at the winner.
    print('\nBest genome:\n{!s}'.format(WINNER))

    # Verify network output against training data.
    print('\nOutput:')
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, DYNAMIC_PARAMS)
    # This will also draw winner_net.
    WINNER_NET = NETWORK.create_phenotype_network(
        filename=f'pureples/experiments/xor/es_hyperneat_xor_{VERSION_TEXT}_winner.png')

    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        new_input = inputs + (1.0,)
        WINNER_NET.reset()

        for i in range(NETWORK.activations):
            output = WINNER_NET.activate(new_input)

        print("  input {!r}, expected output {!r}, got {!r}".format(
            inputs, expected, output))

    # Save CPPN if wished reused and draw it to file.
    draw_net(
        CPPN, filename=f"pureples/experiments/xor/es_hyperneat_xor_{VERSION_TEXT}_cppn")
    with open(f'pureples/experiments/xor/es_hyperneat_xor_{VERSION_TEXT}_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
