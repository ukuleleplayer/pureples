"""
An experiment using NEAT to perform the simple XOR task.
Fitness threshold set in config
- by default very high to show the high possible accuracy of the NEAT library.
"""

import pickle
import neat
import neat.nn
from pureples.shared.visualize import draw_net

# Network inputs and expected outputs.
XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]

# Config for FeedForwardNetwork.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/xor/config_neat_xor')


def eval_fitness(genomes, config):
    """
    Fitness function.
    For each genome evaluate its fitness, in this case, as the mean squared error.
    """
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        sum_square_error = 0.0

        for xor_inputs, xor_expected in zip(XOR_INPUTS, XOR_OUTPUTS):
            new_xor_input = xor_inputs + (1.0,)
            xor_output = net.activate(new_xor_input)
            sum_square_error += ((xor_output[0] - xor_expected[0])**2.0)/4.0

        genome.fitness = 1 - sum_square_error


def run(gens):
    """
    Create the population and run the XOR task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(eval_fitness, gens)
    print("neat_xor done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    WINNER = run(300)[0]  # Only relevant to look at the winner.
    print('\nBest genome:\n{!s}'.format(WINNER))

    # Verify network output against training data.
    print('\nOutput:')
    WINNER_NET = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)

    for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
        new_input = inputs + (1.0,)
        output = WINNER_NET.activate(new_input)
        print("  input {!r}, expected output {!r}, got {!r}".format(
            inputs, expected, output))

    # Save net if wished reused and draw it to a file.
    with open('pureples/experiments/xor/winner_neat_xor.pkl', 'wb') as output:
        pickle.dump(WINNER_NET, output, pickle.HIGHEST_PROTOCOL)
    draw_net(WINNER_NET, filename="pureples/experiments/xor/neat_xor_winner")
