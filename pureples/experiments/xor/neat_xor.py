import neat 
import neat.nn
import cPickle as pickle
import sys
import os.path
from pureples.shared.visualize import draw_net

# Network inputs and expected outputs.
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [    (0.0,),     (1.0,),     (1.0,),     (0.0,)]

# Config for FeedForwardNetwork.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_xor')


def eval_fitness(genomes, config):
    
    for idx, g in genomes:

        net = neat.nn.FeedForwardNetwork.create(g, config)
        
        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):

            new_input = inputs + (1.0,)
            output = net.activate(new_input)
            sum_square_error += ((output[0] - expected[0])**2.0)/4.0
 
        g.fitness = 1 - sum_square_error


# Create the population and run the XOR task by providing the above fitness function.
def run(gens):
    pop = neat.population.Population(config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(eval_fitness, gens)
    print("neat_xor done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    winner = run(300)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for inputs, expected in zip(xor_inputs, xor_outputs):
        new_input = inputs + (1.0,)
        output = winner_net.activate(new_input)
        print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # Save net if wished reused and draw it to a file.
    with open('winner_neat_xor.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)
    draw_net(winner_net, filename="neat_xor_winner")

