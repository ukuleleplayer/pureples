import neat 
import neat.nn
import cPickle as pickle
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network inputs and expected outputs.
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [    (0.0,),     (1.0,),     (1.0,),     (0.0,)]

input_coordinates  = [(-1.0, -1.0),(0.0, -1.0),(1.0, -1.0)]
hidden_coordinates = [[(-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)]]
output_coordinates = [(0.0, 1.0)]
activations = len(hidden_coordinates) + 2

sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_xor')


def eval_fitness(genomes, config):
    
    for idx, g in genomes:

        cppn = neat.nn.FeedForwardNetwork.create(g, config)
        net = create_phenotype_network(cppn, sub)
        
        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):

            new_input = inputs + (1.0,)
            net.reset()
            for i in range(activations):
                
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
    print("hyperneat_xor done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    winner = run(300)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_net = create_phenotype_network(cppn, sub)
    for inputs, expected in zip(xor_inputs, xor_outputs):
        new_input = inputs + (1.0,)
        winner_net.reset()
        for i in range(activations):
            output = winner_net.activate(new_input)
        print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # Save CPPN if wished reused and draw it to file along with the winner.
    with open('hyperneat_xor_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
    draw_net(cppn, filename="hyperneat_xor_cppn")
    draw_net(winner_net, filename="hyperneat_xor_winner")
