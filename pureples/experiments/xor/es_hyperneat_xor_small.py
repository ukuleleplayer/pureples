import neat 
import neat.nn
import cPickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# Network inputs and expected outputs.
xor_inputs  = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [    (0.0,),     (1.0,),     (1.0,),     (0.0,)]

input_coordinates  = [(-1.0, -1.0),(0.0, -1.0),(1.0, -1.0)]
output_coordinates = [(0.0, 1.0)]

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 0, 
          "max_depth": 1, 
          "variance_threshold": 0.03, 
          "band_threshold": 0.3, 
          "iteration_level": 1,
          "division_threshold": 0.5, 
          "max_weight": 5.0, 
          "activation": "sigmoid"}

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_xor')


def eval_fitness(genomes, config):
    
    for idx, g in genomes:

        cppn = neat.nn.FeedForwardNetwork.create(g, config)
        network = ESNetwork(sub, cppn, params)
        net = network.create_phenotype_network()
        
        sum_square_error = 0.0
        for inputs, expected in zip(xor_inputs, xor_outputs):

            new_input = inputs + (1.0,)
            net.reset()
            for i in range(network.activations):
                
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
    print("es_hyperneat_xor_small done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    winner = run(300)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    winner_net = network.create_phenotype_network(filename='es_hyperneat_xor_small_winner.png')  # This will also draw winner_net.
    for inputs, expected in zip(xor_inputs, xor_outputs):
        new_input = inputs + (1.0,)
        winner_net.reset()
        for i in range(network.activations):
            output = winner_net.activate(new_input)
        print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # Save CPPN if wished reused and draw it to file.
    draw_net(cppn, filename="es_hyperneat_xor_small_cppn")
    with open('es_hyperneat_xor_small_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

