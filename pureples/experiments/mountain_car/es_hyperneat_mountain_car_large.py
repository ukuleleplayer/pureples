import neat 
import logging
try:
   import cPickle as pickle
except:
   import pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_es
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# Network input and output coordinates.
input_coordinates = [(-0.33, -1.), (0.33, -1.)]
output_coordinates = [(-0.5, 1.), (0., 1.), (0.5, 1.)]

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 2,
          "max_depth": 3,
          "variance_threshold": 0.03,
          "band_threshold": 0.3,
          "iteration_level": 1,
          "division_threshold": 0.5,
          "max_weight": 8.0,
          "activation": "sigmoid"}

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_mountain_car')


def run(gens, env):
    winner, stats = run_es(gens, env, 200, config, params, sub, max_trials=0)
    print("es_hyperneat_mountain_car_large done") 
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make("MountainCar-v0")

    # Run!
    winner = run(200, env)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    net = network.create_phenotype_network(filename="es_hyperneat_mountain_car_large_winner")
    draw_net(cppn, filename="es_hyperneat_mountain_car_large_cppn")
    with open('es_hyperneat_mountain_car_large_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

