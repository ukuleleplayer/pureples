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
input_coordinates = []
for i in range(0,4):
    input_coordinates.append((-1. +(2.*i/3.), -1.))
output_coordinates = [(-1., 1.), (1., 1.)]

sub = Substrate(input_coordinates, output_coordinates)

# ES-HyperNEAT specific parameters.
params = {"initial_depth": 0,
          "max_depth": 1,
          "variance_threshold": 0.03,
          "band_threshold": 0.3,
          "iteration_level": 1,
          "division_threshold": 0.5,
          "max_weight": 8.0,
          "activation": "sigmoid"}

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_pole_balancing')


# Use the gym_runner to run this experiment using ES-HyperNEAT.
def run(gens, env):
    winner, stats = run_es(gens, env, 500, config, params, sub)
    print("es_hyperneat_polebalancing_small done") 
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make("CartPole-v1")

    # Run!
    winner = run(100, env)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(winner, config)
    network = ESNetwork(sub, cppn, params)
    net = network.create_phenotype_network(filename="es_hyperneat_pole_balancing_small_winner")
    draw_net(cppn, filename="es_hyperneat_pole_balancing_small_cppn")
    with open('es_hyperneat_pole_balancing_small_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

