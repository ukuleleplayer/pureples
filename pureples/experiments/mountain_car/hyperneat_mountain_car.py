"""
An experiment using HyperNEAT to perform a mountain car task.
"""

import logging
import pickle
import gym
import neat
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network input and output coordinates.
input_coordinates = [(-0.33, -1.), (0.33, -1.)]
OUTPUT_COORDINATES = [(-0.5, 1.), (0., 1.), (0.5, 1.)]
HIDDEN_COORDINATES = [[(-0.5, 0.5), (0.5, 0.5)],
                      [(0.0, 0.0)], [(-0.5, -0.5), (0.5, -0.5)]]

SUBSTRATE = Substrate(
    input_coordinates, OUTPUT_COORDINATES, HIDDEN_COORDINATES)
ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_mountain_car')


def run(gens, env):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_hyper(gens, env, 200, CONFIG,
                              SUBSTRATE, ACTIVATIONS, max_trials=0)
    print("hyperneat_mountain_car done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = gym.make("MountainCar-v0")

    # Run! Only relevant to look at the winner.
    WINNER = run(200, ENVIRONMENT)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    cppn = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NET = create_phenotype_network(cppn, SUBSTRATE)
    draw_net(cppn, filename="hyperneat_mountain_car_cppn")
    draw_net(NET, filename="hyperneat_mountain_car_winner")
    with open('hyperneat_mountain_car_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
