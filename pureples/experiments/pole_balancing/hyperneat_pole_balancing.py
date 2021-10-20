"""
An experiment using HyperNEAT to perform a pole balancing task.
"""

import pickle
import logging
import neat
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network input, hidden and output coordinates.
INPUT_COORDINATES = []
for i in range(0, 4):
    INPUT_COORDINATES.append((-1. + (2.*i/3.), -1.))
HIDDEN_COORDINATES = [[(-0.5, 0.5), (0.5, 0.5)], [(-0.5, -0.5), (0.5, -0.5)]]
OUTPUT_COORDINATES = [(-1., 1.), (1., 1.)]
ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

SUBSTRATE = Substrate(
    INPUT_COORDINATES, OUTPUT_COORDINATES, HIDDEN_COORDINATES)

# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/pole_balancing/config_cppn_pole_balancing')


def run(gens, env):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_hyper(gens, env, 500, CONFIG, SUBSTRATE, ACTIVATIONS)
    print("hyperneat_polebalancing done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = gym.make("CartPole-v1")

    # Run! Only relevant to look at the winner.
    WINNER = run(100, ENVIRONMENT)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)
    draw_net(
        CPPN, filename="pureples/experiments/pole_balancing/hyperneat_pole_balancing_cppn")
    with open('pureples/experiments/pole_balancing/hyperneat_pole_balancing_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
    draw_net(
        WINNER_NET, filename="pureples/experiments/pole_balancing/hyperneat_pole_balancing_winner")
