"""
An experiment using a variable-sized ES-HyperNEAT network to perform a pole balancing task.
"""

import pickle
import logging
import neat
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_es
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = []

for i in range(0, 4):
    INPUT_COORDINATES.append((-1. + (2.*i/3.), -1.))

OUTPUT_COORDINATES = [(-1., 1.), (1., 1.)]
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
            "max_weight": 8.0,
            "activation": "sigmoid"}


# Config for CPPN.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/pole_balancing/config_cppn_pole_balancing')


def run(gens, env, version):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_es(gens, env, 500, CONFIG, params(version), SUBSTRATE)
    print(f"es_hyperneat_polebalancing_{VERSION_TEXT} done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = gym.make("CartPole-v1")

    # Run! Only relevant to look at the winner.
    WINNER = run(100, ENVIRONMENT, VERSION)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, params(VERSION))
    NET = NETWORK.create_phenotype_network(
        filename=f"pureples/experiments/pole_balancing/es_hyperneat_pole_balancing_{VERSION_TEXT}_winner")
    draw_net(
        CPPN, filename=f"pureples/experiments/pole_balancing/es_hyperneat_pole_balancing_{VERSION_TEXT}_cppn")
    with open(f'pureples/experiments/pole_balancing/es_hyperneat_pole_balancing_{VERSION_TEXT}_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
