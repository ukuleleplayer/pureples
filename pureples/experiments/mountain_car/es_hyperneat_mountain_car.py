"""
An experiment using a variable-sized ES-HyperNEAT network to perform a mountain car task.
"""

import logging
import pickle
import neat
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_es
from pureples.es_hyperneat.es_hyperneat import ESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network input and output coordinates.
INPUT_COORDINATES = [(-0.33, -1.), (0.33, -1.)]
OUTPUT_COORDINATES = [(-0.5, 1.), (0., 1.), (0.5, 1.)]

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
                            'pureples/experiments/mountain_car/config_cppn_mountain_car')


def run(gens, env, version):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_es(gens, env, 200, CONFIG, params(
        version), SUBSTRATE, max_trials=0)
    print(f"es_hyperneat_mountain_car_{VERSION_TEXT} done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = gym.make("MountainCar-v0")

    # Run! Only relevant to look at the winner.
    WINNER = run(200, ENVIRONMENT, VERSION)[0]

    # Save CPPN if wished reused and draw it + winner to file.
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    NETWORK = ESNetwork(SUBSTRATE, CPPN, params)
    NET = NETWORK.create_phenotype_network(
        filename=f"es_hyperneat_mountain_car_{VERSION_TEXT}_winner")
    draw_net(CPPN, filename=f"es_hyperneat_mountain_car_{VERSION_TEXT}_cppn")
    with open(f'es_hyperneat_mountain_car_{VERSION_TEXT}_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
