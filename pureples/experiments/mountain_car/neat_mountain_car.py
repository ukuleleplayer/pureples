"""
An experiment using NEAT to perform a mountain car task.
"""

import logging
import pickle
import gym
import neat
from pureples.shared.visualize import draw_net
from pureples.shared.gym_runner import run_neat

# Config for NEAT.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_mountain_car')


def run(gens, env):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_neat(gens, env, 200, CONFIG, max_trials=0)
    print("neat_mountain_car done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = gym.make("MountainCar-v0")

    # Run! Only relevant to look at the winner.
    WINNER = run(200, ENVIRONMENT)[0]

    # Save net if wished reused and draw it to file.
    NET = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    draw_net(NET, filename="neat_mountain_car_winner")
    with open('neat_mountain_car_winner.pkl', 'wb') as output:
        pickle.dump(NET, output, pickle.HIGHEST_PROTOCOL)
