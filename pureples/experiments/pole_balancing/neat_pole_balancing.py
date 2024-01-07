"""
An experiment using NEAT to perform a pole balancing task.
"""

import logging
import pickle
import neat
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.gym_runner import run_neat


# Config for FeedForwardNetwork.
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/pole_balancing/config_neat_pole_balancing')


def run(gens, env):
    """
    Create the population and run the XOR task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_neat(gens, env, 500, CONFIG)
    print("neat_pole_balancing done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = gym.make("CartPole-v1")

    # Run!
    WINNER = run(500, ENVIRONMENT)[0]

    # Save net if wished reused and draw it + winner to file.
    WINNER_NET = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    draw_net(
        WINNER_NET, filename="pureples/experiments/pole_balancing/neat_pole_balancing_winner")
    with open('pureples/experiments/pole_balancing/neat_pole_balancing_winner.pkl', 'wb') as output:
        pickle.dump(WINNER_NET, output, pickle.HIGHEST_PROTOCOL)
