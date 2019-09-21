import neat 
import logging
try:
   import cPickle as pickle
except:
   import pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.gym_runner import run_neat


# Config for FeedForwardNetwork.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_pole_balancing')


# Use the gym_runner to run this experiment using NEAT.
def run(gens, env):
    winner, stats = run_neat(gens, env, 500, config)
    print("neat_pole_balancing done") 
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make("CartPole-v1")

    # Run!
    winner = run(500, env)[0]

    # Save net if wished reused and draw it + winner to file.
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    draw_net(winner_net, filename="neat_pole_balancing_winner")
    with open('neat_pole_balancing_winner.pkl', 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)

