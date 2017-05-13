import neat 
import logging
import cPickle as pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.gym_runner import run_neat

# Config for NEAT.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_neat_mountain_car')


def run(gens, env):
    winner, stats = run_neat(gens, env, 200, config, max_trials=0)
    print("neat_mountain_car done") 
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make("MountainCar-v0")

    # Run!
    winner = run(200, env)[0]

    # Save net if wished reused and draw it to file.
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    draw_net(net, filename="neat_mountain_car_winner")
    with open('neat_mountain_car_winner.pkl', 'wb') as output:
        pickle.dump(net, output, pickle.HIGHEST_PROTOCOL)

