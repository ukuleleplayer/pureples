import neat 
import logging
import cPickle as pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network input and output coordinates.
input_coordinates = [(-0.33, -1.), (0.33, -1.)]
output_coordinates = [(-0.5, 1.), (0.,1.), (0.5, 1.)]
hidden_coordinates = [[(-0.5, 0.5), (0.5, 0.5)], [(0.0, 0.0)], [(-0.5, -0.5), (0.5, -0.5)]]

sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)
activations = len(hidden_coordinates) + 2

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_mountain_car')


def run(gens, env):
    winner, stats = run_hyper(gens, env, 200, config, sub, activations, max_trials=0)
    print("hyperneat_mountain_car done") 
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
    net = create_phenotype_network(cppn, sub)
    draw_net(cppn, filename="hyperneat_mountain_car_cppn")
    draw_net(net, filename="hyperneat_mountain_car_winner")
    with open('hyperneat_mountain_car_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
