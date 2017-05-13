import neat 
import logging
import cPickle as pickle
import gym
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from pureples.hyperneat.hyperneat import create_phenotype_network

# Network input, hidden and output coordinates.
input_coordinates = []
for i in range(0,4):
    input_coordinates.append((-1. +(2.*i/3.), -1.))
hidden_coordinates = [[(-0.5, 0.5), (0.5, 0.5)], [(-0.5, -0.5), (0.5, -0.5)]]
output_coordinates = [(-1., 1.), (1., 1.)]
activations = len(hidden_coordinates) + 2

sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)

# Config for CPPN.
config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'config_cppn_pole_balancing')


# Use the gym_runner to run this experiment using HyperNEAT.
def run(gens, env):
    winner, stats = run_hyper(gens, env, 500, config, sub, activations)
    print("hyperneat_polebalancing done") 
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
    winner_net = create_phenotype_network(cppn, sub)
    draw_net(cppn, filename="hyperneat_pole_balancing_cppn")
    with open('hyperneat_pole_balancing_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)
    draw_net(winner_net, filename="hyperneat_pole_balancing_winner")
