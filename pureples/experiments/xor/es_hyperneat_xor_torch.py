"""
ES-HyperNEAT XOR experiment with a PyTorch phenotype network.

This mirrors ``es_hyperneat_xor.py`` but uses ``TorchESNetwork`` so that
the evolved substrate network runs in PyTorch. NEAT-Python is still used
for evolving the CPPN, which keeps this implementation directly
comparable to the original pure-Python version.
"""

import pickle

import neat
import torch

from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.torch_es_hyperneat import TorchESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "S"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network inputs and expected outputs.
XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_OUTPUTS = [(0.0,), (1.0,), (1.0,), (0.0,)]

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = [(-1.0, -1.0), (0.0, -1.0), (1.0, -1.0)]
OUTPUT_COORDINATES = [(0.0, 1.0)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def params(version: str):
    """
    ES-HyperNEAT specific parameters.
    """
    return {
        "initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
        "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
        "variance_threshold": 0.03,
        "band_threshold": 0.3,
        "iteration_level": 1,
        "division_threshold": 0.5,
        "max_weight": 5.0,
        "activation": "sigmoid",
    }


DYNAMIC_PARAMS = params(VERSION)

# Config for CPPN (same as the original experiment).
CONFIG = neat.config.Config(
    neat.genome.DefaultGenome,
    neat.reproduction.DefaultReproduction,
    neat.species.DefaultSpeciesSet,
    neat.stagnation.DefaultStagnation,
    "pureples/experiments/xor/config_cppn_xor",
)


def eval_fitness(genomes, config):
    """
    Fitness function using the Torch ES-HyperNEAT phenotype.

    For each genome evaluate its fitness as the mean squared error,
    identical to the original XOR experiment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for _, genome in genomes:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = TorchESNetwork(SUBSTRATE, cppn, DYNAMIC_PARAMS, device=device).to(device)
        network.eval()

        sum_square_error = 0.0

        with torch.no_grad():
            for xor_inputs, xor_expected in zip(XOR_INPUTS, XOR_OUTPUTS):
                # Original experiment appends a bias term 1.0 to the input.
                x0, x1 = xor_inputs
                input_tensor = torch.tensor([[x0, x1, 1.0]], dtype=torch.float32, device=device)
                output_tensor = network(input_tensor)
                xor_output = float(output_tensor[0, 0].item())

                sum_square_error += ((xor_output - xor_expected[0]) ** 2.0) / 4.0

        genome.fitness = 1 - sum_square_error


def run(gens: int, version: str):
    """
    Create the population and run the XOR task by providing eval_fitness
    as the fitness function. Returns the winning genome and the statistics
    of the run.
    """
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    global DYNAMIC_PARAMS
    DYNAMIC_PARAMS = params(version)

    winner = pop.run(eval_fitness, gens)
    print(f"es_hyperneat_xor_torch_{VERSION_TEXT} done")
    return winner, stats


if __name__ == "__main__":
    # Train and inspect a winner using the Torch phenotype.
    WINNER = run(300, VERSION)[0]
    print("\nBest genome:\n{!s}".format(WINNER))

    # Verify network output against training data.
    print("\nOutput (Torch ES-HyperNEAT):")
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NETWORK = TorchESNetwork(SUBSTRATE, CPPN, DYNAMIC_PARAMS, device=DEVICE).to(DEVICE)
    NETWORK.eval()

    with torch.no_grad():
        for inputs, expected in zip(XOR_INPUTS, XOR_OUTPUTS):
            x0, x1 = inputs
            input_tensor = torch.tensor([[x0, x1, 1.0]], dtype=torch.float32, device=DEVICE)
            output_tensor = NETWORK(input_tensor)
            output = float(output_tensor[0, 0].item())
            print("  input {!r}, expected output {!r}, got {!r}".format(inputs, expected, output))

    # Save CPPN if wished reused and draw it to file.
    draw_net(
        CPPN,
        filename=f"pureples/experiments/xor/es_hyperneat_xor_torch_{VERSION_TEXT}_cppn",
    )
    with open(
        f"pureples/experiments/xor/es_hyperneat_xor_torch_{VERSION_TEXT}_cppn.pkl", "wb"
    ) as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)

