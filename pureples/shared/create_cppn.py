"""
CPPN creator.
"""

import neat
from neat.graphs import feed_forward_layers


def create_cppn(genome, config, output_activation_function="tanh"):
    """
    Receives a genome and returns its phenotype (a FeedForwardNetwork).
    """

    # Gather expressed connections.
    connections = [cg.key for cg in genome.connections.values() if cg.enabled]

    layers = feed_forward_layers(
        config.genome_config.input_keys, config.genome_config.output_keys, connections)
    node_evals = []
    for layer in layers:
        for node in layer:
            inputs = []
            node_expr = []  # currently unused

            for conn_key in connections:
                inode, onode = conn_key
                if onode == node:
                    cg = genome.connections[conn_key]
                    inputs.append((inode, cg.weight))
                    node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

            ng = genome.nodes[node]
            aggregation_function = config.genome_config.aggregation_function_defs.get(
                ng.aggregation)
            # Fix the output note's activation function to any function.
            if node in config.genome_config.output_keys:
                ng.activation = output_activation_function
            activation_function = config.genome_config.activation_defs.get(
                ng.activation)
            node_evals.append(
                (node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

    return neat.nn.FeedForwardNetwork(config.genome_config.input_keys,
                                      config.genome_config.output_keys, node_evals)
