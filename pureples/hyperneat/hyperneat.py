import neat 


# Creates a recurrent network using a cppn and a substrate.
def create_phenotype_network(cppn, substrate, activation_function="sigmoid"):
    input_coordinates = substrate.input_coordinates
    output_coordinates = substrate.output_coordinates
    hidden_coordinates = substrate.hidden_coordinates  # List of layers, first index = top layer.

    input_nodes = range(len(input_coordinates))
    output_nodes = range(len(input_nodes), len(input_nodes)+len(output_coordinates))

    counter = 0
    for layer in hidden_coordinates:
        counter += len(layer)
    hidden_nodes = range(len(input_nodes)+len(output_nodes), len(input_nodes)+len(output_nodes)+counter)

    node_evals = []

    # Get activation function.
    activation_functions = neat.activations.ActivationFunctionSet()
    activation = activation_functions.get(activation_function)

    # Connect hidden to output.
    counter = 0
    for oc in output_coordinates:
        idx = 0
        for layer in hidden_coordinates:
            im = find_neurons(cppn, oc, layer, hidden_nodes[idx], False)
            idx += len(layer)
            if im:
                node_evals.append((output_nodes[counter], activation, sum, 0.0, 1.0, im))
        counter += 1

    # Connect hidden to hidden - starting from the top layer.
    current_layer = 1
    idx = 0
    for layer in hidden_coordinates:
        idx += len(layer)
        counter = idx - len(layer)
        for i in range(current_layer, len(hidden_coordinates)):
            for hc in layer:
                im = find_neurons(cppn, hc, hidden_coordinates[i], hidden_nodes[idx], False)
                if im:
                    node_evals.append((hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
                counter += 1
            counter -= idx
        current_layer += 1

    # Connect input to hidden.
    counter = 0
    for layer in hidden_coordinates:
        for hc in layer:
            im = find_neurons(cppn, hc, input_coordinates, input_nodes[0], False)
            if im:
                node_evals.append((hidden_nodes[counter], activation, sum, 0.0, 1.0, im))
            counter += 1

    return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)


# Find the neurons to which the given coord is connected.
def find_neurons(cppn, coord, nodes, start_idx, outgoing, max_weight=5.0):
    im = []
    idx = start_idx

    for node in nodes:
        w = query_cppn(coord, node, outgoing, cppn, max_weight)

        if w != 0.0:  # Only include connection if the weight isn't 0.0.
            im.append((idx, w))
        idx += 1

    return im


# Get the weight from one point to another using the CPPN - takes into consideration which point is source/target.
def query_cppn(coord1, coord2, outgoing, cppn, max_weight=5.0):

    if outgoing:
        i = [coord1[0], coord1[1], coord2[0], coord2[1], 1.0]
    else:
        i = [coord2[0], coord2[1], coord1[0], coord1[1], 1.0]
    w = cppn.activate(i)[0]
    if abs(w) > 0.2:  # If abs(weight) is below threshold, treat weight as 0.0.
        return w * max_weight
    else:
        return 0.0

