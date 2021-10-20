"""
All logic concerning ES-HyperNEAT resides here.
"""
import copy
import neat
import numpy as np
from pureples.hyperneat.hyperneat import query_cppn
from pureples.shared.visualize import draw_es


class ESNetwork:
    """
    The evolvable substrate network.
    """

    def __init__(self, substrate, cppn, params):
        self.substrate = substrate
        self.cppn = cppn
        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.connections = set()
        # Number of layers in the network.
        self.activations = 2 ** params["max_depth"] + 1
        activation_functions = neat.activations.ActivationFunctionSet()
        self.activation = activation_functions.get(params["activation"])

    def create_phenotype_network(self, filename=None):
        """
        Create a RecurrentNetwork using the ES-HyperNEAT approach.
        """
        input_coordinates = self.substrate.input_coordinates
        output_coordinates = self.substrate.output_coordinates

        input_nodes = list(range(len(input_coordinates)))
        output_nodes = list(range(len(input_nodes), len(
            input_nodes)+len(output_coordinates)))
        hidden_idx = len(input_coordinates)+len(output_coordinates)

        coordinates, indices, draw_connections, node_evals = [], [], [], []
        nodes = {}

        coordinates.extend(input_coordinates)
        coordinates.extend(output_coordinates)
        indices.extend(input_nodes)
        indices.extend(output_nodes)

        # Map input and output coordinates to their IDs.
        coords_to_id = dict(zip(coordinates, indices))

        # Where the magic happens.
        hidden_nodes, connections = self.es_hyperneat()

        # Map hidden coordinates to their IDs.
        for x, y in hidden_nodes:
            coords_to_id[x, y] = hidden_idx
            hidden_idx += 1

        # For every coordinate:
        # Check the connections and create a node with corresponding connections if appropriate.
        for (x, y), idx in coords_to_id.items():
            for c in connections:
                if c.x2 == x and c.y2 == y:
                    draw_connections.append(c)
                    if idx in nodes:
                        initial = nodes[idx]
                        initial.append((coords_to_id[c.x1, c.y1], c.weight))
                        nodes[idx] = initial
                    else:
                        nodes[idx] = [(coords_to_id[c.x1, c.y1], c.weight)]

        # Combine the indices with the connections/links;
        # forming node_evals used by the RecurrentNetwork.
        for idx, links in nodes.items():
            node_evals.append((idx, self.activation, sum, 0.0, 1.0, links))

        # Visualize the network?
        if filename is not None:
            draw_es(coords_to_id, draw_connections, filename)

        # This is actually a feedforward network.
        return neat.nn.RecurrentNetwork(input_nodes, output_nodes, node_evals)

    @staticmethod
    def get_weights(p):
        """
        Recursively collect all weights for a given QuadPoint.
        """
        temp = []

        def loop(pp):
            if pp is not None and all(child is not None for child in pp.cs):
                for i in range(0, 4):
                    loop(pp.cs[i])
            else:
                if pp is not None:
                    temp.append(pp.w)
        loop(p)
        return temp

    def variance(self, p):
        """
        Find the variance of a given QuadPoint.
        """
        if not p:
            return 0.0
        return np.var(self.get_weights(p))

    def division_initialization(self, coord, outgoing):
        """
        Initialize the quadtree by dividing it in appropriate quads.
        """
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        q = [root]

        while q:
            p = q.pop(0)

            p.cs[0] = QuadPoint(p.x - p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[1] = QuadPoint(p.x - p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[2] = QuadPoint(p.x + p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[3] = QuadPoint(p.x + p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)

            for c in p.cs:
                c.w = query_cppn(coord, (c.x, c.y), outgoing,
                                 self.cppn, self.max_weight)

            if (p.lvl < self.initial_depth) or (p.lvl < self.max_depth and self.variance(p)
                                                > self.division_threshold):
                for child in p.cs:
                    q.append(child)

        return root

    def pruning_extraction(self, coord, p, outgoing):
        """
        Determines which connections to express - high variance = more connetions.
        """
        for c in p.cs:
            d_left, d_right, d_top, d_bottom = None, None, None, None

            if self.variance(c) > self.variance_threshold:
                self.pruning_extraction(coord, c, outgoing)
            else:
                d_left = abs(c.w - query_cppn(coord, (c.x - p.width,
                                                      c.y), outgoing, self.cppn, self.max_weight))
                d_right = abs(c.w - query_cppn(coord, (c.x + p.width,
                                                       c.y), outgoing, self.cppn, self.max_weight))
                d_top = abs(c.w - query_cppn(coord, (c.x, c.y - p.width),
                                             outgoing, self.cppn, self.max_weight))
                d_bottom = abs(c.w - query_cppn(coord, (c.x, c.y +
                                                        p.width), outgoing, self.cppn, self.max_weight))

                con = None
                if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:
                    if outgoing:
                        con = Connection(coord[0], coord[1], c.x, c.y, c.w)
                    else:
                        con = Connection(c.x, c.y, coord[0], coord[1], c.w)
                if con is not None:
                    # Nodes will only connect upwards.
                    # If connections to same layer is wanted, change to con.y1 <= con.y2.
                    if not c.w == 0.0 and con.y1 < con.y2 and not (con.x1 == con.x2 and con.y1 == con.y2):
                        self.connections.add(con)

    def es_hyperneat(self):
        """
        Explores the hidden nodes and their connections.
        """
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes, unexplored_hidden_nodes = set(), set()
        connections1, connections2, connections3 = set(), set(), set()

        for x, y in inputs:  # Explore from inputs.
            root = self.division_initialization((x, y), True)
            self.pruning_extraction((x, y), root, True)
            connections1 = connections1.union(self.connections)
            for c in connections1:
                hidden_nodes.add((c.x2, c.y2))
            self.connections = set()

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)

        for _ in range(self.iteration_level):  # Explore from hidden.
            for x, y in unexplored_hidden_nodes:
                root = self.division_initialization((x, y), True)
                self.pruning_extraction((x, y), root, True)
                connections2 = connections2.union(self.connections)
                for c in connections2:
                    hidden_nodes.add((c.x2, c.y2))
                self.connections = set()

            unexplored_hidden_nodes = hidden_nodes - unexplored_hidden_nodes

        for x, y in outputs:  # Explore to outputs.
            root = self.division_initialization((x, y), False)
            self.pruning_extraction((x, y), root, False)
            connections3 = connections3.union(self.connections)
            self.connections = set()

        connections = connections1.union(connections2.union(connections3))

        return self.clean_net(connections)

    def clean_net(self, connections):
        """
        Clean a net for dangling connections:
        Intersects paths from input nodes with paths to output.
        """
        connected_to_inputs = set(tuple(i)
                                  for i in self.substrate.input_coordinates)
        connected_to_outputs = set(tuple(i)
                                   for i in self.substrate.output_coordinates)
        true_connections = set()

        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)

        add_happened = True
        while add_happened:  # The path from inputs.
            add_happened = False
            temp_input_connections = copy.deepcopy(initial_input_connections)
            for c in temp_input_connections:
                if (c.x1, c.y1) in connected_to_inputs:
                    connected_to_inputs.add((c.x2, c.y2))
                    initial_input_connections.remove(c)
                    add_happened = True

        add_happened = True
        while add_happened:  # The path to outputs.
            add_happened = False
            temp_output_connections = copy.deepcopy(initial_output_connections)
            for c in temp_output_connections:
                if (c.x2, c.y2) in connected_to_outputs:
                    connected_to_outputs.add((c.x1, c.y1))
                    initial_output_connections.remove(c)
                    add_happened = True

        true_nodes = connected_to_inputs.intersection(connected_to_outputs)
        for c in connections:
            # Only include connection if both source and target node resides in the real path from input to output
            if (c.x1, c.y1) in true_nodes and (c.x2, c.y2) in true_nodes:
                true_connections.add(c)

        true_nodes -= (set(self.substrate.input_coordinates)
                       .union(set(self.substrate.output_coordinates)))

        return true_nodes, true_connections


class QuadPoint:
    """
    Class representing an area in the quadtree.
    Defined by a center coordinate and the distance to the edges of the area.
    """

    def __init__(self, x, y, width, lvl):
        self.x = x
        self.y = y
        self.w = 0.0
        self.width = width
        self.cs = [None] * 4
        self.lvl = lvl


class Connection:
    """
    Class representing a connection from one point to another with a certain weight.
    """

    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight

    # Below is needed for use in set.
    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.weight))


def find_pattern(cppn, coord, res=60, max_weight=5.0):
    """
    From a given point, query the cppn for weights to all other points.
    This can be visualized as a connectivity pattern.
    """
    im = np.zeros((res, res))

    for x2 in range(res):
        for y2 in range(res):

            x2_scaled = -1.0 + (x2/float(res))*2.0
            y2_scaled = -1.0 + (y2/float(res))*2.0

            i = [coord[0], coord[1], x2_scaled, y2_scaled, 1.0]
            n = cppn.activate(i)[0]

            im[x2][y2] = n * max_weight

    return im
