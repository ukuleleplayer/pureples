"""
Reopen the interactive phenotype-network window for a CPPN that was already saved
to disk.

This is the window that `adaptive_des_t_maze.py` / `adaptive_es_t_maze.py` pop up at
the end of a single evolutionary run (see their `show_network_window`): the expanded
substrate network with check buttons for the branches and for the standard/modulatory
connections, plus a Save PNG button. Here it is driven from a file instead of from a
live run, so any winner kept in a results folder can be inspected again.

Run it as a module from the repository root:

    python -m pureples.shared.interactive_visualization

Point `des_cppn_file` OR `es_cppn_file` (exactly one; the other stays None) at either

  * the text `print(WINNER)` writes at the end of evolution -- the `Nodes:` /
    `Connections:` listing, with or without its `Key:` / `Fitness:` header, as in
    `results/adaptive_des_dog_large/Candidate_3119/CPPN.txt`; or
  * a `cppn.pkl` saved next to it.

Which variable is set decides whether the file is read as an Adaptive DES CPPN or an
Adaptive ES one; the substrate, the CPPN config and the S/M/L size below have to be
edited to match the experiment the file came from, since the file itself does not
record them.

Note that the network image is always written to disk as a side effect of opening the
window (the drawing helpers in `visualize.py` save before they block), so a run of
this module leaves a PNG behind even if the window cannot be shown.

For a T-Maze winner there is a second option: set `GENERATE_TMAZE_GIFS = True` and the
network is also rolled out on every deployment of the post-evolution tests named in
`GIF_TESTS`, one gif per deployment, written before the window opens. That needs the
substrate block below to be the T-Maze one.

Alongside the gifs the same rollouts write `reward_sequences.txt`, which records every
deployment's sequence of rewards ("HHHLHHHHHH", one symbol per episode) next to the gif
that shows it, so the behaviour can be read without watching them. Rendering is the slow
part of filming, so `WRITE_GIF_FILES = False` produces that text file on its own in
seconds.
"""

import copy
import math
import os
import pickle
import re

import neat
import matplotlib.pyplot as plt

from pureples.shared.substrate import Substrate
from pureples.des_hyperneat import AdaptiveDESNetwork
from pureples.es_hyperneat import AdaptiveESNetwork


# ---------------------------------------------------------------------------
# What to visualize. Set exactly one of the two paths; leave the other None.
# ---------------------------------------------------------------------------

des_cppn_file = None
es_cppn_file = "pureples/experiments/t_maze/results/adaptive_es_hyperneat_t_maze_1_turn_medium_data/adaptive_es_hyperneat_t_maze_1_turn_medium_run28_cppn.pkl"

# CPPN config. It has to contain the section matching the genome class in use:
# [AdaptiveDesGenome] (+ [DesReproduction]) for a DES file, [AdaptiveDefaultGenome]
# (+ [DefaultReproduction]) for an ES one. The dog config below has the DES sections
# but no [AdaptiveDefaultGenome]; for an ES text file use
# 'pureples/experiments/t_maze/config_cppn_t_maze' instead.
CONFIG_FILE = "pureples/experiments/t_maze/config_cppn_t_maze"

# S, M or L; Small, Medium or Large. This is the VERSION the run used -- it sets the
# quadtree's initial_depth and max_depth, so the wrong value expands the CPPN into a
# different network.
VERSION = "M"

# A results folder saves an exact pickle of the CPPN next to the printed text. The
# text rounds every float to 3 decimals (see BaseGene.__str__), so it reproduces the
# CPPN only approximately; when a sibling pickle is found it is used instead. Set this
# to False to always read the text.
PREFER_PICKLE = True

# Where to write the network image. None puts it next to the CPPN file, named after it.
OUTPUT_IMAGE = None


# ---------------------------------------------------------------------------
# T-Maze behaviour gifs (optional).
#
# With GENERATE_TMAZE_GIFS on, the loaded network is rolled out on every
# deployment of each named post-evolution test and one gif per deployment is
# written before the interactive window opens. This only makes sense for a
# T-Maze winner: the substrate block above must be the T-Maze one (6 inputs,
# 3 outputs), or the network cannot be driven by the environment's observations
# and filming stops with an error rather than producing nonsense.
#
# GIF_TESTS names entries of multi_t_maze.TEST_GIF_SPECS. The default films
# Test 1 (10 deployments), Test 2 (6) and Test 4 (10) -- 26 gifs, the biggest
# part of a run's time; "test3" (delayed feedback) can be added too. Each gif
# covers a whole deployment, so Test 2's are about twice as long as the rest.
# ---------------------------------------------------------------------------

GENERATE_TMAZE_GIFS = True
GIF_TESTS = ("test1", "test2", "test4")

# Every deployment's reward sequence -- "HHHLHHHHHH", one symbol per episode --
# is written to REWARD_SEQUENCE_FILE inside the gif folder, so a run's behaviour
# can be read off without watching 26 gifs. See write_reward_sequence_report.
#
# Set WRITE_GIF_FILES = False to run the deployments and write only the text file.
WRITE_GIF_FILES = False
REWARD_SEQUENCE_FILE = "reward_sequences.txt"

# Where the gifs go. None puts them in a folder named after the CPPN file,
# alongside it.
GIF_OUTPUT_DIR = None

# Playback speed, and the maze the deployments run in. MAZE_LENGTH must match the
# experiment the CPPN came from (adaptive_des_t_maze.py's MAZE_LENGTH).
GIF_FPS = 4
MAZE_LENGTH = 1


# ---------------------------------------------------------------------------
# The substrate and the DES/ES parameters of the experiment being inspected.
# Copied from adaptive_des_dog.py -- replace this block with the corresponding one
# from whichever experiment produced the file.
# ---------------------------------------------------------------------------

# Network coordinates and the resulting substrate. The outer ring is for distance sensors, the middle ring is for output direction control, and the innermost ring is for dog direction/distance.
# RING_RADII = (1., 0.3)
# NODES_PER_RING = 12

# INPUT_COORDINATES = []

# for radius in RING_RADII:
#     for i in range(NODES_PER_RING):
#         # Start at the ring's lowest point and step clockwise.
#         angle = -math.pi / 2 - (2.0 * math.pi * i / NODES_PER_RING)
#         INPUT_COORDINATES.append((radius * math.cos(angle), radius * math.sin(angle)))

# # Add the middle ring of output nodes for direction control. Built with the same
# # ordering as the input rings, so output node i asks the robot to accelerate along
# # the same bearing that input node i senses (see DogEnv.drive_axes).
# RING_RADII = (0.65,)

# OUTPUT_COORDINATES = []

# for radius in RING_RADII:
#     for i in range(NODES_PER_RING):
#         # Start at the ring's lowest point and step clockwise.
#         angle = -math.pi / 2 - (2.0 * math.pi * i / NODES_PER_RING)
#         OUTPUT_COORDINATES.append((radius * math.cos(angle), radius * math.sin(angle)))

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = []

for i in range(0, 5):
    INPUT_COORDINATES.append((-1. + (1.0*i/2), -1.))
INPUT_COORDINATES.append((0., -1.2))

OUTPUT_COORDINATES = [(-1., 1.), (0., 1.), (1., 1.)]

SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def params(version):
    """
    Adaptive DES/ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 8.0,
            "activation": "sigmoid"}


# ---------------------------------------------------------------------------
# Everything below is machinery; the settings above are the knobs.
# ---------------------------------------------------------------------------

# The two algorithms this module can draw, and what each one needs. `node_gene` is the
# gene class whose repr must appear in a text file for that algorithm -- reading a DES
# file as ES (or the other way round) is caught by comparing against it.
ALGORITHMS = {
    "des": {
        "label": "Adaptive DES-HyperNEAT",
        "genome_type": neat.genome.AdaptiveDesGenome,
        "reproduction_type": neat.reproduction.DesReproduction,
        "node_gene": neat.genes.AdaptiveDesNodeGene,
    },
    "es": {
        "label": "Adaptive ES-HyperNEAT",
        "genome_type": neat.genome.AdaptiveDefaultGenome,
        "reproduction_type": neat.reproduction.DefaultReproduction,
        "node_gene": neat.genes.DefaultNodeGene,
    },
}

# One node line of `print(genome)`: the dict key, then the gene's repr. The key is
# printed twice (once as the dict key, once as the gene's `key=` attribute).
NODE_LINE_RE = re.compile(r"^(?P<dict_key>-?\d+)\s+(?P<gene_class>\w+)\((?P<attributes>.*)\)$")

# One connection line. Its `key` is a tuple, whose comma would break a plain split of
# the attribute list, so it is pulled out separately first.
CONNECTION_LINE_RE = re.compile(
    r"^(?P<gene_class>\w+)\(key=\((?P<in_key>-?\d+),\s*(?P<out_key>-?\d+)\),\s*"
    r"(?P<attributes>.*)\)$")


class CppnFileError(Exception):
    """A CPPN file could not be read, or does not match the settings above."""


def _parse_attribute_value(text):
    """
    Turn one printed attribute value back into a Python value: a bool, a float, or --
    for the activation and aggregation names -- the bare string.
    """
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return float(text)
    except ValueError:
        return text


def _parse_attributes(text):
    """
    Parse a printed attribute list (`a=1.000, b=False, activation=gauss`) into a dict.
    """
    attributes = {}
    for item in text.split(", "):
        name, separator, value = item.partition("=")
        if not separator:
            raise CppnFileError(f"could not parse the attribute {item!r}")
        attributes[name.strip()] = _parse_attribute_value(value.strip())
    return attributes


def parse_genome_text(path, algorithm):
    """
    Parse the text `print(genome)` produces into the pieces a genome is built from:
    ``(key, fitness, nodes, connections)``.

    `nodes` maps a node key to its attribute dict and `connections` maps an
    ``(in, out)`` key to its attribute dict; both keep the order the file lists them
    in, which is what the branch-node order is recovered from (see `build_genome`).
    The `Key:` / `Fitness:` header is optional -- some saved files have it split off
    into a separate info file -- so both come back as None when it is absent.
    """
    expected_gene_class = ALGORITHMS[algorithm]["node_gene"].__name__

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    key = fitness = None
    nodes, connections = {}, {}
    section = None

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("Key:"):
            key = line[len("Key:"):].strip()
            continue
        if line.startswith("Fitness:"):
            fitness = line[len("Fitness:"):].strip()
            continue
        if line == "Nodes:":
            section = "nodes"
            continue
        if line == "Connections:":
            section = "connections"
            continue

        if section == "nodes":
            match = NODE_LINE_RE.match(line)
            if match is None:
                raise CppnFileError(f"{path}, line {line_number}: could not parse the "
                                    f"node line {line!r}")
            gene_class = match.group("gene_class")
            if gene_class != expected_gene_class:
                raise CppnFileError(
                    f"{path}, line {line_number}: the nodes are {gene_class} genes, but "
                    f"an {ALGORITHMS[algorithm]['label']} CPPN is written with "
                    f"{expected_gene_class} genes. This file looks like it belongs to "
                    f"the other algorithm -- set it as "
                    f"{'es_cppn_file' if algorithm == 'des' else 'des_cppn_file'} instead.")
            try:
                attributes = _parse_attributes(match.group("attributes"))
            except CppnFileError as error:
                raise CppnFileError(f"{path}, line {line_number}: {error}") from None
            nodes[int(match.group("dict_key"))] = attributes

        elif section == "connections":
            match = CONNECTION_LINE_RE.match(line)
            if match is None:
                raise CppnFileError(f"{path}, line {line_number}: could not parse the "
                                    f"connection line {line!r}")
            try:
                attributes = _parse_attributes(match.group("attributes"))
            except CppnFileError as error:
                raise CppnFileError(f"{path}, line {line_number}: {error}") from None
            connections[(int(match.group("in_key")),
                         int(match.group("out_key")))] = attributes

        else:
            raise CppnFileError(
                f"{path}, line {line_number}: {line!r} comes before any 'Nodes:' or "
                f"'Connections:' header. This does not look like the output of "
                f"print(genome).")

    if not nodes:
        raise CppnFileError(f"{path}: no node genes found. Expected the output of "
                            f"print(genome), which starts with a 'Nodes:' line.")

    return key, fitness, nodes, connections


def _build_gene(gene, attributes, path, description):
    """
    Fill in a freshly constructed gene from a parsed attribute dict, insisting that the
    dict holds exactly the attributes the gene class declares.
    """
    expected = {attribute.name for attribute in gene._gene_attributes}
    if hasattr(gene, "innovation"):
        expected.add("innovation")
    # `key` is printed inside the gene's repr as well, but it is already set from the
    # line's own key, so it is not an attribute to copy over.
    provided = set(attributes) - {"key"}

    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing {missing}")
        if unknown:
            problems.append(f"unexpected {unknown}")
        raise CppnFileError(
            f"{path}: the {description} does not match {type(gene).__name__} "
            f"({'; '.join(problems)}). The file was probably written by a different "
            f"version of the gene classes.")

    for attribute in gene._gene_attributes:
        setattr(gene, attribute.name, attributes[attribute.name])
    return gene


def build_genome(path, algorithm, config, parsed):
    """
    Rebuild a genome object from parsed text, so the usual CPPN creation can run on it.

    Only what the CPPN needs is restored: the node and connection genes, and -- for DES
    -- the branch nodes. No innovation tracker is set, since only mutation uses one.

    Recovering the DES branch nodes deserves a note, because `print(genome)` does not
    list them. A branch node is exactly a node with ``is_branch=True``: the two are kept
    in step by `AdaptiveDesGenome` (deleting a branch deletes from both dicts, and
    `mutate_delete_node` never picks a branch key), and the end of every `mutate` copies
    each branch gene back from its node gene -- so the printed node *is* the branch
    gene, `scale`, `n` and all. Their order matters as much as their values: both
    `DesFeedForwardNetwork.create` and `AdaptiveDESNetwork.create_phenotype_network`
    walk the branch nodes in `num_outputs`-sized groups, taking the first of each group
    as the branch's standard output and the second as its modulatory one. Branch keys
    enter `nodes` and `branch_nodes` together, so the file's node order reproduces it.
    """
    genome_config = config.genome_config
    node_gene_type = ALGORITHMS[algorithm]["node_gene"]
    key, fitness, nodes, connections = parsed

    # Genome keys are ints during a run, but the file's is only ever printed back out.
    try:
        genome_key = int(key)
    except (TypeError, ValueError):
        genome_key = key
    genome = ALGORITHMS[algorithm]["genome_type"](genome_key)

    for node_key, attributes in nodes.items():
        # is_branch is a constructor argument as well as an attribute, since DES gene
        # classes take it up front.
        if node_gene_type is neat.genes.AdaptiveDesNodeGene:
            gene = node_gene_type(node_key, bool(attributes.get("is_branch", False)))
        else:
            gene = node_gene_type(node_key)
        genome.nodes[node_key] = _build_gene(gene, attributes, path,
                                            f"attributes of node {node_key}")

    for connection_key, attributes in connections.items():
        innovation = attributes.get("innovation")
        if innovation is None:
            raise CppnFileError(f"{path}: connection {connection_key} has no innovation "
                                f"number, which DefaultConnectionGene requires.")
        gene = neat.genes.DefaultConnectionGene(connection_key, innovation=int(innovation))
        genome.connections[connection_key] = _build_gene(
            gene, attributes, path, f"attributes of connection {connection_key}")

    if fitness is not None:
        genome.fitness = float(fitness)

    if algorithm == "des":
        genome.branch_nodes = {key: copy.deepcopy(gene)
                               for key, gene in genome.nodes.items() if gene.is_branch}
        _check_branch_nodes(path, genome, genome_config)
    else:
        missing_outputs = [key for key in genome_config.output_keys
                           if key not in genome.nodes]
        if missing_outputs:
            raise CppnFileError(
                f"{path}: the CPPN output nodes {missing_outputs} are missing. "
                f"{os.path.basename(CONFIG_FILE)} declares num_outputs = "
                f"{genome_config.num_outputs}, so the file and CONFIG_FILE disagree.")

    _check_input_keys(path, genome, genome_config)
    return genome


def _check_branch_nodes(path, genome, genome_config):
    """
    Check the recovered DES branch nodes against the branch ids the file printed.

    The grouping into branches comes from the order the branch nodes are listed in, so
    a disagreement with the printed `branch_id` values means the order was not
    preserved and the standard/modulatory split would be wrong.
    """
    branch_keys = list(genome.branch_nodes)
    if not branch_keys:
        raise CppnFileError(f"{path}: no branch nodes (nodes with is_branch=True) found. "
                            f"An Adaptive DES CPPN always has at least one branch.")

    num_outputs = genome_config.num_outputs
    if len(branch_keys) % num_outputs:
        raise CppnFileError(
            f"{path}: found {len(branch_keys)} branch nodes, which is not a multiple of "
            f"the num_outputs = {num_outputs} declared by "
            f"{os.path.basename(CONFIG_FILE)}. Branch nodes come in groups of "
            f"num_outputs, so the file and CONFIG_FILE disagree.")

    for branch_index, group in enumerate(zip(*(iter(branch_keys),) * num_outputs)):
        printed = {genome.nodes[key].branch_id for key in group}
        if printed != {float(branch_index)}:
            print(f"Warning: branch nodes {list(group)} were recovered as branch "
                  f"{branch_index}, but the file prints branch_id "
                  f"{sorted(printed)} for them. The branches are grouped by the order "
                  f"they are listed in, so the drawing's branch colours and the "
                  f"standard/modulatory split may be wrong.")


def _check_input_keys(path, genome, genome_config):
    """
    Check that no connection comes from an input pin the config does not have.
    """
    input_keys = set(genome_config.input_keys)
    unknown = sorted({in_key for in_key, _ in genome.connections
                      if in_key < 0 and in_key not in input_keys})
    if unknown:
        raise CppnFileError(
            f"{path}: connections come from the input pins {unknown}, which do not "
            f"exist -- {os.path.basename(CONFIG_FILE)} declares num_inputs = "
            f"{genome_config.num_inputs} (pins "
            f"{min(input_keys)}..{max(input_keys)}). The file and CONFIG_FILE disagree.")


def load_config(algorithm):
    """
    Load the CPPN config, reading the section that matches `algorithm`'s genome class.
    """
    algorithm_info = ALGORITHMS[algorithm]
    if not os.path.exists(CONFIG_FILE):
        raise CppnFileError(f"CONFIG_FILE {CONFIG_FILE!r} does not exist. Paths are "
                            f"relative to the working directory, so run this from the "
                            f"repository root.")
    try:
        return neat.config.Config(algorithm_info["genome_type"],
                                  algorithm_info["reproduction_type"],
                                  neat.species.DefaultSpeciesSet,
                                  neat.stagnation.DefaultStagnation,
                                  CONFIG_FILE)
    except Exception as error:
        raise CppnFileError(
            f"could not read {CONFIG_FILE} as an {algorithm_info['label']} config "
            f"({error}). It needs a [{algorithm_info['genome_type'].__name__}] section "
            f"and a [{algorithm_info['reproduction_type'].__name__}] one.") from None


def find_pickle(path):
    """
    Return the exact CPPN pickle saved alongside `path`, or None if there is none.

    Both naming conventions the experiments use are checked: `cppn.pkl` in the same
    folder (the cat-and-mouse checkpoints) and `<prefix>_cppn.pkl` (the t-maze runs,
    whose files share a per-run prefix).
    """
    directory = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    candidates = [os.path.join(directory, "cppn.pkl"),
                  os.path.join(directory, f"{stem}_cppn.pkl"),
                  os.path.join(directory, f"{stem}.pkl")]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_cppn(path, algorithm):
    """
    Load the CPPN named by `path` and return ``(cppn, config, source_description)``.

    A `.pkl` path is unpickled directly. A text path is read as printed genome text,
    unless `PREFER_PICKLE` is on and an exact pickle sits next to it. The config is
    loaded only when it is needed -- an Adaptive ES phenotype is expanded without one,
    so an ES pickle can be viewed with a CONFIG_FILE that has no ES section.
    """
    if not os.path.exists(path):
        raise CppnFileError(f"{path!r} does not exist. Paths are relative to the "
                            f"working directory, so run this from the repository root.")

    pickle_path = path if path.endswith(".pkl") else None
    if pickle_path is None and PREFER_PICKLE:
        pickle_path = find_pickle(path)

    if pickle_path is not None:
        with open(pickle_path, "rb") as handle:
            cppn = pickle.load(handle)
        config = load_config(algorithm) if algorithm == "des" else None
        source = f"the exact CPPN pickle {pickle_path}"
        if pickle_path != path:
            source += f" (saved alongside {path})"
        return cppn, config, source

    config = load_config(algorithm)
    parsed = parse_genome_text(path, algorithm)
    genome = build_genome(path, algorithm, config, parsed)

    note = ("Note: the printed genome rounds every value to 3 decimals, so this CPPN is "
            "a close approximation of the original rather than a copy of it. The "
            "expansion can therefore land on a slightly different network -- a value "
            "near the variance or band threshold may fall on the other side of it.")
    if find_pickle(path) is not None:
        note += " Set PREFER_PICKLE = True to use the saved cppn.pkl instead."
    print(note)

    if algorithm == "des":
        cppn = neat.nn.DesFeedForwardNetwork.create(genome, config)
    else:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)

    key, fitness = parsed[0], parsed[1]
    described = ", ".join(part for part in
                          (f"key {key}" if key is not None else "",
                           f"fitness {fitness}" if fitness is not None else "") if part)
    source = f"the printed genome in {path}"
    if described:
        source += f" ({described})"
    source += (f" -- {len(genome.nodes)} nodes, "
               f"{sum(1 for gene in genome.connections.values() if gene.enabled)} "
               f"enabled of {len(genome.connections)} connections")
    return cppn, config, source


def describe_phenotype(net, algorithm):
    """
    Summarise an expanded phenotype network: its node count, and how many standard and
    modulatory connections it ended up with.

    The link tuples differ between the two algorithms -- a DES node_eval carries one
    list of standard links, each with its own dict of modulatory weights, while an ES
    one carries a standard list and a modulatory list -- so each is counted its own way.
    """
    nodes = set(net.input_nodes) | set(net.output_nodes)
    std_connections = mod_connections = 0

    for node_eval in net.node_evals:
        nodes.add(node_eval[0])
        if algorithm == "des":
            for link in node_eval[5]:
                nodes.add(link[0])
                std_connections += 1
                mod_connections += len(link[8])
        else:
            for link in node_eval[5] or []:
                nodes.add(link[0])
                std_connections += 1
            for link in node_eval[6] or []:
                nodes.add(link[0])
                mod_connections += 1

    summary = (f"{len(nodes)} nodes, {std_connections} standard and "
               f"{mod_connections} modulatory connections")
    if algorithm == "des":
        summary += f", {net.num_branches} branches"
    return summary


def expand_phenotype(cppn, config, algorithm, image_path):
    """
    Expand `cppn` over the substrate and return the phenotype network, writing its
    drawing to `image_path`.

    Kept separate from `show_network_window` so the network can be used headlessly --
    the behaviour gifs are filmed from it under the non-interactive backend, before
    the blocking window switches to TkAgg.
    """
    if algorithm == "des":
        network = AdaptiveDESNetwork(SUBSTRATE, cppn, params(VERSION))
        net = network.create_phenotype_network(config, filename=image_path, show=False)
    else:
        network = AdaptiveESNetwork(SUBSTRATE, cppn, params(VERSION))
        net = network.create_phenotype_network(filename=image_path, show=False)

    print(f"Phenotype network: {describe_phenotype(net, algorithm)}")
    print(f"Network image written to {image_path}")
    return net


def save_behaviour_gifs(net, out_dir, title, cppn_path):
    """
    Run `net` on every deployment of each test in GIF_TESTS, film them (unless
    WRITE_GIF_FILES is off), write the reward-sequence report, and return the
    multi_t_maze.DeploymentRecord list.

    The network is driven exactly as the post-evolution tests drive it (see
    multi_t_maze.save_test_gifs), so each gif -- and each recorded reward
    sequence -- shows the behaviour that produced that test's score, including
    how the agent reacts to the reward arms swapping.

    Raises CppnFileError when the configured substrate cannot be driven by the
    T-Maze's observations, which is what happens if the substrate block at the top
    of this module still belongs to another experiment.
    """
    from pureples.experiments.t_maze.multi_t_maze import MultiTMazeEnv, save_test_gifs

    env = MultiTMazeEnv(MAZE_LENGTH)
    observation, _ = env.reset()
    if len(observation) != len(SUBSTRATE.input_coordinates):
        raise CppnFileError(
            f"the substrate has {len(SUBSTRATE.input_coordinates)} inputs but the "
            f"T-Maze produces {len(observation)}-value observations. The substrate "
            f"block at the top of this module belongs to a different experiment -- "
            f"replace it with the one from adaptive_des_t_maze.py / "
            f"adaptive_es_t_maze.py (and check VERSION and CONFIG_FILE too) before "
            f"filming.")

    verb = "Filming" if WRITE_GIF_FILES else "Running"
    print(f"{verb} {', '.join(GIF_TESTS)} into {out_dir} ...")
    records = save_test_gifs(net, env, out_dir, tests=GIF_TESTS, title=title,
                             fps=GIF_FPS, film=WRITE_GIF_FILES)

    # Each test's mean over its deployments is the figure the study reports for
    # it, so printing it here says whether the filmed behaviour is the behaviour
    # that was scored.
    if WRITE_GIF_FILES:
        print(f"Wrote {len(records)} gif(s) to {out_dir}")
    for test_name in GIF_TESTS:
        scores = [record.fitness for record in records if record.test == test_name]
        if scores:
            print(f"  {test_name}: mean fitness {sum(scores) / len(scores):.4f} "
                  f"over {len(scores)} deployment(s)")

    report_path = os.path.join(out_dir, REWARD_SEQUENCE_FILE)
    write_reward_sequence_report(report_path, records, title, cppn_path)
    print(f"Reward sequences written to {report_path}")
    return records


def write_reward_sequence_report(path, records, title, cppn_path):
    """
    Write the text companion to a folder of gifs: one line per deployment giving
    the sequence of rewards the agent collected, the arms it walked into, and the
    arms the high reward sat on, all one symbol per episode and in order.

    `records` is what save_test_gifs returned, so the rows appear in the order
    the gifs are numbered and each row names the gif it describes -- the point
    being to read a run's behaviour here and only open the gifs that look worth
    watching.
    """
    from pureples.experiments.t_maze.multi_t_maze import TEST_GIF_SPECS

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # The tests in the order they were run, so the file reads in the order the
    # gifs were written.
    tests = []
    for record in records:
        if record.test not in tests:
            tests.append(record.test)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Reward sequences — {title}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"CPPN: {cppn_path}\n")
        f.write(f"Maze length: {MAZE_LENGTH}\n\n")

        f.write("One row per deployment, one symbol per episode, in order:\n")
        f.write("  rewards      H = high reward (1.0), L = low reward (0.1), "
                "- = no arm reached (episode timed out)\n")
        f.write("  chosen arm   l = left arm, r = right arm, . = no arm reached\n")
        f.write("  high arm     the arm the high reward sat on that episode; it flips at\n")
        f.write("               the deployment's switch episode(s), which is the change\n")
        f.write("               the agent has to detect from its reward feedback\n")
        f.write("  steps        how long the deployment ran, which is also how many\n")
        f.write("               frames its gif has\n\n")
        f.write("Fitness is the deployment's score as the test counts it: a reward only\n")
        f.write("counts inside a run of two or more consecutive episodes of the same\n")
        f.write("reward (2.0 per high episode, 1.0 per low), and the whole deployment is\n")
        f.write("zeroed if the agent only ever reached one arm.\n")

        for test_name in tests:
            rows = [record for record in records if record.test == test_name]
            label = TEST_GIF_SPECS[test_name]["label"]
            mean_fitness = sum(record.fitness for record in rows) / len(rows)

            f.write("\n\n")
            f.write(f"{label} — {len(rows)} deployments, "
                    f"mean fitness {mean_fitness:.4f}\n")
            f.write("-" * 70 + "\n")

            columns = [("#", lambda r: f"{r.index:02d}"),
                       ("deployment", lambda r: r.description),
                       ("fitness", lambda r: f"{r.fitness:.1f}"),
                       ("rewards", lambda r: r.rewards),
                       ("chosen arm", lambda r: r.chosen_arms),
                       ("high arm", lambda r: r.high_arms),
                       ("steps", lambda r: str(r.episodes[-1].end_step
                                               if r.episodes else 0))]
            if any(record.gif_path for record in rows):
                columns.append(("gif", lambda r: os.path.basename(r.gif_path or "")))

            table = [[header for header, _ in columns]]
            table += [[get(record) for _, get in columns] for record in rows]
            widths = [max(len(row[i]) for row in table) for i in range(len(columns))]
            for row in table:
                f.write("  " + "  ".join(cell.ljust(width) for cell, width
                                         in zip(row, widths)).rstrip() + "\n")

            # The per-test tallies below are what the rows add up to, so a
            # policy can be recognised without counting symbols by hand: a
            # win-stay/lose-shift agent pays exactly one low reward per switch,
            # while timeouts and single-arm deployments mark an agent that never
            # navigates properly at all.
            episodes = [outcome for record in rows for outcome in record.episodes]
            highs = sum(1 for outcome in episodes if outcome.symbol == "H")
            lows = sum(1 for outcome in episodes if outcome.symbol == "L")
            timeouts = len(episodes) - highs - lows
            single_arm = sum(1 for record in rows
                             if len({outcome.chosen_side for outcome in record.episodes
                                     if outcome.chosen_side}) < 2)
            f.write(f"\n  {len(episodes)} episodes: {highs} high, {lows} low, "
                    f"{timeouts} timed out\n")
            f.write(f"  {single_arm} of {len(rows)} deployment(s) reached only one arm "
                    f"(scored 0 by the single-arm rule)\n")

            # A deployment normally runs all its episodes; a short sequence means
            # the rollout hit the test's step budget first, which is worth saying
            # out loud rather than leaving as a mysteriously short row.
            full = TEST_GIF_SPECS[test_name]["episodes_per_deployment"]
            short = [record.index for record in rows if len(record.episodes) < full]
            if short:
                f.write(f"  deployment(s) {short} ran out of steps before their "
                        f"{full} episodes finished, so their sequences are cut short\n")


def show_network_window(cppn, config, algorithm, image_path):
    """
    Expand `cppn` over the substrate and open the interactive window for the result.

    The window is the one `draw_adaptive_des` / `draw_adaptive_es` build, so it comes
    with the same check buttons and Save PNG button as the window that appears at the
    end of a single evolutionary run. Those helpers write `image_path` before the
    window blocks, so the image is saved whether or not a GUI backend is available.

    The expansion happens AFTER the backend switch on purpose: switching discards
    every figure built under the previous backend, so a network expanded beforehand
    would leave an empty window behind.
    """
    try:
        plt.switch_backend("TkAgg")
    except Exception as error:
        print(f"Could not switch to an interactive backend ({error}); the network image "
              f"will still be written to {image_path}.")

    net = expand_phenotype(cppn, config, algorithm, image_path)
    print("Close the window to finish.")
    plt.show()
    return net


def resolve_target():
    """
    Work out which of the two path settings is in use, and where the image goes.

    Returns ``(path, algorithm, image_path, gif_dir)``.
    """
    if (des_cppn_file is None) == (es_cppn_file is None):
        if des_cppn_file is None:
            raise CppnFileError("both des_cppn_file and es_cppn_file are None. Set one "
                                "of them to the CPPN file to visualize.")
        raise CppnFileError("des_cppn_file and es_cppn_file are both set. Set only the "
                            "one matching the file's algorithm and leave the other None.")

    if des_cppn_file is not None:
        path, algorithm = des_cppn_file, "des"
    else:
        path, algorithm = es_cppn_file, "es"

    directory = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]

    image_path = OUTPUT_IMAGE
    if image_path is None:
        image_path = os.path.join(directory, f"{stem}_phenotype.png")

    gif_dir = GIF_OUTPUT_DIR
    if gif_dir is None:
        gif_dir = os.path.join(directory, f"{stem}_gifs")

    return path, algorithm, image_path, gif_dir


def main():
    """
    Load the configured CPPN, optionally film its T-Maze behaviour, and open its
    phenotype window.

    The gifs are written first, under the non-interactive backend the module starts
    on: filming needs the network but not a window, and the interactive switch that
    follows is one-way for any figure built before it.
    """
    try:
        path, algorithm, image_path, gif_dir = resolve_target()
        print(f"{ALGORITHMS[algorithm]['label']}, version {VERSION}, substrate with "
              f"{len(SUBSTRATE.input_coordinates)} inputs and "
              f"{len(SUBSTRATE.output_coordinates)} outputs")
        cppn, config, source = load_cppn(path, algorithm)
        print(f"Loaded {source}")

        if GENERATE_TMAZE_GIFS:
            net = expand_phenotype(cppn, config, algorithm, image_path)
            save_behaviour_gifs(
                net, gif_dir,
                title=f"{ALGORITHMS[algorithm]['label']} — "
                      f"{os.path.basename(os.path.dirname(path)) or os.path.basename(path)}",
                cppn_path=path)
    except CppnFileError as error:
        print(f"Error: {error}")
        return 1

    show_network_window(cppn, config, algorithm, image_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
