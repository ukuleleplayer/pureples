"""
An experiment using a variable-sized Adaptive DES-HyperNEAT network to perform a pole balancing task.
"""

# Force matplotlib's non-interactive Agg backend BEFORE any pyplot import. The
# default Tk backend spawns Tcl GUI objects; when those are garbage-collected
# while the multiprocessing evaluation pool is alive, Tcl aborts the whole
# process with "Tcl_AsyncDelete: async handler deleted by the wrong thread"
# (preceded by "main thread is not in main loop"), killing the run and losing
# all evolution progress. Agg creates no Tcl interpreter, so this cannot happen.
# Every figure is saved to disk under this backend. Single-run mode switches to
# a GUI backend afterwards (see show_network_window), once the pool is closed
# and every output file is written.
import matplotlib
matplotlib.use("Agg")

import os
import pickle
import logging
import neat
import gymnasium as gym
import matplotlib.pyplot as plt
from pureples.experiments.t_maze.multi_t_maze import (
    MultiTMazeEnv, save_run_gif, sample_generation_schedule,
    all_single_switch_schedules, evaluate_all_single_switch,
    evaluate_double_switch, evaluate_delayed_feedback,
    evaluate_constant_feedback)
from pureples.experiments.t_maze import study
from pureples.shared.visualize import draw_net, plot_fitness_stats
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_adaptive_es
from pureples.es_hyperneat import AdaptiveESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "M"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

MAZE_LENGTH = 1
MAZE_LENGTH_TEXT = "1_turn" if MAZE_LENGTH == 1 else str(MAZE_LENGTH) + "_turns"

NUM_DEPLOYMENTS = 4

# Batch controls. A single invocation runs runs START_RUN..END_RUN (inclusive)
# and then writes a cumulative summary over every run 1..END_RUN currently saved
# on disk -- not just this batch. Run a study in batches by advancing the window,
# e.g. 1..6, then 7..12, ... 25..30; if a batch crashes, only that batch is lost
# and re-running it picks up where the earlier batches left off.
START_RUN = 24
END_RUN = 30

# Maximum number of generations per evolutionary run (also the length the
# per-run fitness series are padded to for the multi-run study).
MAX_GENERATIONS = 1250

# Network coordinates and the resulting substrate.
INPUT_COORDINATES = []

for i in range(0, 5):
    INPUT_COORDINATES.append((-1. + (1.0*i/2), -1.))
INPUT_COORDINATES.append((0., -1.2))

OUTPUT_COORDINATES = [(-1., 1.), (0., 1.), (1., 1.)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES)


def params(version):
    """
    Adaptive DES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 0 if version == "S" else 1 if version == "M" else 2,
            "max_depth": 1 if version == "S" else 2 if version == "M" else 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 8.0,
            "activation": "sigmoid"}


# Config for CPPN. 
CONFIG = neat.config.Config(neat.genome.AdaptiveDefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/t_maze/config_cppn_t_maze')


def run(gens, env, version):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_adaptive_es(gens, env, 500, CONFIG, params(version), SUBSTRATE, NUM_DEPLOYMENTS,
                                    num_workers=12, schedule_sampler=sample_generation_schedule)
    print(f"adaptive_es_hyperneat_t_maze_{MAZE_LENGTH_TEXT}_{VERSION_TEXT} done")
    return winner, stats


def std_connection_an_products(net):
    """
    Yield the A*n product for every standard connection in an Adaptive ES
    phenotype network. Each node_eval is
    ``(node, activation, aggregation, bias, response, std_links, mod_links)``
    and each standard link is ``(in_id, weight, a, b, c, d, n)`` (std_links may
    be None), so ``a`` and ``n`` are at indices 2 and 6.
    """
    products = []
    for node_eval in net.node_evals:
        std_links = node_eval[5]
        if not std_links:
            continue
        for link in std_links:
            products.append(link[2] * link[6])
    return products


def network_size(net):
    """
    Return ``(nodes, standard connections, modulatory connections)`` for an
    Adaptive ES phenotype network.

    Nodes are counted as the inputs and outputs plus every node that is either
    evaluated or feeds a connection, matching the set the network itself keeps
    values for. Each node_eval is
    ``(node, activation, aggregation, bias, response, std_links, mod_links)``
    and either link list may be None (a node reached only by the other kind).
    Modulatory connections whose source is not a node are dropped when the
    phenotype is built, so everything counted here affects the network.

    The standard-connection total also covers the connections whose A*n is
    exactly 0, which ``std_connection_an_products``' inhibitory/excitatory split
    counts as neither.
    """
    nodes = set(net.input_nodes) | set(net.output_nodes)
    std_connections = mod_connections = 0
    for node_eval in net.node_evals:
        nodes.add(node_eval[0])
        for link in node_eval[5] or ():
            nodes.add(link[0])
            std_connections += 1
        for link in node_eval[6] or ():
            nodes.add(link[0])
            mod_connections += 1
    return len(nodes), std_connections, mod_connections


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = MultiTMazeEnv(MAZE_LENGTH)

    BASE_NAME = f"adaptive_es_hyperneat_t_maze_{MAZE_LENGTH_TEXT}_{VERSION_TEXT}"
    RESULTS_DIR = "pureples/experiments/t_maze/results"
    DATA_DIR = f"{RESULTS_DIR}/{BASE_NAME}_data"

    def save_standard_outputs(winner, stats, prefix, network_title,
                              save_artifacts=True, save_gif=False):
        """
        Save a run's outputs under ``prefix``, put ``winner`` through the four
        post-evolution tests, and return the ``(net, cppn, test_means)`` triple
        for analysis.

        The fitness plot, the phenotype network and the four tests are produced
        for EVERY run, successful or not, so an unsuccessful run's best genome is
        measured on exactly the same footing as a successful one. ``winner`` is
        the best genome the run ever produced (neat's ``Population.best_genome``),
        which for a successful run is the one that crossed the threshold.

        ``save_artifacts`` additionally writes the winner network image, the CPPN
        drawing and the CPPN pickle. It is switched off for unsuccessful runs so
        the results directory keeps only the networks worth inspecting; the
        phenotype is still built either way, since the tests and the connection
        analysis need it.

        ``test_means`` is a ``(test1, test2, test3, test4)`` tuple of mean
        per-deployment fitnesses:
          Test 1 -- every training-range single-switch schedule (10 deployments);
          Test 2 -- double-switch generalization over 20 episodes (6 deployments);
          Test 3 -- Test 1's schedules with delayed reward feedback (10 deployments);
          Test 4 -- Test 1's schedules with the feedback signal pinned to 1 during
                    every reward presentation (10 deployments), so the signal says
                    nothing about which arm was reached while fitness is still
                    scored on the true reward.
        All four run post-evolution only and never affect training.

        Behaviour gifs are written only when ``save_gif`` is True (single-run
        mode); multi-run studies save no gifs.
        """
        plot_fitness_stats(stats, f"{prefix}_fitness.png",
                           title=f"{network_title} — fitness over generations")

        cppn = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
        network = AdaptiveESNetwork(SUBSTRATE, cppn, params(VERSION))
        net = network.create_phenotype_network(
            filename=f"{prefix}_winner" if save_artifacts else None, show=False)

        if save_artifacts:
            draw_net(cppn, filename=f"{prefix}_cppn")
            with open(f"{prefix}_cppn.pkl", 'wb') as output:
                pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

        test_means = []
        for label, evaluate in (
                ("Test 1 (all single-switch schedules)", evaluate_all_single_switch),
                ("Test 2 (double-switch generalization)", evaluate_double_switch),
                ("Test 3 (delayed reward feedback)", evaluate_delayed_feedback),
                ("Test 4 (constant reward feedback)", evaluate_constant_feedback)):
            fitnesses = evaluate(net, ENVIRONMENT)
            mean_fitness = sum(fitnesses) / len(fitnesses)
            test_means.append(mean_fitness)
            print(f"{network_title} — {label}: mean over {len(fitnesses)} deployments "
                  f"= {mean_fitness:.4f}  per-deployment: "
                  f"{[round(f, 4) for f in fitnesses]}")

        if save_gif:
            # One gif per feedback mode, both showing the same deployment (the
            # first single-switch schedule): the Test 1 gif has the reward
            # presented on arrival, the Test 3 gif has it delayed, so the two
            # side by side show how the delay changes the behaviour.
            schedule = all_single_switch_schedules()
            for gif_suffix, gif_label, delayed in (("test1", "Test 1", False),
                                                   ("test3", "Test 3", True)):
                save_run_gif(net, ENVIRONMENT, f"{prefix}_{gif_suffix}_run.gif",
                             title=f"{network_title} — {gif_label} deployment", fps=4,
                             schedule=schedule, episodes_per_deployment=10,
                             delayed_feedback=delayed)
        return net, cppn, tuple(test_means)

    def build_record(stats, run_number, solved, winner, net, test_means):
        """
        Collect one run's metrics into a ``study.RunRecord``.

        Called for every run. The network analysis and the four post-evolution
        test means are measured on the run's best genome whether or not it
        reached the fitness threshold, so the study's all-runs statistics are not
        conditioned on solving; ``solved`` only records which group the run falls
        into. ``num_branches`` is always left unset: Adaptive ES networks have no
        branches.
        """
        mean_series, best_series = study.padded_fitness_series(stats, MAX_GENERATIONS)
        mean_an, inhibitory, excitatory = study.connection_sign_metrics(
            std_connection_an_products(net))
        node_count, std_conns, mod_conns = network_size(net)
        return study.RunRecord(
            run_number=run_number,
            generations_to_solution=len(stats.most_fit_genomes),
            mean_fitness_per_gen=mean_series,
            best_fitness_per_gen=best_series,
            solved=solved,
            mean_an_product=mean_an,
            inhibitory_count=inhibitory,
            excitatory_count=excitatory,
            cppn_connection_count=study.count_cppn_connections(winner),
            cppn_node_count=study.count_cppn_nodes(winner, CONFIG),
            network_node_count=node_count,
            std_connection_count=std_conns,
            mod_connection_count=mod_conns,
            all_switch_fitness=test_means[0],
            double_switch_fitness=test_means[1],
            delayed_feedback_fitness=test_means[2],
            constant_feedback_fitness=test_means[3],
        )

    def show_network_window(winner, prefix):
        """
        Open the interactive window for the winning phenotype network.

        Everything above runs under the headless Agg backend, forced at import
        time because Tk objects collected while the evaluation pool is alive
        abort the process (see the module docstring). Evolution -- and with it
        the pool -- has finished by the time this is called, and every output
        file is already on disk, so a GUI backend is safe here.

        Switching backends discards the figures built under Agg, so the
        phenotype is expanded a second time to build its figure on a live Tk
        canvas (that is what wires up the connection check buttons and the Save
        button). The expansion is deterministic, so the window shows the same
        network as the saved ``{prefix}_winner.png``.
        """
        try:
            plt.close('all')
            plt.switch_backend("TkAgg")
        except Exception as exc:
            print(f"Could not open an interactive window ({exc}); the network "
                  f"image was still saved to {prefix}_winner.png")
            return

        cppn = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
        network = AdaptiveESNetwork(SUBSTRATE, cppn, params(VERSION))
        network.create_phenotype_network(filename=f"{prefix}_winner", show=True)

    # Run the batch START_RUN..END_RUN. Every run saves its fitness graph, its
    # per-generation fitness table, its post-evolution test results and network
    # analysis, and a JSON record; successful runs additionally save the network
    # image and the CPPN drawing + pickle. Testing every run -- not just the ones
    # that cross the fitness threshold -- is what keeps the study's all-runs
    # statistics free of the bias that threshold introduces.
    # After the batch, a cumulative summary is written over every run 1..END_RUN
    # currently on disk. A single-run batch (START_RUN == END_RUN) additionally
    # saves the Test 1/Test 3 gifs and pops up the interactive network window,
    # mirroring the old single-run mode.
    os.makedirs(DATA_DIR, exist_ok=True)
    single_run_batch = (START_RUN == END_RUN)
    pending_window = None  # (winner, prefix) shown after the summary, single-run batch only

    for run_number in range(START_RUN, END_RUN + 1):
        print(f"=== Adaptive ES T-Maze: run {run_number} "
              f"(batch {START_RUN}-{END_RUN}) ===")
        winner, stats_tuple = run(MAX_GENERATIONS, ENVIRONMENT, VERSION)
        stats = stats_tuple[0]
        run_prefix = f"{DATA_DIR}/{BASE_NAME}_run{run_number}"
        run_title = f"Adaptive ES T-Maze run {run_number}"

        is_solved = study.solved(winner, CONFIG)
        if not is_solved:
            print(f"Run {run_number} did not reach the fitness threshold within "
                  f"{MAX_GENERATIONS} generations; its best genome is still "
                  f"analysed and tested below.")

        net, _cppn, test_means = save_standard_outputs(
            winner, stats, run_prefix, run_title,
            save_artifacts=is_solved,
            save_gif=single_run_batch and is_solved)
        record = build_record(stats, run_number, is_solved, winner, net, test_means)
        if is_solved and single_run_batch:
            pending_window = (winner, run_prefix)

        study.write_run_report(f"{run_prefix}_data.txt", record, MAX_GENERATIONS,
                               title="Adaptive ES T-Maze")
        study.save_run_record(study.run_record_path(DATA_DIR, BASE_NAME, run_number), record)

        # Release the network figure before the next run.
        plt.close('all')

    # Cumulative summary over every saved run 1..END_RUN (all batches so far).
    all_records = study.load_saved_records(DATA_DIR, BASE_NAME, END_RUN)
    study.write_summary(
        graph_path=f"{DATA_DIR}/{BASE_NAME}_summary_fitness.png",
        text_path=f"{DATA_DIR}/{BASE_NAME}_summary.txt",
        records=all_records, total_runs=END_RUN,
        max_generations=MAX_GENERATIONS, title="Adaptive ES T-Maze")

    solved_count = sum(1 for r in all_records if r.solved)
    print(f"Batch {START_RUN}-{END_RUN} complete. Cumulative summary covers "
          f"{len(all_records)} saved run(s) up to run {END_RUN} "
          f"({solved_count} solved); written to {DATA_DIR}.")

    if pending_window is not None:
        show_network_window(*pending_window)