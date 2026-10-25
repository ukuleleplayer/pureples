"""
An experiment using a variable-sized Adaptive DES-HyperNEAT network to perform a pole balancing task.
"""

import os
import pickle
import logging
import math
import random
import neat
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym
from pureples.experiments.cat_and_mouse.dog import DogEnv
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_adaptive_des
from pureples.des_hyperneat import AdaptiveDESNetwork

# S, M or L; Small, Medium or Large (logic implemented as "Not 'S' or 'M' then Large").
VERSION = "L"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

NUM_DEPLOYMENTS = 11

# Upper bound for the deployment seeds sample_generation_schedule draws.
DEPLOYMENT_SEED_LIMIT = 2 ** 31

# Playback: which deployments get filmed, and how they are turned into gifs.
# Only every GIF_FRAME_STRIDE-th step is kept, so the gif runs through a deployment
# at GIF_FPS * GIF_FRAME_STRIDE simulated steps per second.
#
# Every checkpoint films the two deployments GIF_FIXED_SEEDS pins -- the same rooms,
# starting poses and dog placements each time, so behaviour can be compared directly
# across generations -- plus NUM_RANDOM_GIF_RUNS freshly drawn ones for coverage.
GIF_FIXED_SEEDS = (20250815, 987654321)
NUM_RANDOM_GIF_RUNS = 2
GIF_FPS = 15
GIF_FRAME_STRIDE = 2
GIF_MAX_STEPS = 1400

# A full artifact set (network drawings, pickle and gifs) is written for the best
# genome every CHECKPOINT_INTERVAL generations, into its own folder under
# CHECKPOINT_ROOT, plus one more folder for the final winner.
CHECKPOINT_INTERVAL = 20
CHECKPOINT_ROOT = f"pureples/experiments/cat_and_mouse/results/adaptive_des_dog_{VERSION_TEXT}"

# Size of the fitness-evaluation process pool. Set to 1 to evaluate serially when
# debugging, since worker tracebacks are otherwise hard to read.
NUM_WORKERS = 12

# Network coordinates and the resulting substrate. The outer ring is for distance sensors, the middle ring is for output direction control, and the innermost ring is for dog direction/distance.
RING_RADII = (1., 0.3)
NODES_PER_RING = 12

INPUT_COORDINATES = []

for radius in RING_RADII:
    for i in range(NODES_PER_RING):
        # Start at the ring's lowest point and step clockwise.
        angle = -math.pi / 2 - (2.0 * math.pi * i / NODES_PER_RING)
        INPUT_COORDINATES.append((radius * math.cos(angle), radius * math.sin(angle)))

# Add the middle ring of output nodes for direction control. Built with the same
# ordering as the input rings, so output node i asks the robot to accelerate along
# the same bearing that input node i senses (see DogEnv.drive_axes).
RING_RADII = (0.65,)

OUTPUT_COORDINATES = []

for radius in RING_RADII:
    for i in range(NODES_PER_RING):
        # Start at the ring's lowest point and step clockwise.
        angle = -math.pi / 2 - (2.0 * math.pi * i / NODES_PER_RING)
        OUTPUT_COORDINATES.append((radius * math.cos(angle), radius * math.sin(angle)))

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
CONFIG = neat.config.Config(neat.genome.AdaptiveDesGenome, neat.reproduction.DesReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'pureples/experiments/cat_and_mouse/config_cppn_dog')


def draw_environment_frame(state, ax):
    """Render the current room state for a single frame."""
    ax.clear()

    room_points = state.get("room") if isinstance(state, dict) else getattr(state, "room", None)
    if not room_points:
        return

    xs = [p[0] for p in room_points]
    ys = [p[1] for p in room_points]

    min_x, max_x = min(xs) - 2, max(xs) + 2
    min_y, max_y = min(ys) - 2, max(ys) + 2

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)

    for x, y in room_points:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor="black"))

    current_position = state.get("current_position") if isinstance(state, dict) else getattr(state, "current_position", None)
    if current_position is not None:
        ax.scatter(current_position[0], current_position[1], color="blue", s=120, label="Robot")
        rotation = state.get("rotation") if isinstance(state, dict) else getattr(state, "rotation", 0.0)
        heading_x = math.sin(math.radians(rotation))
        heading_y = math.cos(math.radians(rotation))
        ax.plot(
            [current_position[0], current_position[0] + heading_x * 0.5],
            [current_position[1], current_position[1] + heading_y * 0.5],
            color="red",
            linewidth=2,
        )

    dog_position = state.get("dog_position") if isinstance(state, dict) else getattr(state, "dog_position", None)
    if dog_position is not None:
        ax.scatter(dog_position[0], dog_position[1], color="red", s=120, label="Dog")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Adaptive DES Dog Run")
    ax.legend(loc="upper right")


def env_snapshot(env):
    """Copy out everything a frame needs, so the env can move on."""
    return {
        "room": list(env.room),
        "current_position": list(env.current_position),
        "rotation": env.rotation,
        "dog_position": list(env.dog_position),
    }


def collect_deployment(env, net, max_steps=GIF_MAX_STEPS, stride=GIF_FRAME_STRIDE):
    """
    Run one deployment (a fresh room, robot pose and dog) to its natural end and
    return the states to animate. Only every `stride`-th step is kept.
    """
    ob, _ = env.reset()
    net.reset()

    states = [env_snapshot(env)]
    for step in range(1, max_steps + 1):
        action = net.activate(ob)
        ob, _, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        # Keep the final state even when it lands between two sampled steps.
        if step % stride == 0 or done:
            states.append(env_snapshot(env))
        if done:
            break

    return states


def save_deployment_gif(states, path, fps=GIF_FPS):
    """
    Animate one deployment. The room never changes mid-deployment, so its squares
    are drawn once and only the robot and dog markers are updated per frame --
    redrawing every square each frame is what made this slow.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        room_points = states[0]["room"]
        xs = [p[0] for p in room_points]
        ys = [p[1] for p in room_points]

        ax.set_xlim(min(xs) - 2, max(xs) + 2)
        ax.set_ylim(min(ys) - 2, max(ys) + 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(os.path.splitext(os.path.basename(path))[0])

        ax.add_collection(PatchCollection(
            [Rectangle((x - 0.5, y - 0.5), 1, 1) for x, y in room_points],
            facecolor="white", edgecolor="black", match_original=False))

        first = states[0]
        robot = ax.scatter(*first["current_position"], color="blue", s=120, label="Robot")
        dog = ax.scatter(*first["dog_position"], color="red", s=120, label="Dog")
        heading, = ax.plot([], [], color="red", linewidth=2)
        ax.legend(loc="upper right")

        def update(frame):
            state = states[frame]
            position = state["current_position"]
            robot.set_offsets([position])
            dog.set_offsets([state["dog_position"]])
            heading_x = math.sin(math.radians(state["rotation"]))
            heading_y = math.cos(math.radians(state["rotation"]))
            heading.set_data(
                [position[0], position[0] + heading_x * 0.5],
                [position[1], position[1] + heading_y * 0.5])
            return robot, dog, heading

        animation = FuncAnimation(fig, update, frames=len(states), blit=False)
        animation.save(path, writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)


def film_deployment(env, net, path):
    """
    Film one deployment of `net` in `env` and write it to `path`, falling back to a
    still image of the final state if the gif cannot be written.
    """
    states = collect_deployment(env, net)
    try:
        save_deployment_gif(states, path)
        print(f"Saved animation to {path} ({len(states)} frames at {GIF_FPS} fps)")
    except Exception as exc:
        fallback_path = f"{os.path.splitext(path)[0]}.png"
        fig, ax = plt.subplots(figsize=(8, 8))
        draw_environment_frame(states[-1], ax)
        fig.savefig(fallback_path)
        plt.close(fig)
        print(f"Could not save animation ({exc}); saved a still image to {fallback_path}")


def save_checkpoint(genome, config, out_dir, env):
    """
    Write everything worth keeping about `genome` into `out_dir`: the CPPN drawing
    and pickle, the phenotype network image, and one gif per filmed deployment.

    `env` is filmed in place, so pass an env that is not being used for training --
    its deployment schedule is overwritten here.
    """
    os.makedirs(out_dir, exist_ok=True)

    cppn = neat.nn.DesFeedForwardNetwork.create(genome, config)
    network = AdaptiveDESNetwork(SUBSTRATE, cppn, params(VERSION))
    net = network.create_phenotype_network(config, filename=os.path.join(out_dir, "winner"))
    draw_net(cppn, filename=os.path.join(out_dir, "cppn"))
    with open(os.path.join(out_dir, "cppn.pkl"), 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

    # The two pinned deployments first: DogEnv draws a deployment's room, robot pose
    # and dog placement from the schedule's next seed alone, so these two are the
    # same scenario at every checkpoint and only the behaviour differs.
    env.set_training_schedule(list(GIF_FIXED_SEEDS))
    for run_index, _ in enumerate(GIF_FIXED_SEEDS, start=1):
        film_deployment(env, net, os.path.join(out_dir, f"run_fixed_{run_index}.gif"))

    # Then fresh ones: with no schedule the env draws each deployment unseeded.
    env.set_training_schedule(None)
    for run_index in range(1, NUM_RANDOM_GIF_RUNS + 1):
        film_deployment(env, net, os.path.join(out_dir, f"run_random_{run_index}.gif"))

    return net


class CheckpointReporter(neat.reporting.BaseReporter):
    """
    Save a full artifact set for the generation's best genome every `interval`
    generations, while evolution is still running.

    post_evaluate runs in the parent process between generations, so each folder
    lands on disk as soon as its generation finishes rather than at the end of the
    run. Filming uses this reporter's own env so the training env (and, in a serial
    run, its schedule) is left alone.
    """

    def __init__(self, root, interval=CHECKPOINT_INTERVAL):
        self.root = root
        self.interval = interval
        self.generation = 0
        self.env = DogEnv()

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        completed = self.generation + 1
        if completed % self.interval:
            return

        out_dir = os.path.join(self.root, f"gen_{completed:04d}")
        print(f"Saving checkpoint for generation {completed} to {out_dir}")
        try:
            save_checkpoint(best_genome, config, out_dir, self.env)
        except Exception as exc:
            # Never let a failed drawing or gif write kill a long evolutionary run.
            logging.exception("Checkpoint for generation %d failed: %s", completed, exc)


def sample_generation_schedule():
    """
    Draw one generation's deployments: a seed per deployment.

    The runner calls this once per generation and hands the result to every
    genome in it (see run_adaptive_des), and DogEnv rebuilds a deployment's room,
    robot pose and dog placement from its seed alone -- so within a generation all
    genomes are compared on exactly the same rooms, and each generation gets a
    fresh set of them. Seeds rather than the rooms themselves keeps what crosses
    to the worker processes small.
    """
    return [random.randrange(DEPLOYMENT_SEED_LIMIT) for _ in range(NUM_DEPLOYMENTS)]


def run(gens, env, version, reporters=None):
    """
    Run the pole balancing task using the Gym environment
    Returns the winning genome and the statistics of the run.
    """
    winner, stats = run_adaptive_des(gens, env, 1400, CONFIG, params(version), SUBSTRATE, NUM_DEPLOYMENTS,
                                     num_workers=NUM_WORKERS,
                                     generation_schedule_sampler=sample_generation_schedule,
                                     reporters=reporters)
    print(f"adaptive_des_hyperneat_dog_{VERSION_TEXT} done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    # Setup logger and environment.
    LOGGER = logging.getLogger()
    LOGGER.setLevel(logging.INFO)
    ENVIRONMENT = DogEnv()

    GENERATIONS = 300

    # Run! Every CHECKPOINT_INTERVAL generations the reporter saves the best
    # genome's networks and gifs to disk, so there is something to look at long
    # before the run ends.
    CHECKPOINTS = CheckpointReporter(CHECKPOINT_ROOT)
    WINNER = run(GENERATIONS, ENVIRONMENT, VERSION, reporters=[CHECKPOINTS])[0]

    print(WINNER)

    # The winner gets its own folder, marked final, alongside the checkpoints.
    # Filmed on the reporter's env: the training env may still hold the last
    # generation's schedule (only in a serial run, since workers hold their own
    # copies), and save_checkpoint sets the schedule it wants anyway.
    save_checkpoint(WINNER, CONFIG,
                    os.path.join(CHECKPOINT_ROOT, f"gen_{GENERATIONS:04d}_final"),
                    CHECKPOINTS.env)
