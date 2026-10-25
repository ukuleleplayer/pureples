import os
import math
import random
import itertools
from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym

# Fixed schedule of the four deployments every genome is evaluated on. Each entry
# is (starting high-reward arm, switch_reward_episode), where switch_reward_episode
# is the last episode BEFORE the reward arms swap (episodes are 0-indexed, 10 per
# deployment). In order, the four variations reproduce the reward sequences:
#   1. ("left",  2) -> L L L R R R R R R R  (high reward starts left,  switch on 4th episode)
#   2. ("right", 6) -> R R R R R R R L L L  (high reward starts right, switch on 8th episode)
#   3. ("right", 2) -> R R R L L L L L L L  (high reward starts right, switch on 4th episode)
#   4. ("left",  6) -> L L L L L L L R R R  (high reward starts left,  switch on 8th episode)
DEPLOYMENT_SCHEDULE = [
    ("left", 2),
    ("right", 6),
    ("right", 2),
    ("left", 6),
]

# ---------------------------------------------------------------------------
# Post-evolution test schedules.
#
# Four tests are run on a winning genome AFTER evolution; none of them ever
# influence training. Training always uses the sampled per-genome schedules (see
# sample_generation_schedule), which are left untouched.
#
#   Test 1 (evaluate_all_single_switch): NOT a generalization test. It sweeps
#     every training-range schedule -- each combination of start side and single
#     switch episode the genome could have trained on -- so it measures behaviour
#     across the full set of deployments evolution draws from. 2 start sides x the
#     5 SWITCH_EPISODE_CHOICES = 10 single-switch, 10-episode deployments.
#
#   Test 2 (evaluate_double_switch): a genuine generalization test. Each
#     deployment runs 20 episodes and switches the reward arms TWICE, at a
#     distinct pair of switch episodes drawn from DOUBLE_SWITCH_EPISODES. Every
#     start side x switch pair is tested: 2 sides x 3 pairs = 6 deployments. The
#     genome only ever trained on single switches, so two switches is unseen.
#
#   Test 3 (evaluate_delayed_feedback): Test 1's 10 schedules, but with the
#     reward feedback the network observes delayed -- on reaching a goal the agent
#     advances to the next episode immediately and only observes the reward
#     DELAYED_FEEDBACK_DELAY_STEPS steps later (see MultiTMazeEnv.step).
#
#   Test 4 (evaluate_constant_feedback): Test 1's 10 schedules again, presented
#     with the same immediate timing, but the feedback value the network observes
#     is pinned to 1.0 for every step of every reward presentation -- a high
#     reward, a low reward and a timeout all look identical (see
#     MultiTMazeEnv.feedback_signal). Fitness keeps being scored on the TRUE
#     reward, so the test asks how much of an agent's adaptation actually depends
#     on the content of its feedback channel rather than on the mere arrival of a
#     signal or on an internal timer.

# Test 2 switch episodes, used directly as switch_reward_episode values (the
# reward swaps AFTER episodes 5, 10 and/or 15 of the 0-indexed 20-episode
# deployment). Two distinct switches per deployment -> the three unordered pairs.
DOUBLE_SWITCH_EPISODES = (5, 10, 15)
DOUBLE_SWITCH_PAIRS = list(itertools.combinations(DOUBLE_SWITCH_EPISODES, 2))
TEST2_EPISODES_PER_DEPLOYMENT = 20

# Test 3 delay: how many timesteps into the next episode the withheld reward
# signal is presented to the network. It still pauses movement for
# reward_process_steps steps once presented -- just delayed to here.
DELAYED_FEEDBACK_DELAY_STEPS = 3

# ---------------------------------------------------------------------------
# Reference fitness per post-evolution test, as ``(expected, optimum, ceiling)``
# keyed by the RunRecord field the test's result is stored in. All three are
# exact: they follow from the run-credit scoring in _transition_episode applied to
# each test's schedules, assuming an agent always reaches the arm it aims for.
# They are reported alongside the measured means in the study summary, so a result
# can be read against what perfect play would score rather than in isolation. Each
# is a mean per deployment -- the same quantity the tests report, and the same
# quantity training optimises (gym_runner averages a genome's deployments).
#
#   "expected" is the natural reward-driven adaptive strategy: win-stay/lose-shift.
#     It probes one arm on a deployment's first episode, stays while the high
#     reward keeps coming, and switches on the first low reward. The network is
#     reset between deployments and is deterministic, so it necessarily probes the
#     SAME arm first every time -- starting correctly on half the deployments and
#     wrongly on the other half. In Tests 1/3 that is 18.0 vs 16.0, hence a mean of
#     17.0 rather than 18.0; in Test 2 it is 36.0 vs 34.0, hence 35.0. This is the
#     figure a genuinely adaptive network should be compared against.
#
#   "optimum" is the best of ALL deterministic policies that read reward history,
#     found by exhaustive game-tree search over the consistent-deployment sets. It
#     beats win-stay/lose-shift by exploiting the bounded switch window: every
#     switch episode is <= 6, so a long enough run of high rewards proves the swap
#     has already happened and the agent can move across without paying a probe.
#     A score above "expected" is therefore not impossible -- but a score at or
#     above "optimum" means something is wrong with the measurement.
#
#   "ceiling" is a clairvoyant agent that knows which arm is high on every episode,
#     including immediately after a switch. Nothing that has to *detect* switches
#     from feedback can reach it, since detecting a switch costs a probe episode.
#
# Test 3 matches Test 1: the withheld reward arrives DELAYED_FEEDBACK_DELAY_STEPS
# steps into the next episode, still well before the agent reaches the junction
# (the stem alone takes at least 8 steps at full speed), so the delay changes when
# an ideal agent learns the outcome but not which arm it then picks.
#
# Test 4 is the informative exception. With the feedback pinned to 1.0 a
# win-stay/lose-shift agent never observes a loss, so it never switches arms, and
# the single-arm rule in step() zeroes the whole deployment -- expected fitness
# 0.0. Its optimum, 14.7, is the best FIXED open-loop arm sequence (switch arms
# once at a fixed episode, ignoring feedback entirely), found by brute force over
# all 2^10 sequences: with a constant signal every deployment produces an identical
# observation history, so no policy can be anything but open-loop. A Test 4 score
# well above 0 therefore means the agent's arm switching is NOT driven by reward
# content -- an internal timer or oscillation would land in this range.
EXPECTED_ADAPTIVE_FITNESS = {
    "all_switch": (17.0, 17.4, 20.0),
    "double_switch": (35.0, 226 / 6, 40.0),   # Test 2 optimum is 226/6 = 37.6667
    "delayed_feedback": (17.0, 17.4, 20.0),
    "constant_feedback": (0.0, 14.7, 20.0),
}

# Candidate switch_reward_episode values for a sampled training schedule. A
# switch_reward_episode of s is the last episode before the reward arms swap, so
# the swap takes effect on the (s + 2)-th episode (1-indexed); the five values
# below therefore cover switches on the 4th through 8th episode inclusive.
SWITCH_EPISODE_CHOICES = (2, 3, 4, 5, 6)


def sample_generation_schedule(rng=random):
    """
    Sample the four training deployments for a single genome evaluation.

    A fresh schedule is drawn for every genome (see the ``schedule_sampler``
    hook in run_adaptive_es / run_adaptive_des, which both invoke this once per
    genome) so evolution can't settle on a single fixed deployment set. The name
    is historical -- despite it, sampling is now per genome, not per generation.
    The result has the same ``(start_side, switch_reward_episode)`` format
    as DEPLOYMENT_SCHEDULE, with:

      - start sides evenly split -- exactly two "left" and two "right", with only
        their order among the four deployments varying from generation to
        generation;
      - four *distinct* switch episodes drawn uniformly from the 4th..8th episode
        (inclusive), so no two of the four deployments switch on the same episode.

    ``rng`` is the random source (defaults to the ``random`` module) and is
    sampled once per genome, in the worker process that evaluates it (or the
    parent process when running single-threaded).
    """
    # Exactly two left and two right so the start side is always balanced across
    # the four deployments; only their order varies between genomes.
    sides = ["left", "left", "right", "right"]
    rng.shuffle(sides)

    # Four unique switch episodes uniformly chosen from the five candidates.
    switches = rng.sample(SWITCH_EPISODE_CHOICES, 4)

    return list(zip(sides, switches))


def all_single_switch_schedules():
    """
    Test 1 & Test 3 schedule: every (start side, single switch episode) pair over
    the training range. 2 sides x len(SWITCH_EPISODE_CHOICES) = 10 deployments,
    each 10 episodes with a single reward switch -- an exhaustive sweep of the
    schedules evolution samples from (not a held-out generalization test).
    """
    return [(side, switch)
            for side in ("left", "right")
            for switch in SWITCH_EPISODE_CHOICES]


def double_switch_schedules():
    """
    Test 2 schedule: every (start side, switch-episode pair) combination. Each
    deployment runs TEST2_EPISODES_PER_DEPLOYMENT episodes and switches the reward
    arms at BOTH episodes of its pair (drawn from DOUBLE_SWITCH_EPISODES), so the
    reward returns to the starting arm after the second switch. 2 sides x
    len(DOUBLE_SWITCH_PAIRS) = 6 deployments.
    """
    return [(side, pair)
            for side in ("left", "right")
            for pair in DOUBLE_SWITCH_PAIRS]


class MultiTMazeEnv(gym.Env):
    def __init__(self, num_turns=1):
        super().__init__()
        self.num_turns = num_turns
        self.ray_step = 0.05  # Distance increment for ray marching
        self.max_ray_distance = 2.0  # Maximum distance for each sensor ray
        self.reward_process_steps = 6 #TODO: make this an argument to be set in higher level code
        self.reward = 0.0
        # Episodes (0-indexed) AFTER which the reward arms swap this deployment.
        # A single-switch deployment holds one episode; Test 2 holds two.
        self.switch_episodes = set()
        self.reward_process_timestep = 0

        # Episodes per deployment: 10 for training and Tests 1/3/4; Test 2
        # (evaluate_double_switch) raises it to TEST2_EPISODES_PER_DEPLOYMENT.
        self.episodes_per_deployment = 10

        # Delayed-feedback mode (Test 3). When False (training and Tests 1/2) the
        # reward the network observes is presented immediately on reaching a goal;
        # when True it is withheld until DELAYED_FEEDBACK_DELAY_STEPS steps into
        # the next episode. See step(). pending_reward holds the queued reward and
        # _delayed_pending flags that one is awaiting presentation.
        self.delayed_feedback = False
        self.pending_reward = 0.0
        self._delayed_pending = False

        # Constant-feedback mode (Test 4). When True the reward value the network
        # observes is pinned to 1.0 for every step of a reward presentation, so it
        # says nothing about which arm was reached; fitness keeps being scored on
        # the true self.reward. See feedback_signal().
        self.constant_feedback = False

        # True exactly on the steps that present a reward to the network -- the
        # frozen reward-observation steps of either feedback mode. Only
        # constant-feedback mode reads it (see feedback_signal); it is maintained
        # in both modes so the two stay interchangeable.
        self._presenting_reward = False

        # The schedule reset() steps through, and the index into it for the next
        # reset(). It defaults to DEPLOYMENT_SCHEDULE, but training overrides it
        # every generation via set_training_schedule() with a freshly sampled
        # schedule (see sample_generation_schedule). The post-evolution test
        # evaluators (evaluate_all_single_switch / evaluate_double_switch /
        # evaluate_delayed_feedback / evaluate_constant_feedback) temporarily swap
        # in their own schedules and restore this afterwards. Each deployment of a
        # genome steps through the active schedule in order; call
        # reset_switch_history() to start over at deployment 1.
        self.active_schedule = DEPLOYMENT_SCHEDULE
        self._deployment_index = 0


    def reset(self, seed = None, options = None):

        start_side, switch_field = self.active_schedule[
            self._deployment_index % len(self.active_schedule)]
        self._deployment_index += 1

        self.maze_points, self.high_reward, self.low_rewards = self.generate_t_maze(start_side)

        # A deployment's switch field is either a single switch episode (an int)
        # or a collection of them (Test 2 switches twice); normalize to the set of
        # episodes AFTER which the reward arms swap.
        self.switch_episodes = ({switch_field} if isinstance(switch_field, int)
                                else set(switch_field))

        self.current_position = [0.0, 0.0]
        self.rotation = 0.0

        self.current_episode = 0
        self.episode_timestep = 0
        self.reward_process_timestep = 0

        # Delayed feedback (Test 3): no reward is queued at the start of a deployment.
        self.pending_reward = 0.0
        self._delayed_pending = False
        self._presenting_reward = False

        ob = self.observe()

        self.hit_wall = False
        self.last_reward = 0.0
        # Whether the first episode of the current same-reward run is still
        # uncredited (see the fitness calculation in step()).
        self._run_first_pending = False
        self.reward = 0.0

        # Physical arms ("left"/"right") the agent has reached during this
        # deployment, plus the running fitness emitted so far. If the agent
        # never navigates to at least two distinct sides across the whole
        # deployment, the accumulated fitness is zeroed on the terminating step
        # (see step()).
        self._sides_visited = set()
        self._deployment_reward_total = 0.0

        self.info = None

        return ob, self.info

    def reset_switch_history(self):
        """
        Restart the deployment schedule at the first variation.

        Call this at the start of each genome's deployment sequence so every
        genome is evaluated on the same four fixed deployments in order (a single
        env instance is reused for many genomes within a worker).
        """
        self._deployment_index = 0

    def set_training_schedule(self, schedule):
        """
        Replace the schedule reset() steps through and restart it at deployment 1.

        Used to apply a freshly sampled per-genome training schedule (see
        sample_generation_schedule) before a genome is evaluated. The
        post-evolution test evaluators (evaluate_all_single_switch /
        evaluate_double_switch / evaluate_delayed_feedback /
        evaluate_constant_feedback) set active_schedule directly instead; they
        must never run while training.
        """
        self.active_schedule = list(schedule)
        self._deployment_index = 0

    def step(self, action):
        """Advance the agent one step through the maze using the supplied action."""

        terminated, truncated = False, False

        # The agent is frozen whenever it is observing a reward signal. Both the
        # immediate path and the delayed path (Test 3) present the reward over
        # reward_process_steps frozen steps. The delayed path additionally freezes
        # the agent on the step it *begins* that presentation, which is
        # DELAYED_FEEDBACK_DELAY_STEPS steps into the episode -- computed here so
        # the agent does not take one extra move before pausing.
        begin_delayed = (self.delayed_feedback and self._delayed_pending
                         and self.reward_process_timestep == 0
                         and self.episode_timestep >= DELAYED_FEEDBACK_DELAY_STEPS)
        frozen = self.reward_process_timestep > 0 or begin_delayed

        if not frozen:
            # The action is a 3-tuple: [left, forward, right].
            left, forward, right = action

            # Rotate the agent by 17 degrees for each unit of right-minus-left.
            rotation_delta = 17.0 * (right - left)
            self.rotation = (self.rotation + rotation_delta) % 360.0

            # Move forward in the new heading by forward/8 units.
            old_position = self.current_position.copy()
            heading_x = math.sin(math.radians(self.rotation))
            heading_y = math.cos(math.radians(self.rotation))
            self.current_position = [
                self.current_position[0] + heading_x * (forward / 8.0),
                self.current_position[1] + heading_y * (forward / 8.0),
            ] #TODO: Normalize the forward so that the robot can move backwards or stop when forward is 0.5 or less

            # If the new position leaves the maze path, undo the move and end the episode.
            if not self.point_in_maze(self.current_position):
                self.current_position = old_position
                self.hit_wall = True
            else:
                self.hit_wall = False

        # check if reached end of maze
        high_reward = any(in_bounds(self.current_position, p) for p in self.high_reward)
        low_reward = any(in_bounds(self.current_position, p) for p in self.low_rewards)

        # Record which physical arm the agent reached this deployment. The
        # reward endpoints sit at x=-2 (left) and x=+2 (right) with the junction
        # at x=0, so the sign of the reached endpoint's x-coordinate names the
        # side. Accumulated across the whole deployment to detect agents that
        # only ever pick a single arm.
        if high_reward or low_reward:
            reached_point = self.high_reward[0] if high_reward else self.low_rewards[0]
            self._sides_visited.add("left" if reached_point[0] < 0 else "right")

        # Reward observation and episode transitions differ only in *when* the
        # reward is shown to the network; fitness scoring is identical (see
        # _transition_episode). Immediate mode shows it on reaching the goal;
        # delayed mode (Test 3) advances the episode first and shows it later.
        if self.delayed_feedback:
            ob, reward, terminated = self._step_delayed(high_reward, low_reward, begin_delayed)
        else:
            ob, reward, terminated = self._step_immediate(high_reward, low_reward)

        # Track the deployment's running fitness and, on the terminating step,
        # zero it out entirely if the agent only ever reached a single arm (or
        # never reached one). An agent that never navigates to at least two
        # distinct sides is not adapting to the reward switch and earns no
        # fitness, regardless of how much reward it accumulated on that one side.
        self._deployment_reward_total += reward
        if terminated and len(self._sides_visited) < 2:
            reward -= self._deployment_reward_total
            self._deployment_reward_total = 0.0

        return ob, reward, terminated, truncated, self.info

    def _step_immediate(self, high_reward, low_reward):
        """
        Immediate-feedback episode logic (training and Tests 1/2).

        On reaching a goal (or timing out at 30 navigation steps) the agent
        freezes and observes the reward for reward_process_steps steps, then the
        episode ends. Returns ``(ob, reward, terminated)``.
        """
        reward = 0.0
        terminated = False

        if self.episode_timestep >= 50 or high_reward or low_reward:
            # Observe the reward for reward_process_steps frozen steps.
            if self.reward_process_timestep < self.reward_process_steps:
                self.reward = 1.0 if high_reward else 0.1 if low_reward else 0.0
                self._presenting_reward = True
                ob = self.observe()
                self.reward_process_timestep += 1

            # End the episode once the reward has been fully observed.
            if self.reward_process_timestep >= self.reward_process_steps:
                reward, terminated = self._transition_episode()
        else:
            ob = self.observe()
            self.episode_timestep += 1

        return ob, reward, terminated

    def _step_delayed(self, high_reward, low_reward, begin_delayed):
        """
        Delayed-feedback episode logic (Test 3).

        On reaching a goal (or timing out) the episode ends IMMEDIATELY -- the
        agent advances to the next episode and starts navigating it -- while the
        reward it earned is queued and only presented (freezing the agent for
        reward_process_steps steps) DELAYED_FEEDBACK_DELAY_STEPS steps into that
        next episode. Fitness is scored exactly as in immediate mode; only the
        reward SIGNAL the network observes is delayed. A goal takes far more than
        DELAYED_FEEDBACK_DELAY_STEPS steps to reach, so a queued reward is always
        presented before the next goal; the final episode's reward has no next
        episode and is simply never presented. Returns ``(ob, reward, terminated)``.
        """
        reward = 0.0
        terminated = False

        if self.reward_process_timestep > 0:
            # Mid-presentation of a queued reward: stay frozen and keep showing it.
            self.reward = self.pending_reward
            self._presenting_reward = True
            ob = self.observe()
            self.reward_process_timestep += 1
            if self.reward_process_timestep >= self.reward_process_steps:
                self.reward_process_timestep = 0
                self.reward = 0.0
                self.pending_reward = 0.0
                self._presenting_reward = False
        elif begin_delayed:
            # DELAYED_FEEDBACK_DELAY_STEPS into the new episode: begin presenting
            # the queued reward (this frozen step is the first of the presentation).
            self._delayed_pending = False
            self.reward = self.pending_reward
            self._presenting_reward = True
            ob = self.observe()
            self.reward_process_timestep = 1
        elif self.episode_timestep >= 50 or high_reward or low_reward:
            # Goal or timeout: score and advance to the next episode right away,
            # queuing the reward for delayed presentation. self.reward is set only
            # so _transition_episode can score the run; _transition_episode clears
            # it before the post-advance observation, so the network does not see
            # the reward yet.
            self.reward = 1.0 if high_reward else 0.1 if low_reward else 0.0
            queued = self.reward
            reward, terminated = self._transition_episode()
            self.pending_reward = queued
            self._delayed_pending = True
            ob = self.observe()
        else:
            # Ordinary navigation step.
            ob = self.observe()
            self.episode_timestep += 1

        return ob, reward, terminated

    def _transition_episode(self):
        """
        End the current episode and advance to the next.

        Scores the episode (the run-length credit), flips the reward arms if this
        is a switch episode, resets the agent to the start, and increments the
        episode counter. Reads ``self.reward`` (the reward reached this episode)
        and clears it to 0. Returns ``(reward_delta, terminated)`` where
        ``terminated`` is True once the deployment's last episode has ended.
        """
        # end deployment if appropriate
        terminated = self.current_episode >= self.episodes_per_deployment - 1
        reward = 0.0

        # reset agent
        self.current_position = [0.0, 0.0]
        self.rotation = 0.0

        # Fitness: score each reward that is part of a run of two or more
        # consecutive episodes with the same reward type. A high reward in such a
        # run is worth 2.0; a low reward is worth 1. Every member of the run
        # counts, including the first (applied retroactively once the run reaches
        # length two); an isolated reward -- a run of length one -- scores
        # nothing, which keeps the single probe episode that detects a reward
        # switch free of penalty.
        reward_value = (2.0 if self.reward == 1.0
                        else 1 if self.reward == 0.1
                        else 0.0)
        if self.reward != 0.0 and self.reward == self.last_reward:
            # Continuing a run: credit this episode, plus the run's first
            # episode if it is still pending.
            if self._run_first_pending:
                reward += reward_value
                self._run_first_pending = False
            reward += reward_value
        else:
            # Start of a new run (or no reward reached this episode).
            # Defer crediting until the run is known to reach length two.
            self._run_first_pending = self.reward != 0.0
        self.last_reward = self.reward
        self.reward = 0.0
        # The presentation (if any) ended with this episode, so any observation
        # taken after this point is an ordinary navigation observation.
        self._presenting_reward = False

        # switch reward sides if appropriate (always flip to the opposite arm)
        if self.current_episode in self.switch_episodes:
            self.high_reward, self.low_rewards = self.low_rewards, self.high_reward #TODO: This will not work as intended if using longer than a single t-maze

        self.current_episode += 1
        self.episode_timestep = 0
        self.reward_process_timestep = 0
        return reward, terminated


    def observe(self):
        """Return a 7-element observation with five distance sensors, one mode value, and a reward value."""
        ob = []

        # Cast five rays from the agent's current position.
        # The middle sensor uses angle 0, which corresponds to the +y direction.
        for angle_offset in (-90.0, -45.0, 0.0, 45.0, 90.0):
            ray_angle = self.rotation + angle_offset
            ray_x = math.sin(math.radians(ray_angle))
            ray_y = math.cos(math.radians(ray_angle))

            # Ray march outward in 0.05-unit increments until the ray hits the maze.
            sensed_distance = self.max_ray_distance
            for distance in [i * self.ray_step for i in range(int(self.max_ray_distance / self.ray_step) + 1)]:
                sample_x = self.current_position[0] + ray_x * distance
                sample_y = self.current_position[1] + ray_y * distance
                if not self.point_in_maze([sample_x, sample_y]):
                    sensed_distance = distance
                    break

            ob.append(sensed_distance)

        # Append the reward value the network is allowed to see this step.
        ob.append(self.feedback_signal())

        return ob

    def feedback_signal(self):
        """
        Return the reward value the network observes this step.

        Normally this is just the reward reached (``self.reward``): 0.0 while
        navigating, and the true 1.0 / 0.1 / 0.0 throughout a reward presentation.

        Under constant-feedback mode (Test 4) every step of a presentation reports
        1.0 instead, whichever arm was actually reached -- so a high reward, a low
        reward and a timeout are indistinguishable and the signal carries no
        information beyond its own arrival. Fitness scoring is untouched:
        _transition_episode reads the true ``self.reward``, so the test measures
        how much of an agent's adaptation really depends on reward content.
        """
        if self.constant_feedback and self._presenting_reward:
            return 1.0
        return self.reward

    def point_in_maze(self, point):
        """Return True when the point lies inside any maze-path square."""
        return any(in_bounds(point, maze_point) for maze_point in self.maze_points)

    def generate_t_maze(self, start_side):
        """
        Returns a single, static T-maze with the high reward on ``start_side``.

        The maze is a stem rising from the origin to a T-junction, with a left
        arm and a right arm. The reward layout is fixed by ``start_side``:
        ``"left"`` puts the high reward on the left arm (low on the right), and
        ``"right"`` puts it on the right arm (low on the left).

        Returns:
            maze_points : list[(x, y)]
                The centers of the seven 1x1 blocks making up the maze.

            high_reward : list[(x, y)]
                A single coordinate for the high-reward block (coincides with
                one arm's endpoint).

            low_rewards : list[(x, y)]
                A single coordinate for the low-reward block on the opposite arm.
        """

        # Stem (0,0)->(0,1)->(0,2), then left arm and right arm off the junction.
        maze_points = [
            (0, 0),
            (0, 1),
            (0, 2),
            (-1, 2),
            (-2, 2),
            (1, 2),
            (2, 2),
        ]

        left_end = (-2, 2)
        right_end = (2, 2)

        if start_side == "left":
            # High reward on the left, low reward on the right.
            high_reward = [left_end]
            low_rewards = [right_end]
        else:
            # High reward on the right, low reward on the left.
            high_reward = [right_end]
            low_rewards = [left_end]

        return maze_points, high_reward, low_rewards


    def visualize_maze(self, maze_points, goal_endpoint=None):
        """
        Visualizes maze points as 1x1 white squares centered at each coordinate.
        """

        xs = [p[0] for p in maze_points]
        ys = [p[1] for p in maze_points]

        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2

        fig, ax = plt.subplots(figsize=(8, 8))

        # "Gaussian space" / coordinate background
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)

        # Draw each maze point as a 1x1 square centered on its coordinate
        for x, y in maze_points:
            square = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="white",
                edgecolor="black"
            )
            ax.add_patch(square)

        # Mark start
        ax.scatter(0, 0, s=100, marker="o", label="Start")

        # Mark goal, if given
        if goal_endpoint is not None:
            ax.scatter(
                goal_endpoint[0],
                goal_endpoint[1],
                s=120,
                marker="*",
                label="Goal"
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Generated Multiple T-Maze")
        ax.legend()

        plt.show()

def in_bounds(current_position, maze_point):
    """
    Check if the current position is within the bounds (of +-0.5 in x and y directions) of the given maze point.

    Returns T/F
    """
    return (
        maze_point[0] - 0.5 < current_position[0] < maze_point[0] + 0.5
        and maze_point[1] - 0.5 < current_position[1] < maze_point[1] + 0.5
    )


def draw_tmaze_frame(state, ax, title):
    """Render a single captured T-maze state onto ``ax`` (one animation frame)."""
    ax.clear()

    maze_points = state["maze"]
    xs = [p[0] for p in maze_points]
    ys = [p[1] for p in maze_points]
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.35)

    # Maze blocks.
    for x, y in maze_points:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white", edgecolor="black"))

    # Reward locations: green star = high reward, orange star = low reward.
    # These swap arms mid-deployment, so watching the agent track the green star
    # is what verifies it adapts after the switch.
    for x, y in state["high_reward"]:
        ax.scatter(x, y, color="green", s=220, marker="*", zorder=3, label="High reward")
    for x, y in state["low_rewards"]:
        ax.scatter(x, y, color="orange", s=140, marker="*", zorder=3, label="Low reward")

    # Agent position and heading.
    px, py = state["current_position"]
    ax.scatter(px, py, color="blue", s=120, zorder=4, label="Agent")
    heading_x = math.sin(math.radians(state["rotation"]))
    heading_y = math.cos(math.radians(state["rotation"]))
    ax.plot([px, px + heading_x * 0.5], [py, py + heading_y * 0.5],
            color="red", linewidth=2, zorder=4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # Small enough that a two-line title identifying the test and deployment
    # still fits the figure width; a clipped title makes a folder of gifs
    # impossible to tell apart.
    ax.set_title(f"{title} — episode {state['episode']}", fontsize=10)
    ax.legend(loc="upper right")


def _run_eval_schedule(net, env, schedule, episodes_per_deployment=10,
                       delayed_feedback=False, constant_feedback=False,
                       max_steps=1000):
    """
    Roll ``net`` out on every deployment in ``schedule`` and return the list of
    per-deployment total fitnesses.

    Post-evolution only: this configures the env for the requested test
    (schedule, episodes-per-deployment, delayed feedback, constant feedback),
    runs one deployment per schedule entry with one activation per step (matching
    how genomes are evaluated during training), resets the network between
    deployments, and restores the env's previous configuration before returning
    so a shared env instance stays safe to reuse for further evolution.
    """
    prev_schedule = env.active_schedule
    prev_episodes = env.episodes_per_deployment
    prev_delayed = env.delayed_feedback
    prev_constant = env.constant_feedback

    env.active_schedule = list(schedule)
    env.episodes_per_deployment = episodes_per_deployment
    env.delayed_feedback = delayed_feedback
    env.constant_feedback = constant_feedback
    env._deployment_index = 0
    try:
        fitnesses = []
        for _ in range(len(schedule)):
            ob, _ = env.reset()
            net.reset()

            total_reward = 0.0
            for _ in range(max_steps):
                ob, reward, terminated, truncated, _ = env.step(net.activate(ob))
                total_reward += reward
                if terminated or truncated:
                    break

            fitnesses.append(total_reward)
    finally:
        env.active_schedule = prev_schedule
        env.episodes_per_deployment = prev_episodes
        env.delayed_feedback = prev_delayed
        env.constant_feedback = prev_constant
        env._deployment_index = 0

    return fitnesses


def evaluate_all_single_switch(net, env, max_steps=1000):
    """
    Test 1: sweep every training-range single-switch schedule.

    Runs one 10-episode, single-switch deployment for each (start side, switch
    episode) combination the genome could have trained on (see
    all_single_switch_schedules) and returns the per-deployment fitnesses. Not a
    generalization test -- it measures behaviour across the full set of
    deployments evolution draws from.
    """
    return _run_eval_schedule(net, env, all_single_switch_schedules(),
                              episodes_per_deployment=10, delayed_feedback=False,
                              max_steps=max_steps)


def evaluate_double_switch(net, env, max_steps=2000):
    """
    Test 2: generalization to two reward switches per deployment.

    Runs one TEST2_EPISODES_PER_DEPLOYMENT-episode deployment for each (start
    side, switch-episode pair) combination (see double_switch_schedules), so the
    reward arms swap twice -- a schedule the genome never trained on -- and
    returns the per-deployment fitnesses.
    """
    return _run_eval_schedule(net, env, double_switch_schedules(),
                              episodes_per_deployment=TEST2_EPISODES_PER_DEPLOYMENT,
                              delayed_feedback=False, max_steps=max_steps)


def evaluate_delayed_feedback(net, env, max_steps=1000):
    """
    Test 3: Test 1's schedules with delayed reward feedback.

    Runs the same all-single-switch sweep as evaluate_all_single_switch, but with
    delayed feedback enabled: on reaching a goal the agent advances to the next
    episode immediately and only observes the reward DELAYED_FEEDBACK_DELAY_STEPS
    steps later (see MultiTMazeEnv.step). Returns the per-deployment fitnesses.
    """
    return _run_eval_schedule(net, env, all_single_switch_schedules(),
                              episodes_per_deployment=10, delayed_feedback=True,
                              max_steps=max_steps)


def evaluate_constant_feedback(net, env, max_steps=1000):
    """
    Test 4: Test 1's schedules with an uninformative reward signal.

    Runs the same all-single-switch sweep as evaluate_all_single_switch, with the
    same immediate presentation timing, but every step of every reward
    presentation shows the network a feedback value of 1.0 regardless of the arm
    actually reached (see MultiTMazeEnv.feedback_signal). Fitness is still scored
    on the true rewards, so the result is what the agent earns once its feedback
    channel can no longer distinguish the high-reward arm from the low-reward one.
    Returns the per-deployment fitnesses.
    """
    return _run_eval_schedule(net, env, all_single_switch_schedules(),
                              episodes_per_deployment=10, delayed_feedback=False,
                              constant_feedback=True, max_steps=max_steps)


# ---------------------------------------------------------------------------
# Recording what a deployment actually did, episode by episode.
#
# A deployment's behaviour is summarised by the sequence of rewards it collected
# -- "HHHLHHHHHH" and the like -- which is what makes a folder of gifs readable
# without watching them. The three symbols are exhaustive: an episode ends
# either on the high-reward arm (1.0), on the low-reward one (0.1), or with the
# agent still in the maze after 30 navigation steps, which scores nothing (see
# MultiTMazeEnv._transition_episode).
# ---------------------------------------------------------------------------
HIGH_REWARD_SYMBOL = "H"
LOW_REWARD_SYMBOL = "L"
NO_REWARD_SYMBOL = "-"

# Arms, written in lower case so an arm sequence ("llllrrrrrr") can never be
# misread as a reward sequence. A timed-out episode reached neither arm.
SIDE_SYMBOLS = {"left": "l", "right": "r"}
NO_SIDE_SYMBOL = "."


def reward_symbol(reward):
    """
    The H/L/- symbol for the reward an episode ended on.

    Compared exactly against the two literals step() assigns, the same way
    _transition_episode scores them.
    """
    if reward == 1.0:
        return HIGH_REWARD_SYMBOL
    if reward == 0.1:
        return LOW_REWARD_SYMBOL
    return NO_REWARD_SYMBOL


def reward_side(endpoints):
    """
    ``"left"`` or ``"right"`` for one of the env's reward endpoint lists.

    The two endpoints sit at x=-2 and x=+2 with the junction at x=0, so the sign
    of the x-coordinate names the arm (the same rule step() uses to record which
    sides were visited).
    """
    return "left" if endpoints[0][0] < 0 else "right"


@dataclass
class EpisodeOutcome:
    """
    How one episode of a rolled-out deployment ended.

    ``symbol`` is the character the episode contributes to the deployment's
    reward sequence; ``high_side`` is the arm holding the high reward DURING the
    episode (the arms swap only afterwards, inside _transition_episode); and
    ``chosen_side`` is the arm the agent actually walked into, None when it timed
    out without reaching either. ``end_step`` is the 1-based index of the step
    the episode ended on, which is also its frame number in a gif of the same
    deployment, so a symbol can be traced back to the moment that produced it.
    """
    episode: int                # 1-based index within the deployment
    reward: float               # the true reward reached: 1.0, 0.1 or 0.0
    symbol: str                 # reward_symbol(reward)
    high_side: str              # arm the high reward sat on this episode
    chosen_side: str = None     # arm reached, or None on a timeout
    end_step: int = 0

    @property
    def side_symbol(self):
        """The l/r/. symbol for the arm this episode's agent walked into."""
        return SIDE_SYMBOLS.get(self.chosen_side, NO_SIDE_SYMBOL)

    @property
    def high_side_symbol(self):
        """The l/r symbol for the arm that held the high reward this episode."""
        return SIDE_SYMBOLS.get(self.high_side, NO_SIDE_SYMBOL)


def reward_sequence(episodes):
    """The deployment's reward sequence, e.g. ``"HHHLHHHHHH"``."""
    return "".join(outcome.symbol for outcome in episodes)


def chosen_side_sequence(episodes):
    """The arms the agent walked into, one symbol per episode, e.g. ``"llllrrrrrr"``."""
    return "".join(outcome.side_symbol for outcome in episodes)


def high_side_sequence(episodes):
    """The arms the high reward sat on, one symbol per episode."""
    return "".join(outcome.high_side_symbol for outcome in episodes)


def roll_out_deployment(net, env, max_steps=1000, schedule=None,
                        episodes_per_deployment=10, delayed_feedback=False,
                        constant_feedback=False, deployment_index=0,
                        capture_states=True):
    """
    Roll ``net`` out on a single deployment and return
    ``(states, total_reward, episodes)``.

    Uses one activation per step, matching how the adaptive networks are
    evaluated during evolution, and resets the network first, so the deployment
    starts from unplastic weights exactly as it does inside a test's
    evaluate_ function. ``total_reward`` is therefore that test's score for this
    deployment, including the single-arm rule that zeroes a deployment the agent
    never switched arms in (see MultiTMazeEnv.step).

    ``states`` is one captured frame per step -- the maze, the agent's pose and
    the (switching) reward locations -- for draw_tmaze_frame to render; pass
    ``capture_states=False`` to skip building it when only the outcome is wanted.
    ``episodes`` is one EpisodeOutcome per episode that finished, in order.

    An episode's end is detected by env.current_episode changing, which
    _transition_episode does on the same step it stores the episode's reward in
    env.last_reward -- so reading last_reward straight after the step gives the
    reward that episode ended on, in both feedback modes. The high arm has to be
    read BEFORE the step instead: a switch episode swaps the arms inside that
    same call.

    ``schedule`` defaults to Test 1's all-single-switch sweep, and
    ``deployment_index`` picks which of its entries to run (0 = the first), so
    looping over a test's schedule covers every deployment it scores. The
    remaining arguments configure the env the way that test's evaluate_ function
    does -- ``episodes_per_deployment`` plus the ``delayed_feedback`` (Test 3)
    and ``constant_feedback`` (Test 4) flags. See TEST_GIF_SPECS, which pairs
    each test with its settings.

    The env's previous configuration is restored before returning so a shared
    instance stays safe to reuse.
    """
    if schedule is None:
        schedule = all_single_switch_schedules()

    prev_schedule = env.active_schedule
    prev_episodes = env.episodes_per_deployment
    prev_delayed = env.delayed_feedback
    prev_constant = env.constant_feedback

    env.active_schedule = list(schedule)
    env.episodes_per_deployment = episodes_per_deployment
    env.delayed_feedback = delayed_feedback
    env.constant_feedback = constant_feedback
    # reset() reads active_schedule[_deployment_index] and then advances it, so
    # seeding the index here is what selects the deployment to run.
    env._deployment_index = deployment_index
    try:
        ob, _ = env.reset()
        net.reset()

        states = []
        episodes = []
        total_reward = 0.0
        for step_number in range(1, max_steps + 1):
            # Both read before stepping: the step that ends a switch episode
            # also swaps the arms, and it advances the episode counter.
            high_side = reward_side(env.high_reward)
            episode_index = env.current_episode

            ob, reward, terminated, truncated, _ = env.step(net.activate(ob))
            total_reward += reward

            if env.current_episode != episode_index:
                symbol = reward_symbol(env.last_reward)
                if symbol == HIGH_REWARD_SYMBOL:
                    chosen_side = high_side
                elif symbol == LOW_REWARD_SYMBOL:
                    chosen_side = "right" if high_side == "left" else "left"
                else:
                    chosen_side = None
                episodes.append(EpisodeOutcome(
                    episode=episode_index + 1, reward=env.last_reward, symbol=symbol,
                    high_side=high_side, chosen_side=chosen_side,
                    end_step=step_number))

            if capture_states:
                states.append({
                    "maze": list(env.maze_points),
                    "current_position": list(env.current_position),
                    "rotation": env.rotation,
                    "high_reward": list(env.high_reward),
                    "low_rewards": list(env.low_rewards),
                    "episode": env.current_episode,
                })
            if terminated or truncated:
                break
    finally:
        # Always restore the env's previous configuration so a shared instance
        # stays safe to reuse for the next run's evolution.
        env.active_schedule = prev_schedule
        env.episodes_per_deployment = prev_episodes
        env.delayed_feedback = prev_delayed
        env.constant_feedback = prev_constant
        env._deployment_index = 0

    return states, total_reward, episodes


def save_run_gif(net, env, gif_path, title, fps=4, max_steps=1000,
                 schedule=None, episodes_per_deployment=10, delayed_feedback=False,
                 constant_feedback=False, deployment_index=0):
    """
    Roll out ``net`` on a single deployment and save a gif of the run.

    Every step is a frame, so the whole deployment -- including the behaviour
    across the reward switch(es) -- is visible. The rollout itself is
    roll_out_deployment's (see it for the arguments, all of which are passed
    straight through); this only renders it.

    Returns ``(total_reward, episodes)``: the deployment's fitness and the
    per-episode EpisodeOutcome list, so a caller filming a whole test can write
    down what each gif shows without re-running it.
    """
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)

    states, total_reward, episodes = roll_out_deployment(
        net, env, max_steps=max_steps, schedule=schedule,
        episodes_per_deployment=episodes_per_deployment,
        delayed_feedback=delayed_feedback, constant_feedback=constant_feedback,
        deployment_index=deployment_index)

    fig, ax = plt.subplots(figsize=(8, 8))
    try:
        from matplotlib.animation import FuncAnimation

        def update(frame):
            draw_tmaze_frame(states[frame], ax, title)
            return ax

        animation = FuncAnimation(fig, update, frames=len(states), interval=1000 / fps, blit=False)
        animation.save(gif_path, writer="pillow", fps=fps)
        print(f"Saved animation to {gif_path} "
              f"({len(states)} frames, deployment fitness {total_reward:.1f}, "
              f"rewards {reward_sequence(episodes)})")
    except Exception as exc:  # visualization is best-effort; fall back to a still
        fallback_path = gif_path.replace(".gif", ".png")
        if states:
            draw_tmaze_frame(states[-1], ax, title)
        fig.savefig(fallback_path)
        print(f"Could not save animation ({exc}); saved a still image to {fallback_path}")
    finally:
        plt.close(fig)

    return total_reward, episodes


# ---------------------------------------------------------------------------
# Filming a whole post-evolution test.
#
# Each entry pairs a test with the env configuration its evaluate_ function
# applies, so a gif made from it shows exactly the behaviour that test scored --
# same schedules, same episode count, same feedback mode, same step budget. The
# tests themselves are described at the top of this module.
# ---------------------------------------------------------------------------
TEST_GIF_SPECS = {
    "test1": {
        "short": "Test 1",
        "label": "Test 1 (all single-switch schedules)",
        "schedules": all_single_switch_schedules,
        "episodes_per_deployment": 10,
        "delayed_feedback": False,
        "constant_feedback": False,
        "max_steps": 1000,
    },
    "test2": {
        "short": "Test 2",
        "label": "Test 2 (double-switch generalization)",
        "schedules": double_switch_schedules,
        "episodes_per_deployment": TEST2_EPISODES_PER_DEPLOYMENT,
        "delayed_feedback": False,
        "constant_feedback": False,
        "max_steps": 2000,
    },
    "test3": {
        "short": "Test 3",
        "label": "Test 3 (delayed reward feedback)",
        "schedules": all_single_switch_schedules,
        "episodes_per_deployment": 10,
        "delayed_feedback": True,
        "constant_feedback": False,
        "max_steps": 1000,
    },
    "test4": {
        "short": "Test 4",
        "label": "Test 4 (constant reward feedback)",
        "schedules": all_single_switch_schedules,
        "episodes_per_deployment": 10,
        "delayed_feedback": False,
        "constant_feedback": True,
        "max_steps": 1000,
    },
}


def describe_deployment(entry):
    """
    ``(human description, filename slug)`` for one schedule entry.

    An entry is ``(start side, switch_reward_episode)`` for the single-switch
    tests and ``(start side, (episode, episode))`` for Test 2's double switches,
    so both are rendered from the same code.
    """
    side, switch_field = entry
    switches = (switch_field,) if isinstance(switch_field, int) else tuple(switch_field)
    listed = "/".join(str(s) for s in switches)
    switch_word = "switch" if len(switches) == 1 else "switches"
    return (f"start {side}, {switch_word} after episode {listed}",
            f"{side}_switch{'-'.join(str(s) for s in switches)}")


@dataclass
class DeploymentRecord:
    """
    One filmed deployment: which test and schedule entry it came from, the gif
    it was written to, the fitness it scored, and how each of its episodes ended.

    ``gif_path`` is None for a deployment that was rolled out without filming
    (see save_test_gifs' ``film`` argument), so the reward sequences can be
    collected on their own.
    """
    test: str                           # TEST_GIF_SPECS key
    index: int                          # 1-based position in that test's schedule
    entry: tuple                        # the schedule entry: (start side, switch field)
    description: str                    # describe_deployment's human text
    fitness: float                      # the deployment's score, as the test counts it
    episodes: List[EpisodeOutcome] = field(default_factory=list)
    gif_path: str = None

    @property
    def rewards(self):
        """The deployment's reward sequence, e.g. ``"HHHLHHHHHH"``."""
        return reward_sequence(self.episodes)

    @property
    def chosen_arms(self):
        """The arms the agent walked into, one symbol per episode."""
        return chosen_side_sequence(self.episodes)

    @property
    def high_arms(self):
        """The arms the high reward sat on, one symbol per episode."""
        return high_side_sequence(self.episodes)


def save_test_gifs(net, env, out_dir, tests=("test1", "test2", "test4"),
                   title=None, prefix="", fps=4, film=True):
    """
    Run every deployment of each named post-evolution test and return one
    DeploymentRecord per deployment -- its fitness, its reward sequence, and the
    gif it was filmed into.

    One gif per deployment: Test 1 and Test 4 have 10 schedules each, Test 2 has
    6, so the default selection produces 26. ``net`` is reset before each one
    (inside roll_out_deployment), exactly as the tests themselves reset it
    between deployments, so no deployment inherits another's plastic weights.

    The env is driven exactly as each test's evaluate_ function drives it, so the
    fitnesses returned here match that test's per-deployment scores and their mean
    is the figure the study reports.

    ``tests`` names entries of TEST_GIF_SPECS; ``title`` heads every frame (the
    test and deployment are appended to it), and ``prefix`` is prepended to each
    filename. ``film=False`` skips the rendering -- by far the slow part -- and
    returns the same records with ``gif_path`` left None, which is how the reward
    sequences can be collected on their own.
    """
    os.makedirs(out_dir, exist_ok=True)
    written = []

    for test_name in tests:
        try:
            spec = TEST_GIF_SPECS[test_name]
        except KeyError:
            raise ValueError(
                f"Unknown test {test_name!r}: expected one of "
                f"{', '.join(sorted(TEST_GIF_SPECS))}.") from None

        schedule = spec["schedules"]()
        for index, entry in enumerate(schedule):
            described, slug = describe_deployment(entry)
            # Two lines, and the short test name: the full label plus the
            # deployment description on one line overruns the figure and gets
            # clipped at both ends.
            frame_title = (f"{title}\n" if title else "") + \
                          f"{spec['short']} deployment {index + 1}/{len(schedule)}: {described}"
            gif_path = os.path.join(
                out_dir, f"{prefix}{test_name}_deploy{index + 1:02d}_{slug}.gif")

            rollout = dict(max_steps=spec["max_steps"], schedule=schedule,
                           episodes_per_deployment=spec["episodes_per_deployment"],
                           delayed_feedback=spec["delayed_feedback"],
                           constant_feedback=spec["constant_feedback"],
                           deployment_index=index)
            if film:
                fitness, episodes = save_run_gif(net, env, gif_path, frame_title,
                                                 fps=fps, **rollout)
            else:
                _states, fitness, episodes = roll_out_deployment(
                    net, env, capture_states=False, **rollout)

            written.append(DeploymentRecord(
                test=test_name, index=index + 1, entry=entry, description=described,
                fitness=fitness, episodes=episodes,
                gif_path=gif_path if film else None))

    return written