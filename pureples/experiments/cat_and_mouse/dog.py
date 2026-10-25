import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import gymnasium as gym

class DogEnv(gym.Env):
    def __init__(self, num_turns=1):
        super().__init__()
        self.ray_step = 0.15  # Distance increment for ray marching
        self.max_ray_distance = 7.0  # Maximum distance for each sensor ray
        points = set()
        for x in range(-8, 8):
            for y in range(-8, 8):
                points.add((x, y))
        # Sorted, not just listed out of the set: a deployment is reproduced from
        # a seed alone (see _next_deployment_rng), and the picks that seed drives
        # are indices into this list, so its order has to be identical in every
        # process that rebuilds the env.
        self.default_room = sorted(points)
        self.max_timesteps = 1400
        self.caught = 0

        # Deployment scheduling. A schedule is a list of seeds, one per
        # deployment: reset() draws that deployment's room, robot pose and dog
        # placement from a private RNG seeded with the next entry, so two envs
        # handed the same schedule produce identical deployments in the same
        # order. Training uses this to put every genome of a generation through
        # the same rooms (see sample_generation_schedule in adaptive_des_dog.py).
        # Left as None -- playback, manual use, any other experiment -- reset()
        # draws from the global `random` module exactly as it always has.
        self._deployment_seeds = None
        self._deployment_index = 0

        # Shared by the 12 wall rays, the 12 dog-locator nodes and the 12 drive
        # nodes on the substrate's middle ring. All three rings are built with the
        # same ordering, so index i means the same bearing on every ring.
        self.sensor_bearings = np.arange(-180.0, 180.0, 30.0)

        # The direction each drive node asks the robot to accelerate in, as a unit
        # vector in the robot's own frame (x = its right, y = its forward). One row
        # per drive node, so this is 12x2.
        self.drive_axes = np.column_stack((
            np.sin(np.radians(self.sensor_bearings)),
            np.cos(np.radians(self.sensor_bearings)),
        ))

        # Wheel mounting bearings, clockwise from the heading. Each wheel pushes
        # tangentially, so its thrust acts 90 degrees off the bearing it is
        # mounted at. Three wheels, so this is 3x2.
        self.wheel_bearings = (240.0, 0.0, 120.0)
        self.wheel_axes = np.column_stack((
            np.sin(np.radians(np.asarray(self.wheel_bearings) + 90.0)),
            np.cos(np.radians(np.asarray(self.wheel_bearings) + 90.0)),
        ))
        self.max_wheel_thrust = 1.0

        self.linear_friction = 0.985

        # Pure translation demands zero net torque, and torque is just the raw sum
        # of the three thrusts, so the wheel thrusts are pinned to a unique
        # solution (see drive_force). Under that constraint, wheels limited to
        # +-1 reach 1.5 thrust units in EVERY direction but sqrt(3) only along the
        # three wheel-mount bearings -- and since the drive nodes sit every 30
        # degrees, taking the larger figure would leave 6 of the 12 nodes 15%
        # stronger than the other 6. The isotropic 1.5 keeps every node equal.
        self.max_drive_force = 1.5 * self.max_wheel_thrust
        # units/step^2 per thrust unit. Full drive is therefore
        # 1.5 * 0.0013333 = 0.002 units/step^2, and since velocity decays by
        # linear_friction every step the terminal speed is
        # 0.002 / (1 - 0.985) = 0.1333 units/step -- a third faster than the dog,
        # which closes at a constant 0.1 units/step.
        self.force_per_unit = 0.002 / self.max_drive_force

    def reset(self, seed = None, options = None):

        rng = self._next_deployment_rng()

        self.room = self.generate_room(rng)

        self.current_position = rng.choice(self.room)
        # The drive ring produces no torque, so this heading is fixed for the whole
        # episode. It is redrawn each reset so the robot still has to cope with an
        # arbitrary orientation of its sensor and drive rings relative to the room.
        self.rotation = rng.uniform(0, 360)  # Random initial rotation in degrees
        self.velocity = [0.0, 0.0]
        self.wheel_thrusts = np.zeros(3)

        self.dog_position = self.current_position
        self.dog_rotation = rng.uniform(0, 360)

        self.current_timestep = 0

        ob = self.observe()

        self.caught = 0

        self.info = None

        return ob, self.info

    def set_training_schedule(self, schedule):
        """
        Replace the deployment schedule and restart it at the first deployment.

        ``schedule`` is a sequence of ints, one seed per deployment; None or an
        empty sequence puts the env back to drawing unseeded deployments from the
        global RNG. Named to match the hook the gym runners probe for (see
        gym_runner.run_adaptive_des), so handing that runner a sampler is all it
        takes to drive this env.
        """
        self._deployment_seeds = list(schedule) if schedule else None
        self._deployment_index = 0

    def reset_switch_history(self):
        """
        Restart the deployment schedule at the first deployment.

        A worker reuses one env across many genomes, so this is what keeps each
        genome's deployments starting at the same place in the schedule. The
        runners call it (under this name, shared with the T-Maze env) whenever
        they are not applying a fresh schedule.
        """
        self._deployment_index = 0

    def _next_deployment_rng(self):
        """
        Return the RNG the next deployment should be drawn from.

        With a schedule set, that is a private generator seeded from the
        schedule's next entry -- deterministic, and independent of the global
        `random` stream so seeding a deployment cannot perturb anything else.
        Without one, it is the global `random` module, i.e. the original
        behaviour.
        """
        if not self._deployment_seeds:
            return random

        seed = self._deployment_seeds[self._deployment_index % len(self._deployment_seeds)]
        self._deployment_index += 1
        return random.Random(seed)

    def step(self, action):
        """Advance the agent one step through the room using the supplied action."""

        terminated, truncated = False, False
        reward = 0.0

        # The action is the 12 drive-ring activations. Turn them into a force in the
        # robot's own frame, plus the three wheel thrusts that produce it.
        (local_fx, local_fy), self.wheel_thrusts = self.drive_force(action)

        # The robot cannot rotate, so its heading stays whatever reset() drew; only
        # this timestep's force needs rotating out of the robot's frame and into
        # world coordinates. Momentum already banked is already in world coordinates.
        heading_x = math.sin(math.radians(self.rotation))
        heading_y = math.cos(math.radians(self.rotation))
        world_fx = local_fx * heading_y + local_fy * heading_x
        world_fy = local_fy * heading_y - local_fx * heading_x

        self.velocity = [
            self.linear_friction * self.velocity[0] + self.force_per_unit * world_fx,
            self.linear_friction * self.velocity[1] + self.force_per_unit * world_fy,
        ]

        old_position = list(self.current_position).copy()
        self.current_position = [
            self.current_position[0] + self.velocity[0],
            self.current_position[1] + self.velocity[1],
        ]

        # rotate dog towards robot
        dx = self.current_position[0] - self.dog_position[0]
        dy = self.current_position[1] - self.dog_position[1]
        self.dog_rotation = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

        # if the dog is in a wall, slow it down
        if not self.point_in_room(self.dog_position):
            dog_speed_scale = 0.035
        else:
            dog_speed_scale = 0.075

        # in the beginning of the episode, pause the dog
        if self.current_timestep < 90:
            dog_speed_scale = 0.01

        # move dog towards robot
        distance = math.hypot(dx, dy)
        if distance > 0:
            move_x = dx / distance * 0.5
            move_y = dy / distance * 0.5
            self.dog_position = [self.dog_position[0] + move_x * dog_speed_scale, self.dog_position[1] + move_y * dog_speed_scale]

        # figure out if the robot hit the wall
        if not self.point_in_room(self.current_position):
            self.current_position = old_position
            self.velocity = [0.0, 0.0]  # the impact kills the banked momentum
            hit_wall = True
        else:
            hit_wall = False
        
        # terminate if trial is over
        if self.current_timestep >= self.max_timesteps:
            terminated = True

        if distance < 0.1 and self.current_timestep > 90:  # if the dog is close to the robot
            self.caught += 1
            reward -= 50.0
        if self.caught >= 3:  # terminate if the dog has caught the robot three times
            terminated = True

        self.current_timestep += 1
        reward += 2.5  # reward for surviving a timestep

        # update observation
        ob = self.observe()

        reward += min(10, distance) / 2  # reward for being far from the dog

        if hit_wall:
            reward -= 15.0

        return ob, reward, terminated, truncated, self.info

    def drive_force(self, action):
        """
        Map the 12 drive-ring activations to a force in the robot's own frame.

        Each drive node stands for one bearing on the substrate's middle ring and
        asks the robot to accelerate that way, with its activation as the weight.
        Summing those votes fixes the direction, so nodes pulling against each
        other cancel: two opposed nodes at 1.0 sum to nothing and the robot coasts
        on whatever momentum it has banked.

        The magnitude is capped by two things at once -- the strongest single
        activation, and how much of the vote survives the cancellation. Taking the
        smaller of the two gives every case its natural reading: a lone node at 1.0
        drives at full power, a lone node at 0.4 drives at 40%, a broad weak lobe
        drives at the strength of its peak rather than its width, and no
        combination of votes can ever exceed full power.

        Returns ``(force, thrusts)``: the force in thrust units (magnitude at most
        max_drive_force) as a 2-vector in the robot's frame, and the three wheel
        thrusts producing it, each within +-max_wheel_thrust.
        """
        activations = np.asarray(action, dtype=float)

        # Direction: the activation-weighted sum of every node's own bearing.
        votes = activations @ self.drive_axes
        demand = float(np.hypot(votes[0], votes[1]))

        if demand <= 1e-12:
            # The votes cancelled exactly -- no power to the wheels, so the robot
            # coasts. This is also the resting case, since a substrate outputting
            # the same value on all 12 nodes sums to zero whatever that value is.
            return np.zeros(2), np.zeros(3)

        # Sigmoid outputs already sit in [0, 1]; the clip keeps the wheel thrusts
        # inside their limit even if the substrate activation is ever changed.
        strongest = float(np.clip(activations.max(), 0.0, 1.0))
        force = votes / demand * min(demand, strongest) * self.max_drive_force

        # Zero-torque inverse kinematics. Torque is the raw sum of the thrusts
        # (every wheel shares a lever arm), so demanding no rotation means the
        # thrusts must sum to zero -- which leaves exactly one solution for a given
        # force, since the only other freedom is adding a constant to all three and
        # that is pure torque. The 2/3 falls out of the three thrust axes
        # satisfying sum(axis_i axis_i^T) = 1.5 * I.
        thrusts = (2.0 / 3.0) * (self.wheel_axes @ force)

        return force, thrusts

    def observe(self):
        """Return 12 wall-distance rays followed by 12 dog-locator activations."""
        ob = []

        # Cast 12 rays from the agent's current position.
        # The middle sensor uses angle 0, which corresponds to the +y direction.
        for angle_offset in self.sensor_bearings:
            ray_angle = self.rotation + angle_offset
            ray_x = math.sin(math.radians(ray_angle))
            ray_y = math.cos(math.radians(ray_angle))

            # Ray march outward in 0.05-unit increments until the ray hits the maze.
            sensed_distance = self.max_ray_distance
            for distance in [i * self.ray_step for i in range(int(self.max_ray_distance / self.ray_step) + 1)]:
                sample_x = self.current_position[0] + ray_x * distance
                sample_y = self.current_position[1] + ray_y * distance
                if not self.point_in_room([sample_x, sample_y]):
                    sensed_distance = distance
                    break

            ob.append(sensed_distance)

        # calculate the distance between the robot and the dog
        dog_distance = math.hypot(
            self.current_position[0] - self.dog_position[0],
            self.current_position[1] - self.dog_position[1],
        )

        # calculate the direction in which the dog is relative to the robot's rotation
        dx = self.dog_position[0] - self.current_position[0]
        dy = self.dog_position[1] - self.current_position[1]
        dog_angle = math.atan2(dx, dy)
        robot_angle = math.radians(self.rotation)
        relative_angle = dog_angle - robot_angle

        # append the dog location input nodes' activations to ob
        for activation in self.dog_location_activations(relative_angle, dog_distance, 0.45, 3.0,
                                                        np.radians(self.sensor_bearings)):
            ob.append(activation)

        return ob

    def dog_location_activations(self, A, D, b, c, node_angles):
        """
        Returns the activation value for all 12 dog-locator nodes at once.
        A: dog bearing (radians), D: dog distance, b: bump width, c: activation cutoff distance.
        """

        NORM = 1.0 / (np.sqrt(2 * np.pi))
        # shortest angular distance between each node and the dog bearing, in [-pi, pi]
        delta = (node_angles - A + np.pi) % (2*np.pi) - np.pi
        # proximity factor: full strength when close, 0 at/beyond distance c
        proximity = c - min(D, c)
        # wrapped Gaussian bump across all nodes
        bump = (NORM / b) * np.exp(-(delta**2) / (2 * b**2))
        return proximity * bump
        
    def point_in_room(self, point):
        """Return True when the point lies inside any maze-path square."""
        return any(in_bounds(point, room_point) for room_point in self.room)

    def generate_room(self, rng=random):
        """
        Args:
            rng : the source of randomness for the block placements. Defaults to
                the global `random` module; reset() passes the deployment's own
                generator when a schedule is active.

        Returns:
            room_points : list[(x, y)]
                Every coordinate in the room, in a fixed order so the same seed
                gives the same room in every process (see default_room).
        """
        room_points = set(self.default_room)

        for _ in range(8):
            block_size = rng.choice([[1, 3],[2, 2], [3, 1], [1, 1], [5, 3], [1, 5], [5, 1], [3, 5]])
            block_start = rng.choice(self.default_room)
            for x in range(block_start[0], block_start[0] + block_size[0]):
                for y in range(block_start[1], block_start[1] + block_size[1]):
                    room_points.discard((x, y))

        return sorted(room_points)

    def visualize_step(self):
        """
        Visualizes the current room with the robot and dog positions.
        """

        room_points = getattr(self, "room", None)
        if not room_points:
            return

        xs = [p[0] for p in room_points]
        ys = [p[1] for p in room_points]

        min_x, max_x = min(xs) - 2, max(xs) + 2
        min_y, max_y = min(ys) - 2, max(ys) + 2

        fig, ax = plt.subplots(figsize=(8, 8))

        # "Gaussian space" / coordinate background
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.35)

        # Draw each room point as a 1x1 square centered on its coordinate
        for x, y in room_points:
            square = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor="white",
                edgecolor="black"
            )
            ax.add_patch(square)

        # Mark robot and dog positions
        current_position = getattr(self, "current_position", None)
        if current_position is not None:
            ax.scatter(current_position[0], current_position[1], color="blue", s=120, label="Robot")
            rotation = getattr(self, "rotation", 0.0)
            heading_x = math.sin(math.radians(rotation))
            heading_y = math.cos(math.radians(rotation))
            ax.plot(
                [current_position[0], current_position[0] + heading_x * 0.5],
                [current_position[1], current_position[1] + heading_y * 0.5],
                color="red",
                linewidth=2,
            )

        dog_position = getattr(self, "dog_position", None)
        if dog_position is not None:
            ax.scatter(dog_position[0], dog_position[1], color="red", s=120, label="Dog")

        goal_endpoint = getattr(self, "goal_endpoint", None)
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
        ax.set_title("Current Room")
        ax.legend()

        plt.show()

def in_bounds(current_position, maze_point):
    """
    Check if the current position is within the bounds (of +-0.5 in x and y directions) of the given maze point.
    
    Returns T/F
    """
    if current_position[0] >= maze_point[0] + 0.5 or current_position[1] >= maze_point[1] + 0.5 or current_position[0] <= maze_point[0] - 0.5 or current_position[1] <= maze_point[1] - 0.5:
        return False

    return True