"""
Generic runner for AI Gym - runs Neat, Hyperneat and ES-Hyperneat
"""

import os
import copy
import functools

import neat
import numpy as np
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.es_hyperneat.es_hyperneat import ESNetwork, AdaptiveESNetwork
from pureples.des_hyperneat import DESNetwork
from pureples.des_hyperneat import AdaptiveDESNetwork


# Environment built once per worker process (see _init_adaptive_des_worker).
_WORKER_ENV = None


def _init_adaptive_des_worker(env_factory):
    """
    Pool initializer: build one Gym environment per worker process.

    Gym environments are not safe to share across processes, so each worker
    constructs its own via the (picklable) ``env_factory`` callable instead of
    inheriting the parent's env. Building it once per worker avoids re-creating
    it for every genome.
    """
    global _WORKER_ENV
    _WORKER_ENV = env_factory()


def _eval_adaptive_des_genome(genome, config, substrate, params, max_steps, trials, env,
                              schedule_sampler=None, schedule=None):
    """
    Evaluate a single genome for Adaptive DES-HyperNEAT and return its fitness.

    Defined at module level (rather than as a closure) so it can be pickled and
    dispatched to worker processes by neat's ParallelEvaluator.

    The deployment schedule this genome is scored on comes from one of two
    optional arguments, both no-ops for envs without ``set_training_schedule``:

    ``schedule`` is an already-sampled schedule to apply as-is. The runner
    samples one per *generation* and binds it here, so every genome of that
    generation faces the same deployments and only the genomes differ.

    ``schedule_sampler`` is a zero-argument callable invoked here instead, so a
    *fresh* schedule is drawn per genome and each is scored on its own
    independently sampled deployments -- matching the Adaptive ES path (see
    _eval_adaptive_es_genome).

    With neither, the env's existing schedule is reused and only its deployment
    cycle is restarted (the original behaviour).
    """
    cppn = neat.nn.DesFeedForwardNetwork.create(genome, config)
    network = AdaptiveDESNetwork(substrate, cppn, params)
    net = network.create_phenotype_network(config)

    fitnesses = []

    # Apply this genome's deployments (set_training_schedule also restarts the
    # deployment cycle). Falling through to neither hook, just restart the env's
    # existing cycle so the genome's deployments start at the first variation.
    if schedule is not None and hasattr(env, "set_training_schedule"):
        env.set_training_schedule(schedule)
    elif schedule_sampler is not None and hasattr(env, "set_training_schedule"):
        env.set_training_schedule(schedule_sampler())
    else:
        reset_switch_history = getattr(env, "reset_switch_history", None)
        if callable(reset_switch_history):
            reset_switch_history()

    for _ in range(trials):
        ob = env.reset()[0]
        net.reset()

        total_reward = 0
        done = False

        for _ in range(max_steps):
            action = net.activate(ob)
            ob, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break

        fitnesses.append(total_reward)

    return float(np.array(fitnesses).mean())


def _eval_adaptive_des_genome_worker(genome, config, *, substrate, params, max_steps, trials,
                                     schedule_sampler=None, schedule=None):
    """
    ParallelEvaluator entry point: evaluate a genome using the per-worker env.

    ParallelEvaluator calls this with ``(genome, config)`` (the remaining args are
    bound via functools.partial). The env can't be pickled and shipped from the
    parent, so it's built once per worker in ``_init_adaptive_des_worker`` and read
    here from the module global. ``schedule_sampler`` is bound at partial time and
    invoked fresh for each genome; ``schedule`` is rebound by the runner once per
    generation and shared by every genome in it (see _eval_adaptive_des_genome).
    """
    return _eval_adaptive_des_genome(genome, config, substrate, params,
                                     max_steps, trials, _WORKER_ENV, schedule_sampler,
                                     schedule)


def _eval_adaptive_es_genome(genome, config, substrate, params, max_steps, trials, env,
                             schedule_sampler=None):
    """
    Evaluate a single genome for Adaptive ES-HyperNEAT and return its fitness.

    Defined at module level (rather than as a closure) so it can be pickled and
    dispatched to worker processes by neat's ParallelEvaluator.

    ``schedule_sampler`` is an optional zero-argument callable. When supplied
    (and the env exposes ``set_training_schedule``), a *fresh* deployment
    schedule is sampled for this genome and applied, so every genome is
    evaluated on its own independently sampled set of deployments (the Adaptive
    DES path uses the same per-genome scheme). When None, the env's existing
    schedule is reused and only its deployment cycle is restarted (the original
    behaviour).
    """
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    network = AdaptiveESNetwork(substrate, cppn, params)
    net = network.create_phenotype_network()

    fitnesses = []

    # Per-genome scheduling: with a sampler, draw a fresh set of deployments for
    # this genome and apply it (set_training_schedule also restarts the
    # deployment cycle). Without one, just restart the env's existing cycle so
    # the genome's deployments start at the first variation. Both are no-ops for
    # envs lacking the hooks (e.g. non-maze).
    if schedule_sampler is not None and hasattr(env, "set_training_schedule"):
        env.set_training_schedule(schedule_sampler())
    else:
        reset_switch_history = getattr(env, "reset_switch_history", None)
        if callable(reset_switch_history):
            reset_switch_history()

    for _ in range(trials):
        ob = env.reset()[0]
        net.reset()

        fitness = 0
        done = False

        for _ in range(max_steps):
            ob, reward, terminated, truncated, _ = env.step(net.activate(ob))
            done = terminated or truncated
            fitness += reward
            if done:
                break

        fitnesses.append(fitness)

    return float(np.array(fitnesses).mean())


def _eval_adaptive_es_genome_worker(genome, config, *, substrate, params, max_steps, trials,
                                    schedule_sampler=None):
    """
    ParallelEvaluator entry point: evaluate a genome using the per-worker env.

    ParallelEvaluator calls this with ``(genome, config)`` (the remaining args are
    bound via functools.partial). The env can't be pickled and shipped from the
    parent, so it's built once per worker in ``_init_adaptive_des_worker`` and read
    here from the module global. ``schedule_sampler`` is bound at partial time and
    invoked fresh for each genome (see _eval_adaptive_es_genome).
    """
    return _eval_adaptive_es_genome(genome, config, substrate, params,
                                    max_steps, trials, _WORKER_ENV, schedule_sampler)


def ini_pop(state, stats, config, output):
    """
    Initialize population attaching statistics reporter.
    """
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop

def ini_desPop(state, stats, config, output):
    """
    Initialize DES population attaching statistics reporter.
    """
    pop = neat.population.DesPopulation(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop

def run_adaptive_des(gens, env, max_steps, config, params, substrate, num_deployments=10,
                     max_trials=0, output=True, num_workers=None, schedule_sampler=None,
                     generation_schedule_sampler=None, reporters=None):
    """
    Generic OpenAI Gym runner for Adaptive DES-HyperNEAT.

    Fitness evaluation is parallelized across processes using neat-python's
    ParallelEvaluator. ``num_workers`` controls the size of the process pool and
    defaults to ``os.cpu_count()``. Pass ``num_workers=1`` to fall back to the
    original single-process evaluation (useful for debugging).

    Two mutually exclusive hooks decide which deployments genomes are scored on,
    both applied to the env via ``set_training_schedule`` and both ignored by envs
    without that method:

    ``schedule_sampler`` is a zero-argument callable invoked once per *genome* (in
    the worker that evaluates it), so each genome is scored on its own
    independently sampled deployments -- the same scheme as the Adaptive ES runner.

    ``generation_schedule_sampler`` is a zero-argument callable invoked once per
    *generation*, here in the parent process. Every genome of that generation is
    scored on the one sampled schedule, so within a generation the deployments are
    a fixed test that only the genomes vary against, while each new generation
    faces freshly sampled deployments. The sampled schedule crosses to the workers
    by value, so it must be picklable (the cat-and-mouse env uses a list of seeds).

    With neither, genomes use whatever schedule the env already holds.

    ``reporters`` is an optional iterable of neat reporters attached to the
    population alongside the built-in statistics (and stdout) reporters, so a
    caller can hook into each generation -- e.g. to save artifacts for the
    generation's best genome while evolution is still running.
    """
    trials = num_deployments

    if schedule_sampler is not None and generation_schedule_sampler is not None:
        raise ValueError("pass schedule_sampler or generation_schedule_sampler, not both: "
                         "they sample deployments at different rates (per genome vs "
                         "per generation)")

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    evaluator = None
    if num_workers > 1:
        # Each worker needs its own env (they're not safe to share across
        # processes). Registered envs are rebuilt from their string id via
        # gym.make; custom, unregistered env instances (whose ``spec`` is None)
        # are reconstructed by deep-copying the picklable instance in each
        # worker. Both factories are picklable so they can cross to the pool.
        spec = getattr(env, "spec", None)
        env_id = getattr(spec, "id", None)
        if isinstance(env_id, str):
            import gymnasium as gym
            env_factory = functools.partial(gym.make, env_id)
        else:
            env_factory = functools.partial(copy.deepcopy, env)

        # Parallel path: ParallelEvaluator fans genomes out across a process
        # pool, assigning the returned value to each genome's fitness. The
        # per-genome work (and the constant substrate/params/max_steps/trials,
        # plus schedule_sampler) is bound via functools.partial so the callable
        # stays picklable; the sampler is invoked fresh for each genome.
        base_eval = functools.partial(_eval_adaptive_des_genome_worker,
                                      substrate=substrate, params=params,
                                      max_steps=max_steps, trials=trials,
                                      schedule_sampler=schedule_sampler)
        evaluator = neat.ParallelEvaluator(
            num_workers,
            base_eval,
            initializer=_init_adaptive_des_worker,
            initargs=(env_factory,),
        )

        if generation_schedule_sampler is None:
            eval_fitness = evaluator.evaluate
        else:
            def eval_fitness(genomes, config):
                # One schedule per generation, drawn here in the parent and bound
                # into the callable the pool ships out with each genome, so all of
                # this generation's genomes meet the same deployments no matter
                # which worker takes them. ParallelEvaluator re-reads
                # eval_function on every evaluate(), so rebinding it is enough.
                evaluator.eval_function = functools.partial(
                    base_eval, schedule=generation_schedule_sampler())
                evaluator.evaluate(genomes, config)
    else:
        def eval_fitness(genomes, config):
            # Serial counterpart of the per-generation draw above: one schedule
            # for the whole generation, shared by every genome in it.
            schedule = (generation_schedule_sampler()
                        if generation_schedule_sampler is not None else None)
            for _, g in genomes:
                g.fitness = _eval_adaptive_des_genome(
                    g, config, substrate, params, max_steps, trials, env, schedule_sampler,
                    schedule)

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_desPop(None, stats_one, config, output)
    for reporter in reporters or ():
        pop.add_reporter(reporter)
    try:
        winner = pop.run(eval_fitness, gens)
    finally:
        if evaluator is not None:
            evaluator.close()

    return winner, (stats_one,)

def run_des(gens, env, max_steps, config, params, substrate, max_trials=0, output=True):
    """
    Generic OpenAI Gym runner for Adaptive DES-HyperNEAT.
    """
    trials = num_deployments

    def eval_fitness(genomes, config):

        for _, g in genomes:
            cppn = neat.nn.DesFeedForwardNetwork.create(g, config)
            network = DESNetwork(substrate, cppn, params)
            net = network.create_phenotype_network()

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()[0]
                net.reset()

                total_reward = 0
                done = False
                
                for _ in range(max_steps):
                    action = net.activate(ob)
                    
                    # action = np.argmax(o)
                    ob, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    if done:
                        net.activate(ob)  # Activate the network one last time to update its internal state and process potential reward signals
                        break
                    # print("done check")
                    
                fitnesses.append(total_reward)
                
                g.fitness = np.array(fitnesses).mean()
    # """
    # Generic OpenAI Gym runner for ES-HyperNEAT.
    # """
    # trials = 1

    # def eval_fitness(genomes, config):

    #     for _, g in genomes:
    #         cppn = neat.nn.DesFeedForwardNetwork.create(g, config)
    #         network = DESNetwork(substrate, cppn, params)
    #         net = network.create_phenotype_network()

    #         fitnesses = []

    #         for _ in range(trials):
    #             ob = env.reset()[0]
    #             net.reset()

    #             total_reward = 0
    #             done = False
                
    #             for _ in range(max_steps):
    #                 for _ in range(network.activations):
    #                     o = net.activate(ob)
                    
    #                 action = np.argmax(net.activate(ob))
    #                 ob, reward, terminated, truncated, _ = env.step(action)
    #                 done = terminated or truncated
    #                 total_reward += reward
    #                 if done:
    #                     break
                    
    #                 fitnesses.append(total_reward)
                
    #             g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_desPop(None, stats_one, config, output)
    winner = pop.run(eval_fitness, gens)

    return winner, (stats_one,)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_desPop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_desPop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

def run_es(gens, env, max_steps, config, params, substrate, max_trials=100, output=True):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):

        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(substrate, cppn, params)
            net = network.create_phenotype_network()

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()[0]
                net.reset()

                total_reward = 0
                done = False
                
                for _ in range(max_steps):
                    for _ in range(network.activations):
                        o = net.activate(ob)
                    
                    action = np.argmax(o)
                    ob, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    if done:
                        break
                    
                    fitnesses.append(total_reward)
                
                g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)

def run_adaptive_es(gens, env, max_steps, config, params, substrate, num_deployments=10,
                     max_trials=0, output=True, num_workers=None, schedule_sampler=None):
    """
    Generic OpenAI Gym runner for Adaptive ES-HyperNEAT.

    Fitness evaluation is parallelized across processes using neat-python's
    ParallelEvaluator. ``num_workers`` controls the size of the process pool and
    defaults to ``os.cpu_count()``. Pass ``num_workers=1`` to fall back to the
    original single-process evaluation (useful for debugging).

    ``schedule_sampler`` is an optional zero-argument callable invoked once per
    *genome* (in the worker that evaluates it) to produce that genome's own
    deployment schedule, applied to the env via ``set_training_schedule``. This
    differs from the DES runner, which shares one sampled schedule across a whole
    generation. When None, genomes use whatever schedule the env already holds.
    """
    trials = num_deployments

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    evaluator = None
    if num_workers > 1:
        # Each worker needs its own env (they're not safe to share across
        # processes). Registered envs are rebuilt from their string id via
        # gym.make; custom, unregistered env instances (whose ``spec`` is None)
        # are reconstructed by deep-copying the picklable instance in each
        # worker. Both factories are picklable so they can cross to the pool.
        spec = getattr(env, "spec", None)
        env_id = getattr(spec, "id", None)
        if isinstance(env_id, str):
            import gymnasium as gym
            env_factory = functools.partial(gym.make, env_id)
        else:
            env_factory = functools.partial(copy.deepcopy, env)

        # Parallel path: ParallelEvaluator fans genomes out across a process
        # pool, assigning the returned value to each genome's fitness. The
        # per-genome work (and the constant substrate/params/max_steps/trials)
        # is bound via functools.partial so the callable stays picklable.
        evaluator = neat.ParallelEvaluator(
            num_workers,
            functools.partial(_eval_adaptive_es_genome_worker,
                              substrate=substrate, params=params,
                              max_steps=max_steps, trials=trials,
                              schedule_sampler=schedule_sampler),
            initializer=_init_adaptive_des_worker,
            initargs=(env_factory,),
        )
        eval_fitness = evaluator.evaluate
    else:
        def eval_fitness(genomes, config):
            for _, g in genomes:
                g.fitness = _eval_adaptive_es_genome(
                    g, config, substrate, params, max_steps, trials, env, schedule_sampler)

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    try:
        winner_one = pop.run(eval_fitness, gens)
    finally:
        if evaluator is not None:
            evaluator.close()

    return winner_one, (stats_one,)

def run_hyper(gens, env, max_steps, config, substrate, activations, max_trials=100,
              activation="sigmoid", output=True):
    """
    Generic OpenAI Gym runner for HyperNEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):

        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            net = create_phenotype_network(cppn, substrate, activation)

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()
                net.reset()

                total_reward = 0

                for _ in range(max_steps):
                    for _ in range(activations):
                        o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    """
    Generic OpenAI Gym runner for NEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()

                total_reward = 0

                for _ in range(max_steps):
                    o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)
