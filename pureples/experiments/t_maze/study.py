"""
Helpers for multi-run data collection across the adaptive ES/DES T-Maze studies.

Used when an experiment's ``NUM_RUNS`` is greater than 1. EVERY run -- successful
or not -- records its per-generation fitness, a handful of network statistics,
and the four post-evolution test results measured on its best genome; once every
run has finished, a summary fitness graph and a summary text file aggregate them.

The summary reports each solution-quality statistic twice: over all saved runs
and over the successful runs alone. Because unsuccessful runs are now tested too,
the all-runs figures describe what the algorithm typically produces, rather than
only what it produces on the runs that happened to clear the fitness threshold.

Running this file directly regenerates a finished study's summary files from the
saved run records, without re-running evolution; see the ``__main__`` block at
the bottom.
"""

import os
import re
import sys
import json
import statistics as _stats
from dataclasses import dataclass, asdict, fields
from typing import List

# Force matplotlib's non-interactive Agg backend BEFORE any pyplot import. Only
# figures written to disk are produced here, and the experiment modules that
# import this one already force Agg themselves (a Tk interpreter created on a
# worker thread aborts the process); their later plt.switch_backend("TkAgg")
# calls are runtime switches and are unaffected by this.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pureples.experiments.t_maze.multi_t_maze import EXPECTED_ADAPTIVE_FITNESS


@dataclass
class RunRecord:
    """
    Metrics collected from a single evolutionary run.

    Every run -- successful or not -- records its per-generation fitness series
    plus the solution-quality fields below, all measured on the run's best
    genome. They stay ``None`` only where the metric does not apply
    (``num_branches`` is Adaptive DES only) or in records written by an older
    version of the study, which measured them for successful runs alone.
    """
    run_number: int
    generations_to_solution: int        # generations recorded; == the budget for an unsolved run
    mean_fitness_per_gen: List[float]   # padded to the generation budget
    best_fitness_per_gen: List[float]   # padded to the generation budget
    solved: bool = True                 # reached the fitness threshold within the budget
    # Solution-quality metrics, measured on the run's best genome (see above).
    mean_an_product: float = None       # mean of A*n over standard connections
    inhibitory_count: int = None        # # std connections with A*n < 0
    excitatory_count: int = None        # # std connections with A*n > 0
    cppn_connection_count: int = None   # enabled connections in the evolved CPPN genome
    cppn_node_count: int = None         # input + hidden + output nodes of the CPPN
    num_branches: int = None            # # branches (Adaptive DES only; None for ES)
    # Size of the phenotype network the CPPN expands into. std_connection_count
    # is the total, so it also covers the connections with A*n exactly 0 that
    # neither inhibitory_count nor excitatory_count includes.
    network_node_count: int = None      # nodes in the phenotype (inputs + hidden + outputs)
    std_connection_count: int = None    # standard connections in the phenotype
    mod_connection_count: int = None    # modulatory connections that affect the phenotype
    # Post-evolution test means (see multi_t_maze). None until measured.
    all_switch_fitness: float = None       # Test 1: mean over all training-range single-switch schedules
    double_switch_fitness: float = None    # Test 2: mean over the double-switch generalization deployments
    delayed_feedback_fitness: float = None  # Test 3: mean over the delayed-feedback deployments
    constant_feedback_fitness: float = None  # Test 4: mean over the constant-feedback deployments

    @property
    def dominant_connection_share(self):
        """
        Abundance of whichever signed connection type is more common in this
        network, as a fraction of the signed standard connections: 1.0 when the
        network has only one type, 0.5 when the two are exactly balanced (and
        also when there are no signed connections at all).

        This is deliberately sign-blind. A plain inhibitory/excitatory ratio
        carries the direction of the imbalance, so averaging it over many runs --
        half of which lean one way and half the other -- cancels out towards a
        meaningless middle. Taking the majority type's share instead measures how
        *lopsided* each network is regardless of which way it leans, so the
        across-runs mean stays interpretable: near 0.5 means runs typically
        produce balanced networks, near 1.0 means they typically commit to one
        connection type.

        Returns None when the connection counts were never measured (a record
        written before every run was analysed).
        """
        if self.inhibitory_count is None or self.excitatory_count is None:
            return None
        signed = self.inhibitory_count + self.excitatory_count
        if not signed:
            return 0.5  # no signed connections: perfectly "balanced" by convention
        return max(self.inhibitory_count, self.excitatory_count) / signed

    @property
    def dominant_connection_type(self):
        """
        Which connection type ``dominant_connection_share`` describes:
        ``"inhibitory"``, ``"excitatory"``, or ``"balanced"`` on a tie (including
        a network with no signed connections). None when unmeasured.
        """
        if self.inhibitory_count is None or self.excitatory_count is None:
            return None
        if self.inhibitory_count > self.excitatory_count:
            return "inhibitory"
        if self.excitatory_count > self.inhibitory_count:
            return "excitatory"
        return "balanced"


def save_run_record(path, record):
    """
    Persist ``record`` to ``path`` as JSON so a later batch can reload it.

    The summary written after each batch reloads every run's record from disk
    (see load_saved_records), so a study can be built up over several separate
    invocations without losing earlier runs' data.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2)


def load_run_record(path):
    """Reload a RunRecord previously written by save_run_record."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    known = {field.name for field in fields(RunRecord)}
    return RunRecord(**{k: v for k, v in data.items() if k in known})


def run_record_path(data_dir, base_name, run_number):
    """Path of the persisted JSON record for a single run."""
    return os.path.join(data_dir, f"{base_name}_run{run_number}.json")


def load_saved_records(data_dir, base_name, end_run):
    """
    Load every run record currently saved on disk for runs 1..``end_run``.

    Missing run numbers (never run, or an unsolved run in an old-style study that
    saved nothing) are simply skipped, so the summary covers whatever data exists
    -- exactly what lets a study accumulate across batches. Records are returned
    ordered by run number.
    """
    records = []
    for run_number in range(1, end_run + 1):
        path = run_record_path(data_dir, base_name, run_number)
        if os.path.exists(path):
            records.append(load_run_record(path))
    records.sort(key=lambda r: r.run_number)
    return records


_RECORD_FILENAME = re.compile(r"^(?P<base>.+)_run(?P<number>\d+)\.json$")


def discover_study(data_dir):
    """
    Return ``(base_name, run_numbers)`` for the records saved in ``data_dir``.

    Both are read off the record filenames rather than off the directory name,
    which cannot be trusted: an archived copy such as
    ``..._medium_data (pre-extra-data)`` still holds ``..._medium_run*.json``
    records. Raises ValueError when the directory holds no records at all, or
    when it mixes records from more than one study (nothing writes such a
    directory, but a hand-merged one would silently summarise two studies as
    one).
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"No such study directory: {data_dir}")

    base_names = {}
    for name in os.listdir(data_dir):
        match = _RECORD_FILENAME.match(name)
        if match:
            base_names.setdefault(match.group("base"), []).append(
                int(match.group("number")))

    if not base_names:
        raise ValueError(f"No *_run<N>.json records found in {data_dir}")
    if len(base_names) > 1:
        raise ValueError(
            f"{data_dir} mixes records from several studies "
            f"({', '.join(sorted(base_names))}); pass the one you want as "
            f"base_name.")

    base_name, run_numbers = next(iter(base_names.items()))
    return base_name, sorted(run_numbers)


def infer_title(base_name):
    """
    The report title a study's records were originally written with.

    The experiment scripts pass a fixed title to write_summary; regenerating a
    summary has only the base name to go on, so map it back to the same string
    (anything unrecognised falls back to the base name itself).
    """
    if base_name.startswith("adaptive_des"):
        return "Adaptive DES T-Maze"
    if base_name.startswith("adaptive_es"):
        return "Adaptive ES T-Maze"
    return base_name


# Short name and colour each algorithm is drawn with in the cross-study
# comparison graphs. This ASES/AES naming is deliberately confined to those
# graphs -- the per-study summaries keep the "Adaptive DES/ES T-Maze" titles
# infer_title returns, so nothing already written changes meaning.
COMPARISON_STYLE = {
    "adaptive_des": ("ASES", "tab:red"),
    "adaptive_es": ("AES", "tab:blue"),
}


def comparison_style(base_name):
    """
    ``(short name, colour)`` a study's base name is plotted with in the
    comparison graphs. Raises ValueError for anything that is not one of the two
    adaptive algorithms, which are the only ones the comparison is defined for.
    """
    for prefix, style in COMPARISON_STYLE.items():
        if base_name.startswith(prefix):
            return style
    raise ValueError(
        f"No comparison style for {base_name}: expected a study whose name "
        f"starts with {' or '.join(sorted(COMPARISON_STYLE))}.")


def study_variant(base_name):
    """
    The part of a base name that identifies the maze and network size, with the
    algorithm prefix stripped: ``adaptive_des_hyperneat_t_maze_1_turn_medium``
    becomes ``1_turn_medium``. Used to name the comparison folder after what the
    two studies have in common rather than after either algorithm.
    """
    for prefix in ("adaptive_des_hyperneat_t_maze_", "adaptive_es_hyperneat_t_maze_",
                   "adaptive_des_", "adaptive_es_"):
        if base_name.startswith(prefix):
            return base_name[len(prefix):]
    return base_name


def comparison_dir_name(base_names):
    """
    Folder name for the comparison of the given studies, e.g.
    ``ases_vs_aes_1_turn_medium_comparison`` when both studies are the same
    variant. Differing variants are both named, so comparing medium against
    large cannot silently overwrite the medium-against-medium graphs.
    """
    variants = [study_variant(name) for name in base_names]
    shared = variants[0] if len(set(variants)) == 1 else "_vs_".join(variants)
    return f"ases_vs_aes_{shared}_comparison"


def records_generation_budget(records):
    """
    The number of generations every record in ``records`` has fitness data for.

    Each record's series was padded to its own study's generation budget, so
    this is normally the same number for all of them. It is the *shortest*
    series that bounds what can be summarised, since write_summary indexes every
    record at every generation it reports; a directory whose records disagree
    (one built either side of a budget change) is summarised over the common
    prefix, with a warning.
    """
    lengths = {len(r.mean_fitness_per_gen) for r in records}
    if len(lengths) > 1:
        print(f"Warning: saved runs have different generation budgets "
              f"({', '.join(str(n) for n in sorted(lengths))}); summarising "
              f"over the first {min(lengths)} generations.")
    return min(lengths)


def load_study(data_dir, base_name=None, total_runs=None):
    """
    Return ``(base_name, total_runs, records)`` for the study saved in
    ``data_dir``.

    The study identifies itself: ``base_name`` and the run range come from the
    record filenames (see discover_study) unless overridden. Raises ValueError
    when the directory holds no usable records, so callers fail with an
    explanation instead of on an empty list further down.
    """
    discovered_base, run_numbers = discover_study(data_dir)
    if base_name is None:
        base_name = discovered_base
    if total_runs is None:
        total_runs = max(run_numbers)

    records = load_saved_records(data_dir, base_name, total_runs)
    if not records:
        raise ValueError(
            f"No records for {base_name} runs 1..{total_runs} in {data_dir}")
    return base_name, total_runs, records


def regenerate_summary(data_dir, base_name=None, title=None, total_runs=None,
                       max_generations=None):
    """
    Rebuild a finished study's summary graph and summary text from the run
    records saved in ``data_dir``, and return the records they were built from.

    This is the whole point of persisting a RunRecord per run: everything the
    summary reports is already on disk, so changing how the summary presents it
    needs no evolution. The two files are written to the same paths the
    experiment scripts use, so they are overwritten in place.

    Every argument except ``data_dir`` is recovered from the saved records when
    left None -- pass one only to override what was detected.
    """
    base_name, total_runs, records = load_study(data_dir, base_name, total_runs)
    if title is None:
        title = infer_title(base_name)
    if max_generations is None:
        max_generations = records_generation_budget(records)

    write_summary(
        graph_path=os.path.join(data_dir, f"{base_name}_summary_fitness.png"),
        text_path=os.path.join(data_dir, f"{base_name}_summary.txt"),
        records=records, total_runs=total_runs,
        max_generations=max_generations, title=title)
    return records


def _mean_std(values):
    """
    Return ``(mean, std dev, n)`` over the values that are present.

    ``None`` entries are dropped, so a metric that some records never measured
    still reports a meaningful figure alongside the count it was taken over.
    Returns ``(None, None, 0)`` when nothing is left, and a std dev of 0.0 for a
    single value.
    """
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    return (_stats.mean(values),
            _stats.stdev(values) if len(values) > 1 else 0.0,
            len(values))


def solved(winner, config):
    """
    Return True when ``winner`` reached the configured fitness threshold.

    Mirrors neat's own termination test: a run is successful only when the best
    genome's fitness meets ``fitness_threshold`` (runs that exhaust the
    generation budget without doing so end with a below-threshold best genome).
    """
    return winner.fitness is not None and winner.fitness >= config.fitness_threshold


def padded_fitness_series(stats, max_generations):
    """
    Return ``(mean_per_gen, best_per_gen)`` for one run, each padded to
    ``max_generations`` entries.

    A run that solves the task in G < ``max_generations`` generations only has G
    recorded generations; the remaining entries are filled with the final
    generation's value so every run contributes a full-length series.
    """
    mean_per_gen = list(stats.get_fitness_mean())
    best_per_gen = [g.fitness for g in stats.most_fit_genomes]

    def pad(series):
        if not series:
            return [0.0] * max_generations
        if len(series) >= max_generations:
            return series[:max_generations]
        return series + [series[-1]] * (max_generations - len(series))

    return pad(mean_per_gen), pad(best_per_gen)


def connection_sign_metrics(an_products):
    """
    Given the A*n product for every standard connection, return
    ``(mean_product, inhibitory_count, excitatory_count)``.

    A connection is inhibitory when A*n < 0 and excitatory when A*n > 0
    (products of exactly 0 are counted as neither).
    """
    products = list(an_products)
    mean_product = sum(products) / len(products) if products else 0.0
    inhibitory = sum(1 for p in products if p < 0.0)
    excitatory = sum(1 for p in products if p > 0.0)
    return mean_product, inhibitory, excitatory


def count_cppn_connections(genome):
    """Number of expressed (enabled) connections in the evolved CPPN genome."""
    return sum(1 for cg in genome.connections.values() if cg.enabled)


def count_cppn_nodes(genome, config):
    """
    Number of nodes in the evolved CPPN genome: inputs + hidden + outputs.

    neat only stores node genes for the hidden and output nodes (the inputs are
    implicit, one per ``genome_config.input_keys`` entry), so the inputs are
    added back here to give the size of the CPPN as drawn.
    """
    return len(genome.nodes) + len(config.genome_config.input_keys)


def write_run_report(path, record, max_generations, title):
    """
    Write the per-run text report: the generations-to-solution count (successful
    runs only), the four post-evolution test means, the two A*n ratios, the CPPN
    connection/node counts, and the padded per-generation mean/best fitness
    series. Everything except the generations-to-solution line is written for
    every run, successful or not, since every run's best genome is tested.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title} — run {record.run_number} report\n")
        f.write("=" * 60 + "\n\n")

        if record.solved:
            f.write(f"Generations to reach fitness threshold: {record.generations_to_solution}\n\n")
        else:
            f.write("Result: did NOT reach the fitness threshold within "
                    f"{record.generations_to_solution} generations.\n")
            f.write("  (The run's best genome was still analysed and put through the four\n"
                    "   post-evolution tests below, so it contributes to the summary graph\n"
                    "   and to the all-runs statistics in the summary file.)\n\n")

        if (record.all_switch_fitness is not None
                or record.double_switch_fitness is not None
                or record.delayed_feedback_fitness is not None
                or record.constant_feedback_fitness is not None):
            f.write("Post-evolution tests (mean fitness per deployment; not seen in training):\n")
            if record.all_switch_fitness is not None:
                f.write(f"  Test 1 -- all training-range single-switch schedules: "
                        f"{record.all_switch_fitness:.6f}\n")
            if record.double_switch_fitness is not None:
                f.write(f"  Test 2 -- double-switch generalization (20 episodes): "
                        f"{record.double_switch_fitness:.6f}\n")
            if record.delayed_feedback_fitness is not None:
                f.write(f"  Test 3 -- delayed reward feedback: "
                        f"{record.delayed_feedback_fitness:.6f}\n")
            if record.constant_feedback_fitness is not None:
                f.write(f"  Test 4 -- constant (uninformative) reward feedback: "
                        f"{record.constant_feedback_fitness:.6f}\n")
            f.write("\n")

        if (record.network_node_count is not None
                or record.std_connection_count is not None
                or record.num_branches is not None):
            f.write("Final network size (the phenotype the CPPN expands into):\n")
            if record.network_node_count is not None:
                f.write(f"  Nodes: {record.network_node_count}\n")
            if record.std_connection_count is not None:
                f.write(f"  Standard connections: {record.std_connection_count}\n")
            if record.mod_connection_count is not None:
                f.write(f"  Modulatory connections: {record.mod_connection_count}\n")
            if record.num_branches is not None:
                f.write(f"  Branches: {record.num_branches}\n")
            f.write("\n")

        if record.mean_an_product is not None:
            share = record.dominant_connection_share
            share_text = ("not measured" if share is None
                          else f"{share:.6f} ({record.dominant_connection_type})")
            f.write("Excitatory/inhibitory analysis (sign of A*n per standard connection):\n")
            f.write(f"  Inhibition/excitation ratio (mean of A*n over standard connections): "
                    f"{record.mean_an_product:.6f}\n")
            f.write(f"  Dominant connection type abundance (0.5 = balanced, 1 = one type "
                    f"only): {share_text}\n")
            f.write(f"    inhibitory (A*n < 0) standard connections: {record.inhibitory_count}\n")
            f.write(f"    excitatory (A*n > 0) standard connections: {record.excitatory_count}\n\n")

        if (record.cppn_connection_count is not None
                or record.cppn_node_count is not None):
            f.write("Evolved CPPN genome size:\n")
            if record.cppn_connection_count is not None:
                f.write(f"  Enabled connections: {record.cppn_connection_count}\n")
            if record.cppn_node_count is not None:
                f.write(f"  Nodes: {record.cppn_node_count}\n")
            f.write("\n")

        f.write(f"Per-generation fitness (padded to {max_generations} generations):\n")
        f.write("generation,mean_fitness,best_fitness\n")
        for gen in range(max_generations):
            mean_v = record.mean_fitness_per_gen[gen]
            best_v = record.best_fitness_per_gen[gen]
            f.write(f"{gen},{mean_v:.6f},{best_v:.6f}\n")


def write_summary(graph_path, text_path, records, total_runs, max_generations, title):
    """
    Write the across-runs summary graph and text file.

    ``records`` is every run saved on disk so far (see load_saved_records),
    successful and unsuccessful alike.

      * The text file is written as two self-contained sections, ALL RUNS first
        and SUCCESSFUL RUNS below it, each carrying the same statistics in the
        same order: post-evolution tests, network analysis, and the
        per-generation fitness table. Every run's best genome is tested, so the
        all-runs section measures what the algorithm typically produces, while
        the successful section measures what it produces on the runs that cleared
        the fitness threshold; reading one against the other shows how much the
        threshold biases the successful-only view.
      * Generations-to-solution appears in the successful section only: an
        unsuccessful run simply used the whole budget, so averaging it across all
        runs is meaningless.
      * The summary graph stays all-runs, so an unsuccessful run drags the curve
        down and the graph reflects the true success rate. The successful
        section's per-generation table is the successful-only counterpart.

    Each statistic is reported alongside the number of runs it was taken over,
    which may be below the group size for records written before every run was
    analysed (those left the fields unset).

    The graph plots, per generation, the mean over runs of each run's mean
    fitness (line) and the mean over runs of each run's best fitness (dots); no
    standard deviation is drawn.
    """
    if not records:
        return

    successful = [r for r in records if r.solved]

    generations = list(range(max_generations))

    def per_generation_means(group):
        """``(mean of per-run means, mean of per-run bests)`` per generation."""
        return ([_stats.mean(r.mean_fitness_per_gen[g] for r in group)
                 for g in generations],
                [_stats.mean(r.best_fitness_per_gen[g] for r in group)
                 for g in generations])

    mean_of_means, mean_of_bests = per_generation_means(records)

    # --- Summary graph (all runs) -----------------------------------------
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(generations, mean_of_means, color="tab:blue", linewidth=1.5,
            label="Mean of per-run mean fitness")
    ax.scatter(generations, mean_of_bests, color="tab:red", s=16, zorder=3,
               label="Mean of per-run best fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(f"{title} — fitness over {len(records)} run(s) "
                 f"({len(successful)} solved)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(graph_path, bbox_inches="tight")
    plt.close(fig)

    # --- Solution-quality statistics ---------------------------------------
    # Every metric is reported for both groups, each group getting its own
    # self-contained section; see this function's docstring.
    groups = (("ALL RUNS", records), ("SUCCESSFUL RUNS", successful))

    def abs_mean_an_product(record):
        """
        Magnitude of a record's mean A*n, or None when unmeasured.

        The signed mean says which way that one network leans, which is why the
        per-run report keeps it; averaged over runs that lean either way it
        cancels towards 0 and stops describing anything. The magnitude answers
        the question the summary is actually asking -- how strongly the typical
        run's connections lean, whichever way they lean.
        """
        return None if record.mean_an_product is None else abs(record.mean_an_product)

    # Each test also carries the key of its EXPECTED_ADAPTIVE_FITNESS entry, so
    # the summary prints what an ideal adaptive agent would score next to what was
    # actually measured. The network metrics below have no such reference and pass
    # None.
    test_metrics = [
        ("Test 1 -- all training-range single-switch schedules",
         lambda r: r.all_switch_fitness, "{:.4f}", "all_switch"),
        ("Test 2 -- double-switch generalization (20 episodes)",
         lambda r: r.double_switch_fitness, "{:.4f}", "double_switch"),
        ("Test 3 -- delayed reward feedback",
         lambda r: r.delayed_feedback_fitness, "{:.4f}", "delayed_feedback"),
        ("Test 4 -- constant (uninformative) reward feedback",
         lambda r: r.constant_feedback_fitness, "{:.4f}", "constant_feedback"),
    ]

    # Grouped: how big the final network is, how its connections are signed, and
    # how big the CPPN genome that generated it is.
    network_metrics = [
        ("Nodes in the final network",
         lambda r: r.network_node_count, "{:.4f}"),
        ("Standard connections in the final network",
         lambda r: r.std_connection_count, "{:.4f}"),
        ("Modulatory connections in the final network",
         lambda r: r.mod_connection_count, "{:.4f}"),
        ("Branches in the final network",
         lambda r: r.num_branches, "{:.4f}"),
        ("Mean |A*n| over standard connections (inhibition/excitation ratio, "
         "magnitude only)",
         abs_mean_an_product, "{:.6f}"),
        ("Dominant connection type abundance (0.5 = balanced, 1 = one type only)",
         lambda r: r.dominant_connection_share, "{:.6f}"),
        ("Inhibitory (A*n < 0) standard connections",
         lambda r: r.inhibitory_count, "{:.4f}"),
        ("Excitatory (A*n > 0) standard connections",
         lambda r: r.excitatory_count, "{:.4f}"),
        ("Enabled connections in the evolved CPPN genome",
         lambda r: r.cppn_connection_count, "{:.4f}"),
        ("Nodes in the evolved CPPN genome",
         lambda r: r.cppn_node_count, "{:.4f}"),
    ]

    # --- Summary text ------------------------------------------------------
    os.makedirs(os.path.dirname(text_path), exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(f"{title} — summary over all saved runs\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Runs with saved data: {len(records)} of {total_runs} expected "
                f"(runs 1..{total_runs})\n")
        f.write(f"Successful (solved) runs: {len(successful)}\n\n")

        def measured_anywhere(getter):
            """
            True when at least one saved run measured this metric.

            Decided over all runs rather than per section, so both sections list
            the same metrics: a metric measured somewhere prints ``no data`` in a
            section that lacks it instead of vanishing from that section only.
            One never measured at all (branches on an Adaptive ES study) is
            dropped from both.
            """
            return any(getter(r) is not None for r in records)

        def write_metric(label, getter, fmt, group, reference_key=None):
            """
            Write one metric's mean/std dev over ``group``, or skip it entirely.

            ``reference_key`` names an EXPECTED_ADAPTIVE_FITNESS entry; when given,
            the expected/optimum/clairvoyant fitnesses are appended to the metric's
            own line, so the measured figures sit directly under the references
            they are read against.
            """
            if not measured_anywhere(getter):
                return
            mean_v, std_v, n = _mean_std(getter(r) for r in group)
            if reference_key is None:
                f.write(f"  {label}:\n")
            else:
                expected, optimum, ceiling = EXPECTED_ADAPTIVE_FITNESS[reference_key]
                f.write(f"  {label}  [expected = {fmt.format(expected)}, "
                        f"optimum = {fmt.format(optimum)}, "
                        f"clairvoyant = {fmt.format(ceiling)}]:\n")
            if mean_v is None:
                f.write(f"    (n={n:>3}): no data\n")
            else:
                f.write(f"    (n={n:>3}): mean = {fmt.format(mean_v)}"
                        f"   std dev = {fmt.format(std_v)}\n")

        def write_section(name, group):
            """
            Write one group's complete set of statistics under its own banner.

            Both sections carry the same metrics in the same order, so the two can
            be read against each other. Generations-to-solution appears only in the
            successful section: an unsuccessful run simply used the whole budget,
            so it has no meaningful value to average.
            """
            f.write("=" * 60 + "\n")
            f.write(f"{name} (n = {len(group)})\n")
            f.write("=" * 60 + "\n\n")

            if not group:
                f.write("No runs in this group.\n\n")
                return

            if group is successful:
                gens_mean, gens_std, gens_n = _mean_std(
                    r.generations_to_solution for r in group)
                f.write("Generations to find a solution:\n")
                f.write(f"    (n={gens_n:>3}): mean = {gens_mean:.4f}"
                        f"   std dev = {gens_std:.4f}\n\n")

            f.write("Post-evolution test fitness (per-run mean over that test's "
                    "deployments):\n")
            for label, getter, fmt, reference_key in test_metrics:
                write_metric(label, getter, fmt, group, reference_key)
            f.write("\n")

            f.write("Network analysis (measured on each run's best genome):\n")
            for label, getter, fmt in network_metrics:
                write_metric(label, getter, fmt, group)
            f.write("\n")

            group_means, group_bests = per_generation_means(group)
            f.write("Per-generation summary fitness:\n")
            f.write("generation,mean_of_mean_fitness,mean_of_best_fitness\n")
            for gen in generations:
                f.write(f"{gen},{group_means[gen]:.6f},{group_bests[gen]:.6f}\n")
            f.write("\n")

        for name, group in groups:
            write_section(name, group)


# --- Cross-study comparison graphs -----------------------------------------
#
# The per-study summary above describes one algorithm; these put the two side by
# side on shared axes. They plot the per-generation BEST fitness only (the
# mean-of-mean series the summary graph also carries would quadruple the lines
# for little gain), each algorithm drawn as one centre line with a translucent
# band of the same colour showing the spread across its runs.

# (statistic, filename stem, title text, band text). Both statistics are drawn
# for both run groups below, giving four graphs.
COMPARISON_GRAPHS = (
    ("mean", "mean_best_fitness", "mean best fitness", "±1 std dev"),
    ("median", "median_best_fitness", "median best fitness", "Q1–Q3"),
)

# (filename suffix, title text, what the group holds, selector). "All runs"
# shows what the algorithm typically produces; "successful runs only" shows what
# it produces when it works, and the two read against each other the same way
# the summary's two sections do.
COMPARISON_GROUPS = (
    ("all_runs", "all runs", "saved runs", lambda records: list(records)),
    ("successful_runs", "successful runs only", "successful runs",
     lambda records: [r for r in records if r.solved]),
)


def per_generation_spread(records, max_generations, statistic):
    """
    Per-generation series describing the spread of ``records``' BEST fitness.

    Returns a dict of equal-length series, the first entry being the centre line
    the graph draws and the rest the numbers its band is built from:

      * ``"mean"``   -> ``mean``, ``std_dev``, ``variance``
      * ``"median"`` -> ``median``, ``q1``, ``q3``

    A single-run group gets a std dev of 0.0 (matching _mean_std's convention)
    and quartiles equal to its one value, so a study with one saved run still
    plots a line with a zero-width band instead of raising.
    """
    if statistic == "mean":
        series = {"mean": [], "std_dev": [], "variance": []}
        for gen in range(max_generations):
            values = [r.best_fitness_per_gen[gen] for r in records]
            std_dev = _stats.stdev(values) if len(values) > 1 else 0.0
            series["mean"].append(_stats.mean(values))
            series["std_dev"].append(std_dev)
            series["variance"].append(std_dev * std_dev)
        return series

    if statistic == "median":
        series = {"median": [], "q1": [], "q3": []}
        for gen in range(max_generations):
            values = [r.best_fitness_per_gen[gen] for r in records]
            q1, _q2, q3 = _stats.quantiles(values, n=4, method="inclusive")
            series["median"].append(_stats.median(values))
            series["q1"].append(q1)
            series["q3"].append(q3)
        return series

    raise ValueError(f"Unknown statistic {statistic!r}: expected 'mean' or 'median'.")


def spread_band(statistic, series):
    """
    ``(lower, upper)`` bounds of the translucent band drawn around a centre line
    produced by per_generation_spread: one std dev either side of the mean, or
    the Q1..Q3 range around the median.
    """
    if statistic == "mean":
        return ([m - s for m, s in zip(series["mean"], series["std_dev"])],
                [m + s for m, s in zip(series["mean"], series["std_dev"])])
    return series["q1"], series["q3"]


def _write_comparison_data(path, heading, generations, plotted, skipped, band_text):
    """
    Write the numbers behind one comparison graph: a header naming each study
    and where it came from, then one row per generation carrying every series
    that graph plots.

    ``plotted`` is a list of ``(short name, colour, base_name, data_dir,
    records, series)``; ``skipped`` names the algorithms that had no runs in
    this group, so the file explains a missing pair of columns rather than
    leaving the reader to guess.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{heading}\n")
        f.write("=" * 60 + "\n\n")
        for short, _colour, base_name, data_dir, records, _series in plotted:
            f.write(f"{short}: {base_name} -- {len(records)} run(s)\n")
            f.write(f"    from {data_dir}\n")
        for short, base_name, reason in skipped:
            f.write(f"{short}: {base_name} -- {reason}; not plotted\n")
        f.write(f"\nGenerations: {len(generations)}\n")
        f.write(f"Band drawn around each line: {band_text}\n\n")

        columns = [(f"{short}_{name}", series[name])
                   for short, _colour, _base, _dir, _records, series in plotted
                   for name in series]
        f.write("generation," + ",".join(name for name, _ in columns) + "\n")
        for gen in generations:
            values = ",".join(f"{values[gen]:.6f}" for _name, values in columns)
            f.write(f"{gen},{values}\n")


def write_comparison_graphs(output_dir, studies, max_generations=None,
                            y_limits=(0.0, 18.0)):
    """
    Write the cross-study comparison graphs and their data files into
    ``output_dir``, and return the paths written.

    ``studies`` is a sequence of ``(base_name, data_dir, records)`` -- normally
    the adaptive DES study and the adaptive ES one, drawn as ASES in red and AES
    in blue (see COMPARISON_STYLE). Four graphs are produced: mean and median
    best fitness, each over all runs and over the successful runs alone, each
    accompanied by a ``.txt`` of the plotted numbers.

    ``max_generations`` defaults to the span every record covers, so pairing a
    1000-generation study with a 1250-generation one compares them over the
    common 1000 rather than running off the end of the shorter records. An
    algorithm with no runs in a group (an all-unsuccessful study in the
    successful-only graphs) is left out of that graph and named in its title and
    data file; a group empty for both algorithms is skipped entirely.
    """
    os.makedirs(output_dir, exist_ok=True)

    if max_generations is None:
        max_generations = records_generation_budget(
            [record for _base, _dir, records in studies for record in records])
    generations = list(range(max_generations))

    pair_name = " vs ".join(comparison_style(base)[0] for base, _dir, _recs in studies)
    written = []

    for statistic, file_stem, stat_text, band_text in COMPARISON_GRAPHS:
        for group_suffix, group_text, group_holds, select in COMPARISON_GROUPS:
            plotted, skipped = [], []
            for base_name, data_dir, records in studies:
                short, colour = comparison_style(base_name)
                group = select(records)
                if group:
                    plotted.append((
                        short, colour, base_name, data_dir, group,
                        per_generation_spread(group, max_generations, statistic)))
                else:
                    skipped.append((short, base_name, f"no {group_holds}"))

            if not plotted:
                print(f"Skipping {file_stem}_{group_suffix}: neither study has "
                      f"{group_holds}.")
                continue

            heading = f"{pair_name} — {stat_text} ({group_text})"
            if skipped:
                heading += f" [no data for {', '.join(s for s, _b, _r in skipped)}]"

            fig, ax = plt.subplots(figsize=(9, 6))
            for short, colour, _base, _dir, group, series in plotted:
                centre = next(iter(series.values()))
                lower, upper = spread_band(statistic, series)
                ax.fill_between(generations, lower, upper, color=colour, alpha=0.4,
                                linewidth=0, label=f"{short} {band_text}")
                ax.plot(generations, centre, color=colour, linewidth=1.0,
                        label=f"{short} (n={len(group)})")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Best fitness")
            ax.set_ylim(*y_limits)
            ax.set_xlim(0, max_generations - 1)
            ax.set_title(heading)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            graph_path = os.path.join(output_dir, f"{file_stem}_{group_suffix}.png")
            fig.savefig(graph_path, bbox_inches="tight")
            plt.close(fig)

            data_path = os.path.join(output_dir, f"{file_stem}_{group_suffix}.txt")
            _write_comparison_data(data_path, heading, generations, plotted,
                                   skipped, band_text)
            written.extend((graph_path, data_path))

    return written


def _resolve_data_dir(path):
    """
    Accept a study directory given relative to the current directory or to the
    repository root, so the regeneration entry point works from anywhere.
    """
    if os.path.isdir(path):
        return path
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    from_root = os.path.join(repo_root, path)
    return from_root if os.path.isdir(from_root) else path


# If run as script.
if __name__ == '__main__':
    # Rebuild study outputs from the run records already saved on disk -- no
    # evolution, no re-testing -- so how the results are presented can be changed
    # without repeating a 30-run study. Set the paths below and run, from the
    # repo root:
    #
    #     python -m pureples.experiments.t_maze.study
    #
    # What gets written depends on how many of the two paths are set:
    #
    #   * One of them (the other left None): that study's own
    #     <base>_summary.txt and <base>_summary_fitness.png are rewritten in
    #     place. Its per-run <base>_run<N>_data.txt reports are left untouched.
    #   * Both: only the cross-study comparison graphs are written, into a new
    #     folder under results/ (see COMPARISON_DIR). Neither study's own
    #     summary files are touched -- set one path to None to refresh those.
    DES_DATA_DIR = ("pureples/experiments/t_maze/results/"
                    "adaptive_des_hyperneat_t_maze_1_turn_medium_data")
    ES_DATA_DIR = ("pureples/experiments/t_maze/results/"
                   "adaptive_es_hyperneat_t_maze_1_turn_medium_data")

    # Where the comparison graphs go. None names the folder after the two
    # studies, e.g. results/ases_vs_aes_1_turn_medium_comparison; set a path to
    # override. Comparison graphs only -- ignored when just one study is set.
    COMPARISON_DIR = None
    Y_LIMITS = (0.0, 18.0)  # fitness axis of every comparison graph

    # Both are recovered from the saved records when left None; set one only to
    # override what was detected.
    TOTAL_RUNS = None       # highest run number to include; default: highest saved
    MAX_GENERATIONS = None  # default: the length of the saved fitness series

    selected = [_resolve_data_dir(path)
                for path in (DES_DATA_DIR, ES_DATA_DIR) if path is not None]
    if not selected:
        sys.exit("Nothing to do: set DES_DATA_DIR, ES_DATA_DIR, or both.")

    if len(selected) == 1:
        target_dir = selected[0]
        try:
            summarised = regenerate_summary(target_dir, total_runs=TOTAL_RUNS,
                                            max_generations=MAX_GENERATIONS)
            base_name, _run_numbers = discover_study(target_dir)
        except ValueError as error:
            sys.exit(f"Cannot regenerate a summary: {error}")

        solved_runs = sum(1 for r in summarised if r.solved)
        generations_used = (MAX_GENERATIONS if MAX_GENERATIONS is not None
                            else records_generation_budget(summarised))
        print(f"Regenerated the summary for {base_name} from {len(summarised)} "
              f"saved run(s) ({solved_runs} solved) over {generations_used} "
              f"generations.")
        print(f"  {os.path.join(target_dir, base_name + '_summary.txt')}")
        print(f"  {os.path.join(target_dir, base_name + '_summary_fitness.png')}")
    else:
        try:
            loaded = []
            for data_dir in selected:
                base_name, _total, records = load_study(data_dir, total_runs=TOTAL_RUNS)
                comparison_style(base_name)  # fail early on an unrecognised study
                loaded.append((base_name, data_dir, records))

            # Auto-named beside the studies themselves, so the comparison lands
            # in results/ next to the data folders it was built from.
            output_dir = COMPARISON_DIR or os.path.join(
                os.path.dirname(selected[0]),
                comparison_dir_name([base for base, _dir, _recs in loaded]))
            written = write_comparison_graphs(
                output_dir, loaded, max_generations=MAX_GENERATIONS,
                y_limits=Y_LIMITS)
        except ValueError as error:
            sys.exit(f"Cannot build the comparison graphs: {error}")

        for base_name, _dir, records in loaded:
            short, _colour = comparison_style(base_name)
            solved_runs = sum(1 for r in records if r.solved)
            print(f"{short}: {base_name} -- {len(records)} run(s), "
                  f"{solved_runs} solved.")
        print(f"Wrote {len(written)} comparison file(s):")
        for path in written:
            print(f"  {path}")
