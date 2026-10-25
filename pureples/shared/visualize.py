"""
Varying visualisation tools.
"""

import os
import pickle
import graphviz
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from matplotlib.patches import FancyArrowPatch
import colorsys


def draw_net(net, filename=None, node_names={}, node_colors={}):
    """
    Draw neural network with arbitrary topology.
    """
    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph('svg', node_attr=node_attrs)

    inputs = set()
    for k in net.input_nodes:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled',
                       'shape': 'box',
                       'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in net.output_nodes:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled',
                      'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    for node, _, _, _, _, links in net.node_evals:
        for i, w in links:
            node_input, output = node, i
            a = node_names.get(output, str(output))
            b = node_names.get(node_input, str(node_input))
            style = 'solid'
            color = 'green' if w > 0.0 else 'red'
            width = str(0.1 + abs(w / 2.0))
            dot.edge(a, b, _attributes={
                     'style': style, 'color': color, 'penwidth': width})

    dot.render(filename)

    return dot


def onclick(event):
    """
    Click handler for weight gradient created by a CPPN. Will re-query with the clicked coordinate.
    """
    plt.close()
    x = event.xdata
    y = event.ydata

    path_to_cppn = "es_hyperneat_xor_small_cppn.pkl"
    # For now, path_to_cppn should match path in test_cppn.py, sorry.
    with open(path_to_cppn, 'rb') as cppn_input:
        cppn = pickle.load(cppn_input)
        from pureples.es_hyperneat.es_hyperneat import find_pattern
        pattern = find_pattern(cppn, (x, y))
        draw_pattern(pattern)


def draw_pattern(im, res=60):
    """
    Draws the pattern/weight gradient queried by a CPPN.
    """
    fig = plt.figure()
    plt.axis([-1, 1, -1, 1])
    fig.add_subplot(111)

    a = range(res)
    b = range(res)

    for x in a:
        for y in b:
            px = -1.0 + (x/float(res))*2.0+1.0/float(res)
            py = -1.0 + (y/float(res))*2.0+1.0/float(res)
            c = str(0.5-im[x][y]/float(res))
            plt.plot(px, py, marker='s', color=c)

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.grid()
    plt.show()


def draw_es(id_to_coords, connections, filename):
    """
    Draw the net created by ES-HyperNEAT
    """
    fig = plt.figure()
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    fig.add_subplot(111)

    for c in connections:
        color = 'red'
        if c.weight > 0.0:
            color = 'black'
        plt.arrow(c.x1, c.y1, c.x2-c.x1, c.y2-c.y1, head_width=0.00, head_length=0.0,
                  fc=color, ec=color, length_includes_head=True)

    for (coord, _) in id_to_coords.items():
        plt.plot(coord[0], coord[1], marker='o', markersize=8.0, color='grey')

    plt.grid()
    fig.savefig(filename)

def _add_connection_arrow(ax, c, color, linestyle):
    """
    Draw a single connection as an arrow (standard -> solid, modulatory -> dashed)
    and return the patch so its visibility can be toggled later.
    """
    patch = FancyArrowPatch(
        (c.x1, c.y1), (c.x2, c.y2),
        arrowstyle='-|>', mutation_scale=10,
        color=color, linestyle=linestyle, linewidth=1.2,
        shrinkA=3.0, shrinkB=3.0, zorder=1)
    ax.add_patch(patch)
    return patch


def _save_clean(fig, widget_axes, filename):
    """
    Save the figure to `filename` with the toggle panels hidden, so the saved
    image contains only the network (reflecting the currently visible toggles).
    """
    prev = [a.get_visible() for a in widget_axes]
    for a in widget_axes:
        a.set_visible(False)
    fig.savefig(filename, bbox_inches='tight')
    for a, was_visible in zip(widget_axes, prev):
        a.set_visible(was_visible)
    fig.canvas.draw_idle()


def draw_adaptive_es(id_to_coords, std_connections, mod_connections, filename, show=True):
    """
    Draw the net created by Adaptive ES-HyperNEAT.

    Standard connections are solid, modulatory connections are dashed; every
    connection is coloured black (positive weight) or red (negative weight).
    Check buttons toggle the standard and modulatory connections on/off, and a
    Save button writes the currently visible network to `filename`.

    The default image is written to `filename` immediately. When `show` is True
    the interactive window is opened here (blocking); when False the built figure
    is returned so the caller can save other outputs first and display it later
    (e.g. via `plt.show()`).
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes([0.28, 0.08, 0.68, 0.86])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.25, 1.1)
    ax.grid(True)
    ax.set_title("Adaptive ES-HyperNEAT network")

    # Group the drawn patches by connection type so they can be toggled together.
    groups = {'standard': [], 'modulatory': []}
    for c in std_connections:
        color = 'black' if c.weight > 0.0 else 'red'
        groups['standard'].append(_add_connection_arrow(ax, c, color, 'solid'))
    for c in mod_connections:
        color = 'black' if c.weight > 0.0 else 'red'
        groups['modulatory'].append(_add_connection_arrow(ax, c, color, 'dashed'))

    for (coord, _) in id_to_coords.items():
        ax.plot(coord[0], coord[1], marker='o', markersize=8.0,
                color='grey', zorder=2)

    # Toggle panel: standard / modulatory connections.
    type_labels = ['standard', 'modulatory']
    type_ax = fig.add_axes([0.02, 0.60, 0.22, 0.18])
    type_ax.set_title("Connections", fontsize=9)
    type_check = CheckButtons(type_ax, type_labels, [True, True])

    def toggle_type(label):
        for patch in groups[label]:
            patch.set_visible(not patch.get_visible())
        fig.canvas.draw_idle()
    type_check.on_clicked(toggle_type)

    # Save button writes the currently visible network to `filename`.
    save_ax = fig.add_axes([0.02, 0.08, 0.15, 0.06])
    save_btn = Button(save_ax, 'Save PNG')
    widget_axes = [type_ax, save_ax]
    save_btn.on_clicked(lambda _event: _save_clean(fig, widget_axes, filename))

    # Persist a default image (all connections visible) even when run headless.
    _save_clean(fig, widget_axes, filename)

    # Keep the widgets alive so their callbacks still fire if the window is
    # displayed later by the caller (they would otherwise be garbage-collected
    # once this function returns).
    fig._interactive_widgets = [type_check, save_btn]

    if show:
        plt.show()
    return fig


def draw_adaptive_des(id_to_coords, std_connections, mod_connections, filename, show=True):
    """
    Draw the net created by Adaptive DES-HyperNEAT.

    Standard connections are solid, modulatory connections are dashed; every
    connection is coloured by the branch it belongs to. Check buttons toggle
    each branch on/off and, separately, the standard and modulatory connections;
    a connection is shown only when both its branch and its type are enabled. A
    Save button writes the currently visible network to `filename`.

    The default image is written to `filename` immediately. When `show` is True
    the interactive window is opened here (blocking); when False the built figure
    is returned so the caller can save other outputs first and display it later
    (e.g. via `plt.show()`).
    """
    all_connections = list(std_connections) + list(mod_connections)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_axes([0.28, 0.08, 0.68, 0.86])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.25, 1.1)
    ax.grid(True)
    ax.set_title("Adaptive DES-HyperNEAT network")

    num_branches = int(max((c.branch_id for c in all_connections), default=-1)) + 1

    def branch_color(branch_id):
        return hue_to_rgb(branch_id / num_branches) if num_branches > 0 else (0.0, 0.0, 0.0)

    # Each entry pairs a patch with the (branch_id, type) that controls it.
    patches = []
    for c in std_connections:
        patch = _add_connection_arrow(ax, c, branch_color(c.branch_id), 'solid')
        patches.append((patch, int(c.branch_id), 'standard'))
    for c in mod_connections:
        patch = _add_connection_arrow(ax, c, branch_color(c.branch_id), 'dashed')
        patches.append((patch, int(c.branch_id), 'modulatory'))

    for (coord, _) in id_to_coords.items():
        ax.plot(coord[0], coord[1], marker='o', markersize=8.0,
                color='grey', zorder=2)

    # Combined visibility state: a connection shows only when both its branch
    # and its type are enabled.
    branch_on = [True] * num_branches
    type_on = {'standard': True, 'modulatory': True}

    def refresh():
        for patch, branch_id, conn_type in patches:
            patch.set_visible(branch_on[branch_id] and type_on[conn_type])
        fig.canvas.draw_idle()

    # Toggle panel: one check button per branch (labels coloured to match).
    branch_labels = [f"branch {i}" for i in range(num_branches)]
    branch_ax = fig.add_axes([0.02, 0.42, 0.22, 0.40])
    branch_ax.set_title("Branches", fontsize=9)
    branch_check = CheckButtons(branch_ax, branch_labels, [True] * num_branches)
    for i, text in enumerate(branch_check.labels):
        text.set_color(branch_color(i))

    def toggle_branch(label):
        branch_on[branch_labels.index(label)] ^= True
        refresh()
    branch_check.on_clicked(toggle_branch)

    # Toggle panel: standard / modulatory connections.
    type_labels = ['standard', 'modulatory']
    type_ax = fig.add_axes([0.02, 0.22, 0.22, 0.14])
    type_ax.set_title("Connections", fontsize=9)
    type_check = CheckButtons(type_ax, type_labels, [True, True])

    def toggle_type(label):
        type_on[label] ^= True
        refresh()
    type_check.on_clicked(toggle_type)

    # Save button writes the currently visible network to `filename`.
    save_ax = fig.add_axes([0.02, 0.08, 0.15, 0.06])
    save_btn = Button(save_ax, 'Save PNG')
    widget_axes = [branch_ax, type_ax, save_ax]
    save_btn.on_clicked(lambda _event: _save_clean(fig, widget_axes, filename))

    # Persist a default image (all connections visible) even when run headless.
    _save_clean(fig, widget_axes, filename)

    # Keep the widgets alive so their callbacks still fire if the window is
    # displayed later by the caller (they would otherwise be garbage-collected
    # once this function returns).
    fig._interactive_widgets = [branch_check, type_check, save_btn]

    if show:
        plt.show()
    return fig


def hue_to_rgb(h):
    """
    Convert a hue value (between 0 and 1) to an RGB list of values between 0 and 1.
    """
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (r, g, b)


def plot_fitness_stats(statistics, filename, title="Fitness over generations"):
    """
    Plot the evolution of the population's fitness across generations.

    Draws the per-generation mean fitness as a line with a +/-1 standard-deviation
    band around it, and overlays a dot for each generation's best fitness. Saves
    the figure to `filename` (no interactive window).

    `statistics` is a neat StatisticsReporter (as returned by the gym runners).
    """
    generations = list(range(len(statistics.most_fit_genomes)))
    best_fitness = [g.fitness for g in statistics.most_fit_genomes]
    mean_fitness = statistics.get_fitness_mean()
    stdev_fitness = statistics.get_fitness_stdev()
    lower = [m - s for m, s in zip(mean_fitness, stdev_fitness)]
    upper = [m + s for m, s in zip(mean_fitness, stdev_fitness)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(generations, lower, upper, color="tab:blue", alpha=0.2,
                    label="±1 std dev")
    ax.plot(generations, mean_fitness, color="tab:blue", linewidth=1.5,
            label="Mean fitness")
    ax.scatter(generations, best_fitness, color="tab:red", s=16, zorder=3,
               label="Best fitness")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)