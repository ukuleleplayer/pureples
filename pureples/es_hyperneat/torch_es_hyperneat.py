"""
PyTorch-based ES-HyperNEAT phenotype network.

This module keeps NEAT-Python for evolution and CPPNs, but builds the
final ES-HyperNEAT substrate network as a PyTorch ``nn.Module`` so that
evaluation can run on modern PyTorch backends (CPU/GPU).

The construction mirrors ``ESNetwork.create_phenotype_network`` but the
returned object is a Torch module instead of ``neat.nn.RecurrentNetwork``.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn

from pureples.es_hyperneat.es_hyperneat import ESNetwork as BaseESNetwork


_ACTIVATIONS = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "relu": torch.relu,
}


class TorchESNetwork(nn.Module):
    """
    ES-HyperNEAT phenotype implemented as a PyTorch module.

    The API is intentionally simple:

    - Construct with the same ``substrate``, ``cppn`` and ``params`` you
      would pass to ``ESNetwork``.
    - Call it with a float tensor of shape ``(batch, n_inputs)`` to get
      a tensor of shape ``(batch, n_outputs)``.

    Internally this composes a regular ``ESNetwork`` instance to perform
    the ES-HyperNEAT quadtree exploration, then converts the resulting
    connections into a Torch computation graph.
    """

    def __init__(self, substrate, cppn, params, device: str | None = None):
        super().__init__()

        # Use the existing pure-Python ESNetwork to discover topology.
        self._base = BaseESNetwork(substrate, cppn, params)

        self.activations: int = self._base.activations
        self.input_nodes: List[int] = []
        self.output_nodes: List[int] = []
        self.num_nodes: int = 0

        self.activation_name: str = params.get("activation", "sigmoid")
        if self.activation_name not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{self.activation_name}' for TorchESNetwork. "
                f"Supported: {sorted(_ACTIVATIONS.keys())}"
            )
        self._activation = _ACTIVATIONS[self.activation_name]

        # Choose device lazily to keep this drop-in for CPU-only setups.
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Adjacency: node index -> list of (source_index, weight)
        self._node_to_incoming: Dict[int, List[Tuple[int, float]]] = {}
        self._eval_order: List[int] = []

        self._build_from_es_hyperneat()

    def _build_from_es_hyperneat(self) -> None:
        """
        Run ES-HyperNEAT exploration and convert the resulting coordinates
        and connections into node indices and incoming edge lists.
        """
        substrate = self._base.substrate
        input_coordinates = substrate.input_coordinates
        output_coordinates = substrate.output_coordinates

        # Assign contiguous node indices: inputs first, then outputs, then hidden.
        self.input_nodes = list(range(len(input_coordinates)))
        self.output_nodes = list(
            range(len(self.input_nodes), len(self.input_nodes) + len(output_coordinates))
        )
        next_hidden_idx = len(self.input_nodes) + len(self.output_nodes)

        # Map coordinates to node ids.
        coords_to_id: Dict[Tuple[float, float], int] = {}
        for idx, coord in enumerate(input_coordinates):
            coords_to_id[tuple(coord)] = self.input_nodes[idx]
        for idx, coord in enumerate(output_coordinates):
            coords_to_id[tuple(coord)] = self.output_nodes[idx]

        # ES-HyperNEAT exploration to get hidden nodes and connections.
        hidden_nodes, connections = self._base.es_hyperneat()

        for (x, y) in hidden_nodes:
            coords_to_id[(x, y)] = next_hidden_idx
            next_hidden_idx += 1

        self.num_nodes = next_hidden_idx

        node_to_incoming: Dict[int, List[Tuple[int, float]]] = {}
        for c in connections:
            dst = coords_to_id[(c.x2, c.y2)]
            src = coords_to_id[(c.x1, c.y1)]
            node_to_incoming.setdefault(dst, []).append((src, c.weight))

        # Evaluation order: simple ascending node index is sufficient because
        # ES-HyperNEAT constructs a feedforward topology (no cycles).
        eval_order = sorted(node_to_incoming.keys())

        self._node_to_incoming = node_to_incoming
        self._eval_order = eval_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ES-HyperNEAT phenotype.

        - ``x``: tensor of shape ``(batch, n_inputs)``
        - returns: tensor of shape ``(batch, n_outputs)``

        To mirror the original ``neat.nn.RecurrentNetwork`` behaviour used
        in this project, we apply the network ``self.activations`` times,
        updating all non-input nodes each iteration.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)
        batch_size, in_features = x.shape

        if in_features != len(self.input_nodes):
            raise ValueError(
                f"Expected input with {len(self.input_nodes)} features, got {in_features}."
            )

        # Full state vector includes inputs, outputs, and hidden nodes.
        state = x.new_zeros(batch_size, self.num_nodes, device=self.device)
        state[:, self.input_nodes] = x

        # Repeated activation cycles, as in the original implementation.
        for _ in range(self.activations):
            new_state = state.clone()
            for node in self._eval_order:
                links = self._node_to_incoming[node]
                # Accumulate weighted input from all predecessors.
                acc = 0.0
                for src, w in links:
                    acc = acc + state[:, src] * w
                new_state[:, node] = self._activation(acc)
            state = new_state

        return state[:, self.output_nodes]

