import functools
import operator

import numpy as np
import torch
import torch.nn as nn
from torchdrug.data import PackedGraph

from chemicalx.data import DrugPairBatch
from chemicalx.models import Model
from chemicalx.utils import segment_softmax

__all__ = [
    "MPModule",
]


class MessagePassing(nn.Module):
    def __init__(self, node_channels: int, edge_channels: int, hidden_channels: int, dropout: float = 0.5):
        super().__init__()
        # add 
        self.MMLP = MultiMLPModule(node_channels, hidden_channels)
        self.node_projection=self.MMLP(node_channels, hidden_channels)
        self.node_projection = nn.Sequential(
            nn.Linear(node_channels, hidden_channels, bias=False),
            nn.Dropout(dropout),
        )
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        nodes: torch.FloatTensor,
        edges: torch.FloatTensor,
        segmentation_index: torch.LongTensor,
        index: torch.LongTensor,
    ) -> torch.FloatTensor:
        edges = self.edge_projection(edges)
        messages = self.node_projection(nodes)
        messages = self.message_composing(messages, edges, index)
        messages = self.message_aggregation(nodes, messages, segmentation_index)
        return messages

    def message_composing(
        self, messages: torch.FloatTensor, edges: torch.FloatTensor, index: torch.LongTensor
    ) -> torch.FloatTensor:
        messages = messages.index_select(0, index)
        messages = messages * edges
        return messages

    def message_aggregation(
        self, nodes: torch.FloatTensor, messages: torch.FloatTensor, segmentation_index: torch.LongTensor
    ) -> torch.FloatTensor:
        messages = torch.zeros_like(nodes).index_add(0, segmentation_index, messages)
        return messages


class MultiMLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(MultiMLPModule, self).__init__()
        self.block1 = self.build_block(input_dim, hidden_dims[0], 3)  # 三个MLP层
        self.block2 = self.build_block(hidden_dims[0], hidden_dims[1], 2)  # 两个MLP层
        self.block3 = self.build_block(hidden_dims[1], hidden_dims[2], 1)  # 一个MLP层
        self.final_fc = nn.Linear(hidden_dims[2],  1)  # 全连接层

    def build_block(self, input_dim, output_dim, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU(inplace=True))  # 可以使用其他激活函数
            input_dim = output_dim
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.final_fc(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.1):
        super().__init__()
        self.temperature = np.sqrt(input_channels)

        self.key_projection = nn.Linear(input_channels, input_channels, bias=False)
        self.value_projection = nn.Linear(input_channels, input_channels, bias=False)

        nn.init.xavier_normal_(self.key_projection.weight)
        nn.init.xavier_normal_(self.value_projection.weight)

        self.attention_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

        self.out_projection = nn.Sequential(
            nn.Linear(input_channels, output_channels), nn.LeakyReLU(), nn.Dropout(dropout)
        )

    def _calculate_message(
        self,
        translation: torch.Tensor,
        segmentation_number: torch.Tensor,
        segmentation_index: torch.Tensor,
        index: torch.Tensor,
        node: torch.Tensor,
        node_hidden_channels: torch.Tensor,
        node_neighbor: torch.Tensor,
    ):
        """Calculate the outer message."""
        node_edge = self.attention_dropout(
            segment_softmax(translation, segmentation_number, segmentation_index, index, self.temperature)
        )
        node_edge = node_edge.view(-1, 1)
        message = node.new_zeros((segmentation_number, node_hidden_channels)).index_add(
            0, segmentation_index, node_edge * node_neighbor
        )
        message_graph = self.out_projection(message)
        return message_graph

    def forward(
        self,
        node_left: torch.FloatTensor,
        segmentation_index_left: torch.LongTensor,
        index_left: torch.LongTensor,
        node_right: torch.FloatTensor,
        segmentation_index_right: torch.LongTensor,
        index_right: torch.LongTensor,
    ):
        node_left_hidden_channels = node_left.size(1)
        node_right_hidden_channels = node_right.size(1)

        segmentation_number_left = node_left.size(0)
        segmentation_number_right = node_right.size(0)

        node_left_center = self.key_projection(node_left).index_select(0, segmentation_index_left)
        node_right_center = self.key_projection(node_right).index_select(0, segmentation_index_right)

        node_left_neighbor = self.value_projection(node_right).index_select(0, segmentation_index_right)
        node_right_neighbor = self.value_projection(node_left).index_select(0, segmentation_index_left)

        translation = (node_left_center * node_right_center).sum(1)

        message_graph_left = self._calculate_message(
            translation,
            segmentation_number_left,
            segmentation_index_left,
            index_left,
            node_left,
            node_left_hidden_channels,
            node_left_neighbor,
        )
        message_graph_right = self._calculate_message(
            translation,
            segmentation_number_right,
            segmentation_index_right,
            index_right,
            node_right,
            node_right_hidden_channels,
            node_right_neighbor,
        )

        return message_graph_left, message_graph_right


class MultiHeadAttentionMessagePassingNetwork(nn.Module):
    """Multiheadattention message passing layer."""

    def __init__(
        self,
        hidden_channels: int,
        readout_channels: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.message_passing = MessagePassing(
            node_channels=hidden_channels,
            edge_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

        self.co_attention = MultiHeadAttention(
            input_channels=hidden_channels,
            output_channels=hidden_channels,
            dropout=dropout,
        )

        self.linear = nn.LayerNorm(hidden_channels)
        self.leaky_relu = nn.LeakyReLU()

        self.prediction_readout_projection = nn.Linear(hidden_channels, readout_channels)

    def _get_graph_features(
        self,
        atom_features: torch.Tensor,
        inner_message: torch.Tensor,
        outer_message: torch.Tensor,
        segmentation_molecule: torch.Tensor,
    ):
        """Get the graph representations."""
        message = atom_features + inner_message + outer_message
        message = self.linear(message)
        graph_features = self.readout(message, segmentation_molecule)
        return graph_features

    def forward(
        self,
        segmentation_molecule_left: torch.Tensor,
        atom_left: torch.Tensor,
        bond_left: torch.Tensor,
        inner_segmentation_index_left: torch.Tensor,
        inner_index_left: torch.Tensor,
        outer_segmentation_index_left: torch.Tensor,
        outer_index_left: torch.Tensor,
        segmentation_molecule_right: torch.Tensor,
        atom_right: torch.Tensor,
        bond_right: torch.Tensor,
        inner_segmentation_index_right: torch.Tensor,
        inner_index_right: torch.Tensor,
        outer_segmentation_index_right: torch.Tensor,
        outer_index_right: torch.Tensor,
    ):
        outer_message_left, outer_message_right = self.co_attention(
            atom_left,
            outer_segmentation_index_left,
            outer_index_left,
            atom_right,
            outer_segmentation_index_right,
            outer_index_right,
        )

        inner_message_left = self.message_passing(atom_left, bond_left, inner_segmentation_index_left, inner_index_left)
        inner_message_right = self.message_passing(
            atom_right, bond_right, inner_segmentation_index_right, inner_index_right
        )
        graph_left = self._get_graph_features(
            atom_left, inner_message_left, outer_message_left, segmentation_molecule_left
        )
        graph_right = self._get_graph_features(
            atom_right, inner_message_right, outer_message_right, segmentation_molecule_right
        )
        return graph_left, graph_right

    def readout(self, atom_features: torch.Tensor, segmentation_molecule: torch.Tensor):
        """Aggregate node features.

        :param atom_features: Atom embeddings.
        :param segmentation_molecule: Molecular segmentation index.
        :returns: Graph readout vectors.
        """
        segmentation_max = segmentation_molecule.max() + 1

        atom_features = self.leaky_relu(self.prediction_readout_projection(atom_features))
        hidden_channels = atom_features.size(1)

        readout_vectors = atom_features.new_zeros((segmentation_max, hidden_channels)).index_add(
            0, segmentation_molecule, atom_features
        )
        return readout_vectors


class MPModule(Model):

    def __init__(
        self,
        *,
        atom_feature_channels: int = 16,
        atom_type_channels: int = 16,
        bond_type_channels: int = 16,
        node_channels: int = 16,
        edge_channels: int = 16,
        hidden_channels: int = 16,
        readout_channels: int = 16,
        output_channels: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.atom_projection = nn.Linear(node_channels + atom_feature_channels, node_channels)
        self.atom_embedding = nn.Embedding(atom_type_channels, node_channels, padding_idx=0)
        self.bond_embedding = nn.Embedding(bond_type_channels, edge_channels, padding_idx=0)
        nn.init.xavier_normal_(self.atom_embedding.weight)
        nn.init.xavier_normal_(self.bond_embedding.weight)

        self.encoder = MultiHeadAttentionMessagePassingNetwork(
            hidden_channels=hidden_channels,
            readout_channels=readout_channels,
            dropout=dropout,
        )

        self.head_layer = nn.Linear(readout_channels * 2, output_channels)

    def _get_molecule_features(
        self,
        drug_molecules_primary: PackedGraph,
        drug_molecules_secondary: PackedGraph,
    ):
        outer_segmentation_index, outer_index = self.generate_outer_segmentation(
            drug_molecules_primary.num_nodes, drug_molecules_secondary.num_nodes
        )
        atom = self.dropout(self.atom_comp(drug_molecules_primary.node_feature, drug_molecules_primary.atom_type))
        bond = self.dropout(self.bond_embedding(drug_molecules_primary.bond_type))
        return outer_segmentation_index, outer_index, atom, bond

    def unpack(self, batch: DrugPairBatch):
        """Adjust drug pair batch to model design.

        :param batch: Molecular data in a drug pair batch.
        :returns: Tuple of data.
        """
        return (
            batch.drug_molecules_left,
            batch.drug_molecules_right,
        )

    def forward(
        self,
        drug_molecules_left: PackedGraph,
        drug_molecules_right: PackedGraph,
    ) -> torch.FloatTensor:
        """Forward pass with the data."""
        outer_segmentation_index_left, outer_index_left, atom_left, bond_left = self._get_molecule_features(
            drug_molecules_left, drug_molecules_right
        )
        outer_segmentation_index_right, outer_index_right, atom_right, bond_right = self._get_molecule_features(
            drug_molecules_right, drug_molecules_left
        )

        drug_left, drug_right = self.encoder(
            drug_molecules_left.node2graph,
            atom_left,
            bond_left,
            drug_molecules_left.edge_list[:, 0],
            drug_molecules_left.edge_list[:, 1],
            outer_segmentation_index_left,
            outer_index_left,
            drug_molecules_right.node2graph,
            atom_right,
            bond_right,
            drug_molecules_right.edge_list[:, 0],
            drug_molecules_right.edge_list[:, 1],
            outer_segmentation_index_right,
            outer_index_right,
        )

        prediction_left = self.head_layer(torch.cat([drug_left, drug_right], dim=1))
        prediction_right = self.head_layer(torch.cat([drug_right, drug_left], dim=1))
        prediction_mean = (prediction_left + prediction_right) / 2
        return torch.sigmoid(prediction_mean)

    def atom_comp(self, atom_features: torch.Tensor, atom_index: torch.Tensor):
        """Compute atom projection, a linear transformation of a learned atom embedding and the atom features.

        :param atom_features: Atom input features
        :param atom_index: Index of atom type
        :returns: Node index.
        """
        atom_embedding = self.atom_embedding(atom_index)
        node_embedding = self.atom_projection(torch.cat([atom_embedding, atom_features], -1))
        return node_embedding

    def generate_outer_segmentation(self, graph_sizes_left: torch.LongTensor, graph_sizes_right: torch.LongTensor):
        interactions = graph_sizes_left * graph_sizes_right

        left_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_left, 0) - graph_sizes_left
        shift_sums_left = torch.repeat_interleave(left_shifted_graph_size_cum_sum, interactions)
        outer_segmentation_index = [
            np.repeat(np.array(range(0, left_graph_size)), right_graph_size)
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        outer_segmentation_index = functools.reduce(operator.iconcat, outer_segmentation_index, [])
        outer_segmentation_index = torch.tensor(outer_segmentation_index) + shift_sums_left

        right_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_right, 0) - graph_sizes_right
        shift_sums_right = torch.repeat_interleave(right_shifted_graph_size_cum_sum, interactions)
        outer_index = [
            list(range(0, right_graph_size)) * left_graph_size
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        outer_index = functools.reduce(operator.iconcat, outer_index, [])
        outer_index = torch.tensor(outer_index) + shift_sums_right
        return outer_segmentation_index, outer_index
