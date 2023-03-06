import numpy as np

from torch_geometric.data import Batch

from augment.feature.change_attr import change_attr_degree
from augment.feature.mix_attr import mix_attr_degree
from augment.feature.add_noise import add_noise_degree

from augment.structure.drop_edge import drop_edge_degree
from augment.structure.drop_node import drop_node_degree
from augment.structure.subgraph import subgraph_degree


FEATURE = ['change_attr_degree', 'mix_attr_degree', 'add_noise_degree']
STRUCTURE = ['drop_node_degree', 'subgraph_degree', 'drop_edge_degree']


def feature_augment(graphs, augment_ratio):
    augmented_graphs = []
    for graph in graphs:
        aug_pool = np.random.choice(FEATURE)
        aug = augmentation(aug_pool, graph, augment_ratio)
        augmented_graphs.append(aug)

    return augmented_graphs

def structure_augment(graphs, augment_ratio):
    augmented_graphs = []
    for graph in graphs:
        aug_pool = np.random.choice(STRUCTURE)
        aug = augmentation(aug_pool, graph, augment_ratio)
        augmented_graphs.append(aug)

    return augmented_graphs

def augmentation(aug_name, graph, augment_ratio):
    if aug_name.lower() == 'change_attr_degree':
        return change_attr_degree(graph, augment_ratio)
    elif aug_name.lower() == 'mix_attr_degree':
        return mix_attr_degree(graph, augment_ratio)
    elif aug_name.lower() == 'add_noise_degree':
        return add_noise_degree(graph, augment_ratio)
    elif aug_name.lower() == 'drop_node_degree':
        return drop_node_degree(graph, augment_ratio)
    elif aug_name.lower() == 'subgraph_degree':
        return subgraph_degree(graph, augment_ratio)
    elif aug_name.lower() == 'drop_edge_degree':
        return drop_edge_degree(graph, augment_ratio)
