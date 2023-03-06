import numpy as np
import torch

from utils import get_degrees


def add_noise_degree(graph, augment_ratio):
    node_num, feature_dim = graph.x.size()
    edge_index = graph.edge_index.cpu().numpy()
    degree = get_degrees(graph)
    change_num = int(node_num * augment_ratio)

    sorted_degree, indices = torch.sort(degree)
    select_indices = indices.numpy()[:change_num]

    if change_num > graph.edge_index.shape[1]:
        change_num = graph.edge_index.shape[1]
        change_idx = edge_index[0]
    else:
        change_idx = edge_index[0][select_indices]
    graph.x[change_idx] = graph.x[change_idx] + torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(change_num, feature_dim), ),
                                       dtype=torch.float32)

    return graph
