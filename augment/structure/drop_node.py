import numpy as np
import torch

from utils import get_degrees


def drop_node_degree(graph, augment_ratio):
    node_num, _ = graph.x.size()
    _, edge_num = graph.edge_index.size()
    drop_num = int(node_num * augment_ratio)
    edge_index = graph.edge_index.cpu().numpy()
    degree = get_degrees(graph)

    sorted_degree, indices = torch.sort(degree)
    select_indices = indices[:drop_num]

    if drop_num > edge_num:
        drop_num = edge_num
        drop_idx = edge_index[0]
    else:
        drop_idx = edge_index[0][select_indices]

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[drop_idx, :] = 0
    adj[:, drop_idx] = 0
    edge_index = adj.nonzero().t()

    graph.edge_index = edge_index

    return graph
