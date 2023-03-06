import numpy as np
import math
import torch

from utils import get_degrees


def subgraph_degree(graph, augment_ratio):
    node_num, _ = graph.x.size()
    _, edge_num = graph.edge_index.size()
    subnode_num = node_num - int(node_num * augment_ratio)
    edge_index = graph.edge_index.cpu().numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    degree = get_degrees(graph)

    sorted_degree, indices = torch.sort(degree, descending=True)
    select_indices = indices[:subnode_num]

    if subnode_num > edge_num:
        subnode_num = edge_num
        subnode_idx = edge_index[0]
    else:
        subnode_idx = graph.edge_index[0][select_indices]
    subnode_neigh_idx = set([n for n in edge_index[1][edge_index[0]==subnode_idx[0]]])

    cnt = 0
    while len(subnode_idx) <= subnode_num:
        cnt += 1
        if cnt > subnode_num: break
        if len(subnode_neigh_idx) == 0: break

        new_node = np.random.choice(list(subnode_neigh_idx))
        if new_node in subnode_idx: continue

        subnode_idx.append(new_node)
        subnode_neigh_idx.union(set([n for n in edge_index[1][edge_index[0]==subnode_idx[-1]]]))

    subgraph = subnode_idx
    rest = [n for n in range(node_num) if not n in subnode_idx]

    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[rest, :] = 0
    adj[:, rest] = 0
    edge_index = adj.nonzero().t()

    graph.edge_index = edge_index

    return graph