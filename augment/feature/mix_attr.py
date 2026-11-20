"""
Accurate graph classification via two-staged contrastive curriculum learning (PLOS ONE 2024)

Authors:
- Sooyeon Shim (syshim77@snu.ac.kr), Seoul National University
- Junghun Kim (bandalg97@snu.ac.kr), Seoul National University
- Kahyun Park (kahyun.park@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software is free of charge under research purposes.

For other purposes (e.g. commercial), please contact the authors.

mix_attr.py
    - feature augmentation that mixes node features based on node degrees
"""

import numpy as np
import torch

from utils import get_degrees


def mix_attr_degree(graph, augment_ratio):
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
    mix_idx = np.random.choice(node_num, change_num, replace=False)

    graph.x[change_idx] = (graph.x[change_idx] + graph.x[mix_idx]) / 2

    return graph
