import numpy as np
import torch

from utils import get_degrees


def drop_edge_degree(graph, augment_ratio):
    node_num, _ = graph.x.size()
    _, edge_num = graph.edge_index.cpu().size()
    drop_num = int(edge_num * augment_ratio)
    select_num = edge_num - drop_num
    degree = get_degrees(graph)

    sorted_degree, indices = torch.sort(degree)

    edge_index = graph.edge_index.cpu().transpose(0, 1).numpy()
    idx_add = np.random.choice(node_num, (drop_num, 2))

    select_idx = np.empty(select_num, dtype=np.int)
    drop_cnt, select_cnt = 0, 0
    for i in range(len(indices)):
        for j in range(len(edge_index)):
            if indices[i] == edge_index[j][0]:
                if drop_cnt < drop_num:
                    drop_cnt = drop_cnt + 1
                else:
                    select_idx[select_cnt] = j
                    select_cnt = select_cnt + 1

    edge_index = edge_index[select_idx]
    graph.edge_index = torch.tensor(edge_index).transpose(0, 1)

    return graph