import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class TAG(torch.nn.Module):
    def __init__(self, device, h_dim, num_layers, num_features, dropout=0.5):
        super(TAG, self).__init__()

        layers = []
        for i in range(num_layers):
            h_in = num_features if i == 0 else h_dim
            layers.append(GCNConv(h_in, h_dim))
        self.layers = torch.nn.ModuleList(layers)
        self.linear = nn.Linear(h_dim, h_dim)
        self.dropout = dropout
        self.device = device

    def forward(self, x, edge_index, batch):
        out = x
        for i in range(len(self.layers)):
            out = F.relu(self.layers[i](out, edge_index))
            out = F.dropout(out, self.dropout, self.training)

        return out, self.linear(out)

    def forward_graph(self, x, edge_index, batch):
        out = global_mean_pool(x, batch)

        return self.linear(out)

    def get_embeddings(self, loader):
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(self.device)
                x, _ = self.forward(x, edge_index, batch)
                x = self.forward_graph(x, edge_index, batch)
                ret.append(x.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

    def node_loss(self, x, x_neg, step):
        T = 0.2

        node_num, _ = x.size()
        neg_node_num, _ = x_neg.size()
        x_abs = x.norm(dim=1)
        x_neg_abs = x_neg.norm(dim=1)

        pos_sim_matrix = torch.einsum('ik,jk->ij', x, x_neg) / torch.einsum('i,j->ij', x_abs, x_neg_abs)
        pos_sim_matrix = torch.exp(pos_sim_matrix / T)
        pos_sim = pos_sim_matrix[range(node_num), range(node_num)]

        intra_sim_matrix = torch.einsum('ik,jk->ij', x, x) / torch.einsum('i,j->ij', x_abs, x_abs)
        intra_sim_matrix = torch.exp(intra_sim_matrix / T)
        intra_sim = intra_sim_matrix[range(node_num), range(node_num)]

        neg_inter = pos_sim_matrix.sum(dim=1) / (node_num - 1)
        neg_intra = (intra_sim_matrix.sum(dim=1) - intra_sim) / (node_num - 1)

        scores = pos_sim / (neg_inter + neg_intra)
        scores, _ = torch.sort(scores)

        # define the pacing function
        lamb = 2  # 1/2, 1, 2
        select_num = int(step ** lamb * node_num)
        select_idx = torch.tensor(np.arange(select_num)).to(self.device)
        loss = torch.index_select(scores, -1, select_idx)
        loss = - torch.log(loss).sum()

        return loss

    def node_loss_sum(self, batch, x, x_neg, step):
        node_loss = []
        end_idx = 0
        batch_size = batch.max() + 1
        for i in range(batch_size):
            start_idx = end_idx
            end_idx = end_idx + (batch == torch.ones(batch.size(), dtype=int).mul(i).to(self.device)).sum()
            node_loss.append(self.node_loss(x[start_idx:end_idx], x_neg[start_idx:end_idx], step))

        node_loss = torch.tensor(node_loss, requires_grad=True).to(self.device).mean()

        return node_loss

    def graph_loss(self, x, x_neg, step):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_neg_abs = x_neg.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_neg) / torch.einsum('i,j->ij', x_abs, x_neg_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        neg_sim = sim_matrix.sum(dim=1) - pos_sim

        scores = pos_sim / neg_sim
        scores, _ = torch.sort(scores)

        # define the pacing function
        lamb = 2  # 1/2, 1, 2
        select_num = int(step ** lamb * batch_size)
        select_idx = torch.tensor(np.arange(select_num)).to(self.device)
        loss = torch.index_select(scores, -1, select_idx)
        loss = - torch.log(loss).mean()

        return loss

    def loss_func(self, data, feat_aug, struct_aug, node_out, feat_aug_out, graph_aug_out, struct_aug_out, step, is_sup=False):
        if is_sup is False:
            node_loss = self.node_loss_sum(data.batch, node_out, feat_aug_out, step)
            graph_loss = self.graph_loss(graph_aug_out, struct_aug_out, step)
        else:
            node_loss = self.node_loss_sum(data.batch, node_out, feat_aug_out, step)
            graph_loss = self.graph_sup_loss(feat_aug, struct_aug, graph_aug_out, struct_aug_out, step)

        loss = (node_loss + graph_loss) / 2

        return loss

