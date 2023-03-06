import os
import argparse
from tqdm import tqdm
from copy import deepcopy
from time import time
import numpy as np

import torch
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Batch
from sklearn.metrics import accuracy_score

from augment import feature_augment, structure_augment
from utils import str2bool
from model.tag import TAG
from evaluate_embedding import evaluate_embedding, evaluate_embedding_mlp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='MUTAG')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--gpu', type=int, default=0)

    # Training setup
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--schedule', type=str2bool, default=True)

    # Classifier
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Augmentation
    parser.add_argument('--aug-ratio', type=float, default=0.4)

    return parser.parse_args()


def main():
    args = parse_args()

    args.seed = args.seed + args.fold
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    os.makedirs('./dataset', exist_ok=True)
    graphs = TUDataset(root='./dataset', name=args.data).shuffle()

    feature_graphs = feature_augment(graphs, args.aug_ratio)
    structure_graphs = structure_augment(graphs, args.aug_ratio)

    dataloader = DataLoader(graphs, batch_size=args.batch_size)
    feature_loader = DataLoader(feature_graphs, batch_size=args.batch_size)
    structure_loader = DataLoader(structure_graphs, batch_size=args.batch_size)

    model = TAG(device, args.hidden, args.layers, graphs.num_features).to(device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # CE_loss = nn.CrossEntropyLoss()

    accuracies = []
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0
        step = (epoch + 1) / args.epochs
        for data, feature_data, structure_data in zip(dataloader, feature_loader, structure_loader):
            data = data.to(device)
            feature_data = feature_data.to(device)
            structure_data = structure_data.to(device)

            out, proj = model(data.x, data.edge_index, data.batch)
            node_out, node_proj = model(feature_data.x, feature_data.edge_index, feature_data.batch)
            graph_out, graph_proj = model(structure_data.x, structure_data.edge_index, structure_data.batch)

            n_to_g = model.forward_graph(node_out, feature_data.edge_index, feature_data.batch)
            g_to_g = model.forward_graph(graph_out, structure_data.edge_index, structure_data.batch)

            loss = model.loss_func(data, feature_data, structure_data, proj, node_proj, n_to_g, g_to_g,
                                   step, is_sup=False)

            optimizer.zero_grad()
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

            if args.schedule:
                scheduler.step(epoch)

        print(f'===== Epoch {epoch}, Loss {loss_sum/len(dataloader)} =====')
        model.eval()
        emb, y = model.get_embeddings(dataloader)
        acc = evaluate_embedding(emb, y)
        accuracies.append(acc)
    print(f'Accuracy: {np.mean(accuracies)}, STD: {np.std(accuracies)}')


if __name__ == '__main__':
    main()