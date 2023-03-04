import os
import torch
import numpy as np
import networkx as nx
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn import metrics
from models import *
from layers import aggregator_lookup
from sklearn.decomposition import PCA
from scipy.sparse import csc_matrix


def top_k_preds(y_true, y_pred):
    top_k_list = np.array(np.sum(y_true, 1), np.int32)
    predictions = []
    for i in range(y_true.shape[0]):
        pred_i = np.zeros(y_true.shape[1])
        pred_i[np.argsort(y_pred[i, :])[-top_k_list[i]:]] = 1
        predictions.append(np.reshape(pred_i, (1, -1)))
    predictions = np.concatenate(predictions, axis=0)
    top_k_array = np.array(predictions, np.int64)

    return top_k_array


def cal_f1_score(y_true, y_pred):
    micro_f1 = metrics.f1_score(y_true, y_pred, average='micro')
    macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

    return micro_f1, macro_f1


def batch_generator(nodes, batch_size, shuffle=True):
    num = nodes.shape[0]
    chunk = num // batch_size
    while True:
        if chunk * batch_size + batch_size > num:
            chunk = 0   
            if shuffle:
                idx = np.random.permutation(num)
        b_nodes = nodes[idx[chunk*batch_size:(chunk+1)*batch_size]]
        chunk += 1

        yield b_nodes


def eval_iterate(nodes, batch_size, shuffle=False):
    idx = np.arange(nodes.shape[0])
    if shuffle:
        idx = np.random.permutation(idx)
    n_chunk = idx.shape[0] // batch_size + 1
    for chunk_id, chunk in enumerate(np.array_split(idx, n_chunk)):
        b_nodes = nodes[chunk]

        yield b_nodes


def do_iter(emb_model, cly_model, adj, feature, labels, idx, cal_f1=False, is_social_net=False):
    embs = emb_model(idx, adj, feature)
    preds = cly_model(embs)
    if is_social_net:
        labels_idx = torch.argmax(labels[idx], dim=1)
        cly_loss = F.cross_entropy(preds, labels_idx)   
    else:
        cly_loss = F.multilabel_soft_margin_loss(preds, labels[idx])
    if not cal_f1:
        return embs, cly_loss
    else:
        targets = labels[idx].cpu().numpy()
        preds = top_k_preds(targets, preds.detach().cpu().numpy())
        return embs, cly_loss, preds, targets


def evaluate(emb_model, cly_model, adj, feature, labels, idx, batch_size, mode='val', is_social_net=False):
    assert mode in ['val', 'test']
    embs, preds, targets = [], [], []
    cly_loss = 0
    for b_nodes in eval_iterate(idx, batch_size):
        embs_per_batch, cly_loss_per_batch, preds_per_batch, targets_per_batch = do_iter(emb_model, cly_model, adj, feature, labels,
                                                                                         b_nodes, cal_f1=True, is_social_net=is_social_net)
        embs.append(embs_per_batch.detach().cpu().numpy())
        preds.append(preds_per_batch)
        targets.append(targets_per_batch)
        cly_loss += cly_loss_per_batch.item()

    cly_loss /= len(preds)
    embs_whole = np.vstack(embs)
    targets_whole = np.vstack(targets)
    micro_f1, macro_f1 = cal_f1_score(targets_whole, np.vstack(preds))

    return cly_loss, micro_f1, macro_f1, embs_whole, targets_whole


def get_split(labels, seed):
    idx_tot = np.arange(labels.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx_tot)

    return idx_tot


def make_adjacency(G, max_degree, seed):
    all_nodes = np.sort(np.array(G.nodes()))
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes, max_degree)) + (n_nodes - 1)).astype(int)
    np.random.seed(seed)
    for node in all_nodes:
        neibs = np.array(G.neighbors(node))
        if len(neibs) == 0:
            neibs = np.array(node).repeat(max_degree)
        elif len(neibs) < max_degree:
            neibs = np.random.choice(neibs, max_degree, replace=True)
        else:
            neibs = np.random.choice(neibs, max_degree, replace=False)
        adj[node, :] = neibs

    return adj


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def pre_social_net(adj, features, labels):
    features = csc_matrix(features.astype(np.uint8))
    labels = labels.astype(np.int32)

    return adj, features, labels


def load_data(file_path="./Datasets", dataset='acmv9.mat', device='cpu', seed=123, is_blog=False):
    # load raw data
    data_path = file_path + '/' + dataset
    data_mat = sio.loadmat(data_path)
    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    if is_blog:
        adj, features, labels = pre_social_net(adj, features, labels)
    features = normalize(features)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_dense = np.array(adj.todense())
    edges = np.vstack(np.where(adj_dense)).T
    Graph = nx.from_edgelist(edges)
    adj = make_adjacency(Graph, 128, seed)
    idx_tot = get_split(labels, seed)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = torch.from_numpy(adj)
    idx_tot = torch.LongTensor(idx_tot)

    return adj.to(device), features.to(device), labels.to(device), idx_tot.to(device)
