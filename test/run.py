from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import dgl.nn as dglnn
import dgl
import math
from dgl import from_networkx
import torch.nn as nn
import torch as th
import torch
import torch.nn.functional as F
import dgl.function as fn
import networkx as nx
import pandas as pd
import socket
import struct
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import gc
import os
import torch.nn.init as init
from dgl.nn import GATConv
from dgl.nn.functional import edge_softmax

# Set random seeds for reproducibility
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True

# Load data
data = pd.read_csv('NF-BoT-IoT-v2.csv')

# Comment out IP randomization for consistent results
# data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))

data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)
data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']
data.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT'], inplace=True)

# ========== BINARY CLASSIFICATION MODIFICATION ==========
# Convert multi-class to binary: 0 = Normal, 1 = Attack
print("Original label distribution:")
print(data.Attack.value_counts())

# Create binary labels
data['binary_label'] = data['Attack'].apply(
    lambda x: 0 if x == 'Normal' else 1)

print("\nBinary label distribution:")
print(data.binary_label.value_counts())

# Drop original labels
data.drop(columns=['Label', 'Attack'], inplace=True)
data.rename(columns={"binary_label": "label"}, inplace=True)

label = data.label
data_features = data.drop(columns=['label'])

# Standardization
scaler = StandardScaler()
data = pd.concat([data_features, label], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, label, test_size=0.3, random_state=123, stratify=label)

# Encoding
encoder = ce.TargetEncoder(cols=['TCP_FLAGS', 'L7_PROTO', 'PROTOCOL'])
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)

cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns)) - set(['label']))
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_train['h'] = X_train[cols_to_norm].values.tolist()

# ========== MODEL DEFINITIONS (unchanged) ==========


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(th.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class GATlayer(nn.Module):
    def __init__(self, n_feat, e_feat, out_feat, num_heads):
        super(GATlayer, self).__init__()
        self.n_feat = n_feat
        self.e_feat = e_feat
        self.out_feat = out_feat
        self.num_heads = num_heads
        self.W_msg = nn.Linear(2 * n_feat + e_feat, out_feat)
        self.W = nn.Linear(2 * n_feat + e_feat, 2 * out_feat)
        self.a = nn.Parameter(torch.rand(size=(2 * out_feat, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = math.sqrt(2)
        init.xavier_normal_(self.W.weight, gain=gain)
        init.xavier_normal_(self.a, gain=gain)

    def edge_attention(self, edges):
        feat_cat = torch.cat(
            [edges.src['h'], edges.dst['h'], edges.data['h']], dim=1)
        w_feat_cat = self.W(feat_cat)
        return {'e': F.leaky_relu(torch.matmul(w_feat_cat, self.a))}

    def message_func(self, edges):
        return {'h': self.W_msg(torch.cat([edges.src['h'], edges.dst['h'], edges.data['h']], dim=1)), 'x': edges.data['x']}

    def reduce_func(self, nodes):
        h = (nodes.mailbox['x'] * nodes.mailbox['h']).sum(1)
        return {'h': h}

    def forward(self, g, n_feat, e_feat):
        with g.local_scope():
            g.ndata['h'] = n_feat
            g.edata['h'] = e_feat
            g.apply_edges(self.edge_attention)
            attention = edge_softmax(g, g.edata['e'])
            g.edata['x'] = attention
            g.update_all(self.message_func, self.reduce_func)
            g.ndata['h'] = F.relu(g.ndata['h'])
            feat = g.ndata['h']
            return feat


class MultiHeadGATLayer(nn.Module):
    def __init__(self, n_feat, e_feat, out_feat, num_heads):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATlayer(n_feat, e_feat, out_feat, num_heads))

    def forward(self, g, h, e_feat):
        out_feat = [attn_head(g, h, e_feat) for attn_head in self.heads]
        out_feat = torch.cat(out_feat, dim=1).reshape(
            g.num_nodes(), len(self.heads), -1)
        return out_feat.mean(1)


class GAT(nn.Module):
    def __init__(self, in_dim, e_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, e_dim, 39, num_heads)

    def forward(self, g, h, e_feat):
        h = self.layer1(g, h, e_feat)
        g.ndata['h'] = h
        return h, g


class Genetation(torch.nn.Module):
    def __init__(self, in_feat, out_feat, num_heads, activation):
        super(Genetation, self).__init__()
        self.conv = GATConv(in_feat, out_feat, num_heads)
        self.activation = activation

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, g, feat):
        x = self.activation(self.conv(g, feat))
        g.ndata['h'] = x
        return x.mean(1), g


def sub_sam(nodes, adj_lists, k):
    node_neighbor = [[] for i in range(nodes.shape[0])]
    node_neighbor_cen = [[] for i in range(nodes.shape[0])]
    node_centorr = [[] for i in range(nodes.shape[0])]
    num_nei = 0

    for node in nodes:
        neighbors = set([int(node)])
        neighs = adj_lists[int(node)]
        node_centorr[num_nei] = [int(node)]
        current1 = adj_lists[int(node)]
        if len(neighs) >= k:
            neighs -= neighbors
            current1 = random.sample(neighs, k-1)
            node_neighbor[num_nei] = [neg_node for neg_node in current1]
            current1.append(int(node))
            node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]
            num_nei += 1
        node_neighbor_cen[num_nei] = [neg_node for neg_node in current1]

    node_neighbor_cen = [
        neighbors for neighbors in node_neighbor_cen if neighbors]
    node_neighbor_cen = node_neighbor_cen[:-1]
    return node_neighbor_cen


def b_xent(logits, labels):
    """Binary cross entropy loss function"""
    return nn.BCEWithLogitsLoss()(logits, labels)


class Model(nn.Module):
    def __init__(self, Encoder, gene, tau=0.5):
        super(Model, self).__init__()
        self.encoder = Encoder
        self.tau: float = tau
        self.ge = gene

    def forward(self, graph, node_feats, edge_feats):
        z, g1 = self.encoder(graph, node_feats, edge_feats)
        z_g, g2 = self.ge(graph, z, edge_feats)
        return z, z_g, g1, g2

    def embed(self, graph, node_feats, edge_feats):
        z, _ = self.encoder(graph, node_feats, edge_feats)
        return z

    def loss(self, z1, z2, adj, sub_g1, g1, g2):
        loss = self.sub_loss_batch(z1, z2, adj, sub_g1, g1, g2)
        return loss

    def sub_loss_batch(self, z, z_g, adj, sub_g1, g1, g2):
        subz_s, sub_gene_s = self.subg_centor(z, z_g, sub_g1)
        num = th.randint(0, len(sub_g1)-1, [len(sub_g1),])
        if num[0] == 0:
            num[0] = 1
        for i in range(1, len(num)):
            if num[i] == i:
                num[i] -= 1
        subg2_s_n = subz_s[num]
        sub_gene_s_n = sub_gene_s[num]

        input1 = th.cat((subz_s, subz_s, subz_s), dim=0)
        input2 = th.cat((sub_gene_s, subg2_s_n, sub_gene_s_n), dim=0)

        edges1, edges2 = self.edges_f(g1, g2, sub_g1, z, z_g)
        subg2_se = edges1[num]
        sub_gene_s_e = edges2[num]
        input1_edges = th.cat((edges1, edges1, edges1), dim=0)
        input2_edges = th.cat((edges2, subg2_se, sub_gene_s_e), dim=0)
        input1_edges = input1_edges.requires_grad_(True)
        input2_edges = input2_edges.requires_grad_(True)

        subg1_adj = self.sub_adj(adj, sub_g1)
        input_adj = th.cat((subg1_adj, subg1_adj, subg1_adj), dim=0)

        lbl_1 = th.ones(len(sub_g1)).cuda()
        lbl_2 = th.zeros(len(sub_g1)*2).cuda()
        lbl = th.cat((lbl_1, lbl_2), 0).cuda()

        lbl_1_e = th.ones(len(edges2)).cuda()
        lbl_2_e = th.zeros(len(edges2)*2).cuda()
        lbl_e = th.cat((lbl_1_e, lbl_2_e), 0).cuda()

        wd, T_wd = self.wd(input1, input2, self.tau)
        logits = th.exp(-wd / 0.01)
        loss1 = b_xent(th.squeeze(logits), lbl)

        gwd = self.gwd(input1.transpose(2, 1), input2.transpose(
            2, 1), T_wd, input_adj, self.tau)
        logits2 = th.exp(-gwd / 0.1)
        loss2 = b_xent(th.squeeze(logits2), lbl)

        wd, T_wd = self.wd(input1_edges, input2_edges, self.tau)
        logits3 = th.exp(-wd / 0.01)
        loss3 = b_xent(th.squeeze(logits3), lbl_e)

        loss = 0.5 * loss3 + 0.5 * loss2
        return loss

    def edges_f(self, g1, g2, sub_g1, z, z_g):
        edge_feat_1 = [[] for i in range(len(sub_g1))]
        edge_feat_2 = [[] for i in range(len(sub_g1))]
        sc = MLPPredictor(g1.edata['h'].shape[1], 39).cuda()
        z_e = sc(g1, z)
        z_ge = sc(g2, z_g)
        for i in range(len(sub_g1)):
            cen_node = sub_g1[i][-1]
            dst = sub_g1[i][:-1]
            src_node_id = cen_node
            for j in dst:
                dst_node_id = j
                edge_indices = g1.edge_ids(
                    src_node_id, dst_node_id, return_uv=True)
                edge_feature_1 = torch.Tensor(
                    z_e[edge_indices[2]]).float().tolist()
                edge_feature_2 = torch.Tensor(
                    z_ge[edge_indices[2]]).float().tolist()
                edge_feat_1.append(edge_feature_1)
                edge_feat_2.append(edge_feature_2)
                if len(edge_feat_1[-1]) == 2:
                    edge_feat_1[-1] = [edge_feat_1[-1][0]]
                    edge_feat_2[-1] = [edge_feat_2[-1][0]]
        edge_feat_1 = [neighbors for neighbors in edge_feat_1 if neighbors]
        edge_feat_2 = [neighbors for neighbors in edge_feat_2 if neighbors]
        edge_feat_1 = torch.Tensor(edge_feat_1)
        edge_feat_2 = torch.Tensor(edge_feat_2)
        edge_feat_1 = edge_feat_1.reshape(len(sub_g1), -1, 39)
        edge_feat_2 = edge_feat_2.reshape(len(sub_g1), -1, 39)
        return edge_feat_1, edge_feat_2

    def sub_adj(self, adj, sub_g1):
        subg1_adj = th.zeros(len(sub_g1), len(sub_g1[0]), len(sub_g1[0]))
        for i in range(len(sub_g1)):
            subg1_adj[i] = adj[sub_g1[i]].t()[sub_g1[i]]
        return subg1_adj

    def subg_centor(self, z, z_g, sub_g1):
        sub = [element for lis in sub_g1 for element in lis]
        subz = z[sub]
        subg = z_g[sub]
        sub_s = subz.reshape(len(sub_g1), len(sub_g1[0]), -1)
        subg_s = subg.reshape(len(sub_g1), len(sub_g1[0]), -1)
        return sub_s, subg_s

    def wd(self, x, y, tau):
        cos_distance = self.cost_matrix_batch(
            th.transpose(x, 2, 1), th.transpose(y, 2, 1), tau)
        cos_distance = cos_distance.transpose(1, 2)
        beta = 0.1
        min_score = cos_distance.min()
        max_score = cos_distance.max()
        threshold = min_score + beta * (max_score - min_score)
        cos_dist = nn.functional.relu(cos_distance - threshold)
        wd, T_wd = self.OT_distance_batch(
            cos_dist, x.size(0), x.size(1), y.size(1), 40)
        return wd, T_wd

    def OT_distance_batch(self, C, bs, n, m, iteration=50):
        C = C.float().cuda()
        T = self.OT_batch(C, bs, n, m, iteration=iteration)
        temp = th.bmm(th.transpose(C, 1, 2), T)
        distance = self.batch_trace(temp, m, bs)
        return distance, T

    def OT_batch(self, C, bs, n, m, beta=0.5, iteration=50):
        sigma = th.ones(bs, int(m), 1).cuda() / float(m)
        T = th.ones(bs, n, m).cuda()
        A = th.exp(-C / beta).float().cuda()
        for t in range(iteration):
            Q = A * T
            for k in range(1):
                delta = 1 / (n * th.bmm(Q, sigma))
                a = th.bmm(th.transpose(Q, 1, 2), delta)
                sigma = 1 / (float(m) * a)
            T = delta * Q * sigma.transpose(2, 1)
        return T

    def cost_matrix_batch(self, x, y, tau=0.5):
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(th.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(th.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = th.bmm(th.transpose(x, 1, 2), y)
        cos_dis = th.exp(-cos_dis / tau)
        return cos_dis.transpose(2, 1)

    def batch_trace(self, input_matrix, n, bs):
        a = th.eye(n).cuda().unsqueeze(0).repeat(bs, 1, 1)
        b = a * input_matrix
        return th.sum(th.sum(b, -1), -1).unsqueeze(1)

    def gwd(self, X, Y, T_wd, input_adj, tau, lamda=1e-1, iteration=5, OT_iteration=20):
        m = X.size(2)
        n = Y.size(2)
        bs = X.size(0)
        p = (th.ones(bs, m, 1) / m).cuda()
        q = (th.ones(bs, n, 1) / n).cuda()
        return self.GW_distance(X, Y, p, q, T_wd, input_adj, tau, lamda=lamda, iteration=iteration, OT_iteration=OT_iteration)

    def GW_distance(self, X, Y, p, q, T_wd, input_adj, tau, lamda=0.5, iteration=5, OT_iteration=20):
        cos_dis = th.exp(-input_adj / tau).cuda()
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        Cs = nn.functional.relu(res.transpose(2, 1))
        Ct = self.cos_batch(Y, Y, tau).float().cuda()
        bs = Cs.size(0)
        m = Ct.size(2)
        n = Cs.size(2)
        T, Cst = self.GW_batch(Cs, Ct, bs, n, m, p, q, beta=lamda,
                               iteration=iteration, OT_iteration=OT_iteration)
        temp = th.bmm(th.transpose(Cst, 1, 2), T_wd)
        distance = self.batch_trace(temp, m, bs)
        return distance

    def GW_batch(self, Cs, Ct, bs, n, m, p, q, beta=0.5, iteration=5, OT_iteration=20):
        one_m = th.ones(bs, m, 1).float().cuda()
        one_n = th.ones(bs, n, 1).float().cuda()
        Cst = th.bmm(th.bmm(Cs**2, p), th.transpose(one_m, 1, 2)) + \
            th.bmm(one_n, th.bmm(th.transpose(q, 1, 2), th.transpose(Ct**2, 1, 2)))
        gamma = th.bmm(p, q.transpose(2, 1))
        for i in range(iteration):
            C_gamma = Cst - 2 * \
                th.bmm(th.bmm(Cs, gamma), th.transpose(Ct, 1, 2))
            gamma = self.OT_batch(
                C_gamma, bs, n, m, beta=beta, iteration=OT_iteration)
        Cgamma = Cst - 2 * th.bmm(th.bmm(Cs, gamma), th.transpose(Ct, 1, 2))
        return gamma.detach(), Cgamma

    def cos_batch(self, x, y, tau):
        bs = x.size(0)
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)
        x = x.div(th.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(th.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = th.bmm(th.transpose(x, 1, 2), y)
        cos_dis = th.exp(-cos_dis / tau).transpose(1, 2)
        beta = 0.1
        min_score = cos_dis.min()
        max_score = cos_dis.max()
        threshold = min_score + beta * (max_score - min_score)
        res = cos_dis - threshold
        return nn.functional.relu(res.transpose(2, 1))


# ========== TEST DATA PREPARATION ==========
X_test = encoder.transform(X_test)
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[cols_to_norm].values.tolist()

# Build test graph
G_test = nx.from_pandas_edgelist(X_test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
                                 ['h', 'label'], create_using=nx.MultiGraph())
G_test = G_test.to_directed()
G_test = from_networkx(G_test, edge_attrs=['h', 'label'])

G_test.ndata['feature'] = th.ones(
    G_test.num_nodes(), G_test.edata['h'].shape[1])

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
G_test = G_test.to(device)

print(f"\nTest graph info:")
print(f"Number of nodes: {G_test.num_nodes()}")
print(f"Number of edges: {G_test.num_edges()}")
print(f"Device: {G_test.device}")

# ========== LOAD MODEL AND TEST ==========
print("\nLoading model...")
try:
    # Try loading full model
    model = torch.load('model-botv2.pth', map_location=device)
    log = torch.load('log-botv2.pth', map_location=device)
except AttributeError as e:
    print(f"Error loading full model: {e}")
    print("Trying to load state_dict instead...")

    # If that fails, try loading state dict
    # You'll need to recreate the model architecture first
    # Adjust these parameters based on your actual model
    in_dim = len(cols_to_norm)
    e_dim = len(cols_to_norm)
    num_heads = 4

    encoder = GAT(in_dim, e_dim, 39, num_heads)
    gene = Genetation(39, 39, num_heads, F.relu)
    model = Model(encoder, gene)

    # Create a simple classifier (you may need to adjust architecture)
    class Classifier(nn.Module):
        def __init__(self, in_dim, num_classes=2):
            super().__init__()
            self.fc = nn.Linear(in_dim, num_classes)

        def forward(self, g, h):
            return self.fc(h)

    log = Classifier(39, 2)  # 2 for binary classification

    # Load state dicts
    model.load_state_dict(torch.load('model-botv2.pth', map_location=device))
    log.load_state_dict(torch.load('log-botv2.pth', map_location=device))

model = model.to(device)
log = log.to(device)

model.eval()
log.eval()

print("\nGenerating embeddings...")
with torch.no_grad():
    test_embs = model.embed(G_test, G_test.ndata['feature'], G_test.edata['h'])
    test_lbls = G_test.edata['label']

    logits = log(G_test, test_embs)

    # For binary classification
    if logits.shape[1] == 2:
        # Multi-class output, take argmax
        preds = th.argmax(logits, dim=1)
    else:
        # Single output, apply sigmoid
        preds = (torch.sigmoid(logits) > 0.5).long().squeeze()

preds = preds.cpu().numpy()
test_lbls = test_lbls.cpu().numpy()

print("\nPrediction distribution:")
print(f"Predicted Normal (0): {(preds == 0).sum()}")
print(f"Predicted Attack (1): {(preds == 1).sum()}")

# ========== EVALUATION ==========

print("\n" + "="*60)
print("BINARY CLASSIFICATION RESULTS")
print("="*60)

accuracy = accuracy_score(test_lbls, preds)
precision = precision_score(test_lbls, preds)
recall = recall_score(test_lbls, preds)
f1 = f1_score(test_lbls, preds)

print(f"\nOverall Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Try to compute AUC if probabilities are available
try:
    if logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    else:
        probs = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(test_lbls, probs)
    print(f"ROC-AUC:   {auc:.4f}")
except:
    print("ROC-AUC:   Not available")

print("\n" + "-"*60)
print("Detailed Classification Report:")
print("-"*60)
target_names = ['Normal', 'Attack']
print(classification_report(test_lbls, preds, target_names=target_names, digits=4))

# ========== CONFUSION MATRIX ==========


def plot_confusion_matrix(cm, target_names, title='Confusion matrix',
                          cmap=None, normalize=True):
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


cm = confusion_matrix(test_lbls, preds)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

plot_confusion_matrix(cm=cm,
                      normalize=False,
                      target_names=target_names,
                      title="Binary Classification - Confusion Matrix")

# ========== ADDITIONAL ANALYSIS ==========
print("\n" + "="*60)
print("DETECTION PERFORMANCE")
print("="*60)

# Attack detection rate (Recall for Attack class)
attack_detection_rate = recall
print(
    f"Attack Detection Rate: {attack_detection_rate:.4f} ({attack_detection_rate*100:.2f}%)")

# False Positive Rate
fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
print(f"False Positive Rate:   {fpr:.4f} ({fpr*100:.2f}%)")

# False Negative Rate
fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
print(f"False Negative Rate:   {fnr:.4f} ({fnr*100:.2f}%)")

print("\n" + "="*60)
