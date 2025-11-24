from DyGLib.evaluate_models_utils import evaluate_model_link_prediction, evaluate_edge_bank_link_prediction
from DyGLib.utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from DyGLib.utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from DyGLib.utils.metrics import get_link_prediction_metrics
from DyGLib.utils.DataLoader import get_idx_data_loader
from DyGLib.utils.EarlyStopping import EarlyStopping
from DyGLib.models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from DyGLib.models.GraphMixer import GraphMixer
from DyGLib.models.DyGFormer import DyGFormer
from DyGLib.models.modules import MergeLayer
from DyGLib.models.CAWN import CAWN
from DyGLib.models.TGAT import TGAT
from DyGLib.models.TCL import TCL
from sklearn.decomposition import PCA
import torch.nn as nn
import torch
from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np
import warnings
import logging
import random
import shutil
import json
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Configurations
MODEL = "DyGFormer"  # options = ["TGAT", "JODIE", "DyRep", "GraphMixer", "DyGFormer"]
DOMAINS = ["Political_Science", "Philosophy", "Economics", "Business", "Psychology", "Mathematics", "Medicine",
           "Biology", "Computer_Science", "Geology", "Chemistry", "Art", "Sociology", "Engineering", "Geography",
           "History", "Materials_Science", "Physics", "Environmental_Science"]


node_embeddings_dir = os.path.join("..", "FOS_Benchmark", "node_embeddings", "_".join(DOMAINS))
edges_dir = os.path.join("..", "FOS_Benchmark", "edges", "_".join(DOMAINS))

nodes = pd.read_pickle(os.path.join(node_embeddings_dir, "full_features.pkl"))
edges = pd.read_csv(os.path.join(edges_dir, "all_edges.csv"))

assert np.all(np.diff(edges['ts']) >= 0), "Edges timestamps are not sorted!"
print("edges are sorted, min ts:",
      edges['ts'].min(), "max ts:", edges['ts'].max())
print("edges.shape:", edges.shape)
print("nodes.shape:", nodes.shape)

node_list = nodes["node_id"].to_list()
id2idx = {nid: i for i, nid in enumerate(node_list)}
edges["src"] = edges["src"].map(id2idx)
edges["dst"] = edges["dst"].map(id2idx)

t_train = edges["ts"].quantile(0.7)
t_val = edges["ts"].quantile(0.85)
print(t_train, t_val)

train_edges_df = edges[edges["ts"] <= t_train]
train_edges = list(zip(train_edges_df["src"], train_edges_df["dst"], [
                   {"ts": ts} for ts in train_edges_df["ts"].to_list()]))

G_train = nx.Graph()
G_train.add_nodes_from(range(len(node_list)))
G_train.add_edges_from(train_edges)

train_val_edges_df = edges[edges["ts"] <= t_val]
train_val_edges = list(zip(train_val_edges_df["src"], train_val_edges_df["dst"], [
                       {"ts": ts} for ts in train_val_edges_df["ts"].to_list()]))

G_train_val = nx.Graph()
G_train_val.add_nodes_from(range(len(node_list)))
G_train_val.add_edges_from(train_val_edges)

val_edges_df = edges[(edges["ts"] > t_train) & (edges["ts"] <= t_val)]
# mask = val_edges_df[["src", "dst"]].apply(tuple, 1).isin(train_edges_df[["src", "dst"]].apply(tuple, 1))
# val_edges_df = val_edges_df[~mask]
val_edges = list(zip(val_edges_df["src"], val_edges_df["dst"], [
                 {"ts": ts} for ts in val_edges_df["ts"].to_list()]))

G_val = nx.Graph()
G_val.add_nodes_from(range(len(node_list)))
G_val.add_edges_from(val_edges)

test_edges_df = edges[edges["ts"] > t_val]
# mask = test_edges_df[["src", "dst"]].apply(tuple, 1).isin(train_edges_df[["src", "dst"]].apply(tuple, 1))
# test_edges_df = test_edges_df[~mask]
# mask = test_edges_df[["src", "dst"]].apply(tuple, 1).isin(val_edges_df[["src", "dst"]].apply(tuple, 1))
# test_edges_df = test_edges_df[~mask]
test_edges = list(zip(test_edges_df["src"], test_edges_df["dst"], [
                  {"ts": ts} for ts in test_edges_df["ts"].to_list()]))

G_test = nx.Graph()
G_test.add_nodes_from(range(len(node_list)))
G_test.add_edges_from(test_edges)

possible_nodes = list(range(len(node_list)))


def sample_ts_non_edges(df):
    df = df.copy()
    df['label'] = 1
    new_rows = []
    for key, group in df.groupby(['src', 'ts']):
        src, ts = key
        used_dst = set(group['dst'])
        available = [n for n in possible_nodes if n not in used_dst]
        num_needed = len(group) * 20
        if len(available) < num_needed:
            selected_dst = random.sample(available, len(available))
        else:
            selected_dst = random.sample(available, num_needed)
        for dst in selected_dst:
            new_rows.append({'src': src, 'dst': dst, 'ts': ts, 'label': 0})
    new_df = pd.DataFrame(new_rows)
    return pd.concat([df, new_df], ignore_index=True)


# train_df = sample_ts_non_edges(train_edges_df)
# val_df = sample_ts_non_edges(val_edges_df)
# test_df = sample_ts_non_edges(test_edges_df)
train_df = train_edges_df.copy()
train_df['label'] = 1
assert np.all(np.diff(train_df['ts']) >=
              0), "train_df timestamps are not sorted!"
val_df = val_edges_df.copy()
val_df['label'] = 1
test_df = test_edges_df.copy()
test_df['label'] = 1

del edges, G_val, G_test


class Data:
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


full_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
assert np.all(np.diff(full_df['ts']) >=
              0), "full_df timestamps are not sorted!"

full_df["idx"] = full_df.index

edge_raw_features = np.zeros((len(full_df)+1, 1))

features = []
for i in range(1, 769):
    features.append(nodes["embeddings"].apply(
        lambda x: x[i-1]).rename(f"f{i}"))
full_features = pd.concat(features, axis=1).values
node_raw_features = PCA(n_components=100).fit_transform(full_features)

full_data = Data(src_node_ids=full_df["src"].values.astype(np.longlong),
                 dst_node_ids=full_df["dst"].values.astype(np.longlong),
                 node_interact_times=full_df["ts"].values.astype(np.longlong),
                 edge_ids=full_df["idx"].values.astype(np.longlong),
                 labels=full_df["label"].values.astype(np.longlong))

train_data = Data(src_node_ids=train_df["src"].values.astype(np.longlong),
                  dst_node_ids=train_df["dst"].values.astype(np.longlong),
                  node_interact_times=train_df["ts"].values.astype(
                      np.longlong),
                  edge_ids=np.array(range(len(train_df))),
                  labels=train_df["label"].values.astype(np.longlong))
assert np.all(np.diff(train_data.node_interact_times) >=
              0), "train_data timestamps are not sorted!"

val_data = Data(src_node_ids=val_df["src"].values.astype(np.longlong),
                dst_node_ids=val_df["dst"].values.astype(np.longlong),
                node_interact_times=val_df["ts"].values.astype(np.longlong),
                edge_ids=np.array(
                    range(len(train_df), len(train_df)+len(val_df))),
                labels=val_df["label"].values.astype(np.longlong))

test_data = Data(src_node_ids=test_df["src"].values.astype(np.longlong),
                 dst_node_ids=test_df["dst"].values.astype(np.longlong),
                 node_interact_times=test_df["ts"].values.astype(np.longlong),
                 edge_ids=np.array(
                     range(len(train_df)+len(val_df), len(train_df)+len(val_df)+len(test_df))),
                 labels=test_df["label"].values.astype(np.longlong))

old_nodes = set(train_df["src"].unique()).union(set(train_df["dst"].unique()))
nn_val_df = val_df[~val_df['src'].isin(
    old_nodes) & ~val_df['dst'].isin(old_nodes)]
nn_test_df = test_df[~test_df['src'].isin(
    old_nodes) & ~test_df['dst'].isin(old_nodes)]

# validation and test with edges that at least has one new node (not in training set)
new_node_val_data = Data(src_node_ids=nn_val_df["src"].values.astype(np.longlong),
                         dst_node_ids=nn_val_df["dst"].values.astype(
                             np.longlong),
                         node_interact_times=nn_val_df["ts"].values.astype(
                             np.longlong),
                         edge_ids=np.array(range(len(train_df), len(train_df)+len(val_df))).astype(
                             np.longlong)[~val_df['src'].isin(old_nodes) & ~val_df['dst'].isin(old_nodes)],
                         labels=nn_val_df["label"].values.astype(np.longlong))

new_node_test_data = Data(src_node_ids=nn_test_df["src"].values.astype(np.longlong),
                          dst_node_ids=nn_test_df["dst"].values.astype(
                              np.longlong),
                          node_interact_times=nn_test_df["ts"].values.astype(
                              np.longlong),
                          edge_ids=np.array(range(len(train_df)+len(val_df), len(train_df)+len(val_df)+len(test_df))).astype(
                              np.longlong)[~test_df['src'].isin(old_nodes) & ~test_df['dst'].isin(old_nodes)],
                          labels=nn_test_df["label"].values.astype(np.longlong))

print("The dataset has {} interactions, involving {} different nodes".format(
    full_data.num_interactions, full_data.num_unique_nodes))
print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.num_interactions, train_data.num_unique_nodes))
print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.num_interactions, val_data.num_unique_nodes))
print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.num_interactions, test_data.num_unique_nodes))
print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.num_interactions, new_node_val_data.num_unique_nodes))
print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.num_interactions, new_node_test_data.num_unique_nodes))

# print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

# return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data


class Args:
    def __init__(self):
        self.dataset_name = 'FOS'
        self.batch_size = 300
        self.model_name = 'TGAT'
        self.gpu = 0
        self.num_neighbors = 20
        self.sample_neighbor_strategy = 'recent'
        self.time_scaling_factor = 1e-6
        self.num_walk_heads = 8
        self.num_heads = 2
        self.num_layers = 2
        self.walk_length = 1
        self.time_gap = 2000
        self.time_feat_dim = 100
        self.position_feat_dim = 172
        self.edge_bank_memory_mode = 'unlimited_memory'
        self.time_window_mode = 'fixed_proportion'
        self.patch_size = 1
        self.channel_embedding_dim = 50
        self.max_input_sequence_length = 32
        self.learning_rate = 0.0001
        self.dropout = 0.1
        self.num_epochs = 30
        self.optimizer = 'Adam'
        self.weight_decay = 0.0
        self.patience = 20
        self.val_ratio = 0.15
        self.test_ratio = 0.15
        self.num_runs = 1
        self.test_interval_epochs = 10
        self.negative_sample_strategy = 'random'
        self.load_best_configs = False
        self.device = f'cuda:{self.gpu}' if torch.cuda.is_available(
        ) and self.gpu >= 0 else 'cpu'


def run_model(model_name):
    args = Args()
    args.model_name = model_name
    warnings.filterwarnings('ignore')

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(
        len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(
        len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(
                    train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                   time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                          time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                                     learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                    batch_src_node_embeddings = torch.nan_to_num(
                        batch_src_node_embeddings, nan=0.0)
                    batch_dst_node_embeddings = torch.nan_to_num(
                        batch_dst_node_embeddings, nan=0.0)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                    batch_neg_src_node_embeddings = torch.nan_to_num(
                        batch_neg_src_node_embeddings, nan=0.0)
                    batch_neg_dst_node_embeddings = torch.nan_to_num(
                        batch_neg_dst_node_embeddings, nan=0.0)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)
                else:
                    raise ValueError(
                        f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](
                    input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](
                    input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(
                    positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(
                    predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after training so it can be used for new validation nodes
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank(
                )

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = model[0].memory_bank.backup_memory_bank(
                )

                # reload training memory bank for new validation nodes
                model[0].memory_bank.reload_memory_bank(
                    train_backup_memory_bank)

        #   new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                               model=model,
        #                                                                               neighbor_sampler=full_neighbor_sampler,
        #                                                                               evaluate_idx_data_loader=new_node_val_idx_data_loader,
        #                                                                               evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
        #                                                                               evaluate_data=new_node_val_data,
        #                                                                               loss_func=loss_func,
        #                                                                               num_neighbors=args.num_neighbors,
        #                                                                               time_gap=args.time_gap)

            if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                # reload validation memory bank for testing nodes or saving models
                # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
        #   logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        #   for metric_name in new_node_val_metrics[0].keys():
        #       logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for new testing nodes
                    model[0].memory_bank.reload_memory_bank(
                        val_backup_memory_bank)

              #   new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
              #                                                                                 model=model,
              #                                                                                 neighbor_sampler=full_neighbor_sampler,
              #                                                                                 evaluate_idx_data_loader=new_node_test_idx_data_loader,
              #                                                                                 evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
              #                                                                                 evaluate_data=new_node_test_data,
              #                                                                                 loss_func=loss_func,
              #                                                                                 num_neighbors=args.num_neighbors,
              #                                                                                 time_gap=args.time_gap)

                if args.model_name in ['JODIE', 'DyRep', 'TGN']:
                    # reload validation memory bank for testing nodes or saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model[0].memory_bank.reload_memory_bank(
                        val_backup_memory_bank)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(
                        f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
              #   logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
              #   for metric_name in new_node_test_metrics[0].keys():
              #       logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean(
                    [val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

        #   new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                               model=model,
        #                                                                               neighbor_sampler=full_neighbor_sampler,
        #                                                                               evaluate_idx_data_loader=new_node_val_idx_data_loader,
        #                                                                               evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
        #                                                                               evaluate_data=new_node_val_data,
        #                                                                               loss_func=loss_func,
        #                                                                               num_neighbors=args.num_neighbors,
        #                                                                               time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # the memory in the best model has seen the validation edges, we need to backup the memory for new testing nodes
            val_backup_memory_bank = model[0].memory_bank.backup_memory_bank()

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        if args.model_name in ['JODIE', 'DyRep', 'TGN']:
            # reload validation memory bank for new testing nodes
            model[0].memory_bank.reload_memory_bank(val_backup_memory_bank)

        #   new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
        #                                                                                 model=model,
        #                                                                                 neighbor_sampler=full_neighbor_sampler,
        #                                                                                 evaluate_idx_data_loader=new_node_test_idx_data_loader,
        #                                                                                 evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
        #                                                                                 evaluate_data=new_node_test_data,
        #                                                                                 loss_func=loss_func,
        #                                                                                 num_neighbors=args.num_neighbors,
        #                                                                                 time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean(
                    [val_metric[metric_name] for val_metric in val_metrics])
                logger.info(
                    f'validate {metric_name}, {average_val_metric:.4f}')
                val_metric_dict[metric_name] = average_val_metric

        #   logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            # for metric_name in new_node_val_metrics[0].keys():
            #     average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            #     logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
            #     new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean(
                [test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        #   logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        #   for metric_name in new_node_test_metrics[0].keys():
        #       average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
        #       logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
        #       new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            val_metric_all_runs.append(val_metric_dict)
        #   new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        #   new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
            result_json = {
                "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
                #   "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                #   "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        else:
            result_json = {
                "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
                #   "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    if args.model_name not in ['JODIE', 'DyRep', 'TGN']:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
            logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                        f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

        #   for metric_name in new_node_val_metric_all_runs[0].keys():
        #       logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
        #       logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
        #                   f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

#   for metric_name in new_node_test_metric_all_runs[0].keys():
#       logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
#       logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
#                   f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')


if __name__ == "__main__":
    run_model(model_name=MODEL)
