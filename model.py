import torch
from torch.nn import Linear

#from torch_geometric.data import Data
#from torch_geometric.nn.conv.gcn_conv import gcn_norm

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, Sequential, dense_mincut_pool
from torch_geometric.nn.dense import dense_mincut_pool

#from torch_geometric.utils import to_dense_adj
from torch_geometric import utils

from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize


# simple vanilla model:
class Net(torch.nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()
        
        mp_act = getattr(torch.nn, mp_act)(inplace=True)
        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # Message passing layers
        mp = [
            (GraphConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
            mp_act
        ]
        for i in range(len(mp_units)-1):
            mp.append((GraphConv(mp_units[i], mp_units[i+1]), 'x, edge_index, edge_weight -> x'))
            mp.append(mp_act)
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        out_chan = mp_units[-1]
        
        # MLP layers
        self.mlp = torch.nn.Sequential()
        for units in mlp_units:
            self.mlp.append(Linear(out_chan, units))
            out_chan = units
            self.mlp.append(mlp_act)
        self.mlp.append(Linear(out_chan, n_clusters))
        

    def forward(self, x, edge_index, edge_weight, batch):
        
        # Propagate node feats
        x = self.mp(x, edge_index, edge_weight) 
        
        # Cluster assignments (logits)
        s = self.mlp(x) 

        # Dense adjacency
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)

        B = batch.max().item() + 1
        N = adj.size(-1)
        x_padded = x.new_zeros((B, N, x.size(-1)))
        s_padded = x.new_zeros((B, N, s.size(-1)))
        for i in range(B):
            node_indices = (batch == i).nonzero(as_tuple=True)[0]
            n_nodes = node_indices.size(0)
            x_padded[i, :n_nodes] = x[node_indices]
            s_padded[i, :n_nodes] = s[node_indices]

        # 4. Pool
        x_pool, adj_pool, mc_loss, o_loss = dense_mincut_pool(x_padded, adj, s_padded)
        
        # Obtain MinCutPool losses
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        _, _, mc_loss, o_loss = dense_mincut_pool(x, adj, s)
        
        #return torch.softmax(s, dim=-1), mc_loss, o_loss
        return s, mc_loss, o_loss



# multi-stage added
class Net_second(nn.Module):
    def __init__(self, 
                 mp_units,
                 mp_act,
                 in_channels, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity",
                 dropout=0.1):
        super().__init__()
        
        mp_act_cls = getattr(nn, mp_act)
        mlp_act_cls = getattr(nn, mlp_act)

        # Handle single-stage input for backward compatibility
        if isinstance(mp_units[0], int):
            mp_units = [mp_units]
        if isinstance(mlp_units[0], int):
            mlp_units = [mlp_units]
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters] * len(mp_units)

        self.num_stages = len(mp_units)
        self.mp_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.encode_layers = nn.ModuleList()

        current_in_channels = in_channels

        for stage_idx in range(self.num_stages):
            stage_mp_units = mp_units[stage_idx]
            stage_mlp_units = mlp_units[stage_idx]
            stage_clusters = n_clusters[stage_idx]

            # --- Message Passing ---
            mp = [
                (GraphConv(current_in_channels, stage_mp_units[0]), 'x, edge_index, edge_weight -> x'),
                mp_act_cls(inplace=True)
            ]
            for i in range(len(stage_mp_units) - 1):
                mp.append((GraphConv(stage_mp_units[i], stage_mp_units[i + 1]), 
                           'x, edge_index, edge_weight -> x'))
                mp.append(mp_act_cls(inplace=True))
            self.mp_layers.append(Sequential('x, edge_index, edge_weight', mp))

            # --- MLP for assignments ---
            out_channels = stage_mp_units[-1]
            mlp = nn.Sequential()
            for units in stage_mlp_units:
                mlp.append(nn.Linear(out_channels, units))
                mlp.append(mlp_act_cls(inplace=True))
                if dropout > 0:
                    mlp.append(nn.Dropout(p=dropout))
                out_channels = units
            mlp.append(nn.Linear(out_channels, stage_clusters))
            self.mlp_layers.append(mlp)

            # --- Encoder for x_part ---
            encode = nn.Sequential()
            in_dim = current_in_channels
            for unit in stage_mp_units:
                encode.append(nn.Linear(in_dim, unit))
                encode.append(nn.BatchNorm1d(unit))
                encode.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    encode.append(nn.Dropout(p=dropout))
                in_dim = unit
            self.encode_layers.append(encode)

            # For next stage
            current_in_channels = stage_mp_units[-1]

    def forward(self, x, x_part, edge_index, edge_weight, batch, batch_part):
        mc_loss_total = 0.0
        o_loss_total = 0.0
        for stage in range(self.num_stages):
            # 1. Message Passing
            x = self.mp_layers[stage](x, edge_index, edge_weight)
            x_part = self.encode_layers[stage](x_part)

            # 2. Cluster Assignments
            s = self.mlp_layers[stage](x)

            # 3. Dense adjacency
            adj = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)

            B = batch.max().item() + 1
            N = adj.size(-1)
            x_padded = x.new_zeros((B, N, x.size(-1)))
            s_padded = x.new_zeros((B, N, s.size(-1)))
            for i in range(B):
                node_indices = (batch == i).nonzero(as_tuple=True)[0]
                n_nodes = node_indices.size(0)
                x_padded[i, :n_nodes] = x[node_indices]
                s_padded[i, :n_nodes] = s[node_indices]

            # 4. Pool
            x_pool, adj_pool, mc_loss, o_loss = dense_mincut_pool(x_padded, adj, s_padded)

            #print("Losses:\t", mc_loss, o_loss)

            mc_loss_total += mc_loss
            o_loss_total += o_loss

            # 5. Create new edges
            adj_list, edge_index_list, edge_weight_list = [], [], []
            node_offset = 0
            for g in range(B):
                mask_x = (batch == g)
                mask_part = (batch_part == g)
                x_b = x[mask_x]
                x_part_b = x_part[mask_part]
                x_norm = F.normalize(x_b, p=2, dim=1)
                x_part_norm = F.normalize(x_part_b, p=2, dim=1)
                cos_mat_dist = x_norm @ x_part_norm.T
                adj_2 = cos_mat_dist @ cos_mat_dist.T
                adj_list.append(adj_2)
                e_idx = (adj_2 > 0).nonzero(as_tuple=False).t()
                e_wt = adj_2[e_idx[0], e_idx[1]]
                e_idx_global = e_idx + node_offset
                edge_index_list.append(e_idx_global)
                edge_weight_list.append(e_wt)
                node_offset += mask_x.sum().item()

            edge_index = torch.cat(edge_index_list, dim=1)
            edge_weight = torch.cat(edge_weight_list, dim=0)

        #print("Total_losse:\t", mc_loss_total, o_loss_total)

        return torch.softmax(s, dim=-1), mc_loss_total, o_loss_total


#below i modified code by me but stage is yet to be added

# class Net(nn.Module):
#     def __init__(self, 
#                  mp_units,
#                  mp_act,
#                  in_channels, 
#                  n_clusters, 
#                  mlp_units=[],
#                  mlp_act="Identity",
#                  dropout=0.1):
#         super().__init__()
        
#         mp_act = getattr(nn, mp_act)(inplace=True)
#         mlp_act = getattr(nn, mlp_act)(inplace=True)

#         # Message Passing Layers

#         # i am doing only one MP and then dense_mincut_pool(), so below code not required.
#         mp = [
#             (GraphConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
#             mp_act
#         ]
        
#         for i in range(len(mp_units) - 1):
#             mp.append((GraphConv(mp_units[i], mp_units[i + 1]), 'x, edge_index, edge_weight -> x'))
#             mp.append(mp_act)


#         self.mp = Sequential('x, edge_index, edge_weight', mp)
        
        
#         out_channels = mp_units[-1]
#         # MLP Layers for Assignment Matrix
#         self.mlp = nn.Sequential()
#         for units in mlp_units:
#             self.mlp.append(nn.Linear(out_channels, units))
#             self.mlp.append(mlp_act)
#             if dropout > 0:
#                 self.mlp.append(nn.Dropout(p=dropout))
#             out_channels = units
#         self.mlp.append(nn.Linear(out_channels, n_clusters))

#         # for x_part
#         self.encode = nn.Sequential()
#         in_dim = in_channels
#         for unit in mp_units:
#             self.encode.append(nn.Linear(in_dim, unit))
#             self.encode.append(nn.BatchNorm1d(unit))
#             self.encode.append(nn.ReLU(inplace=True))
#             if dropout > 0:
#                 self.encode.append(nn.Dropout(p=dropout))
#             in_dim = unit
    
#     def forward(self, x, x_part, edge_index, edge_weight, batch, batch_part):

#         print(x.shape, x_part.shape)

#         # 1. Message Passing
#         x = self.mp(x, edge_index, edge_weight)
#         x_part = self.encode(x_part)
#         print(x.shape, x_part.shape) #torch.Size([sum_of_all_nodes_in_batch, 32])
#         # 2. Cluster Assignments
#         s = self.mlp(x)  
#         print(s.shape) # [sum_of_all_nodes_in_batch, num_clusters]

#         # 3. Convert sparse edge_index to dense adjacency with batching
#         adj = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)  # [batch_size, N, N]
#         print("adj:", adj.shape)
        
#         # 4. Reshape node features and assignment matrix to match dense input
#         B = batch.max().item() + 1  # batch size (it is like 0, 0, 0, 1,1,1, 2, 3, 3, 4, 4, 4, 5, 5 ,6, 6, 6, 6, 7, 7, 7) so need to add 1
#         N = adj.size(-1)  # padded size (max num nodes in batch)

#         # Pad x and s to [B, N, ...]
#         # x_padded = torch.zeros((B, N, x.size(-1)), device=x.device)
#         # s_padded = torch.zeros((B, N, s.size(-1)), device=x.device)
#         x_padded = x.new_zeros((B, N, x.size(-1)))
#         s_padded = x.new_zeros((B, N, s.size(-1)))
#         print(x_padded.shape, s_padded.shape) #torch.Size([8, 278, 32]) torch.Size([8, 278, 2])

#         for i in range(B):
#             node_indices = (batch == i).nonzero(as_tuple=True)[0]
#             n_nodes = node_indices.size(0)
#             x_padded[i, :n_nodes] = x[node_indices]
#             s_padded[i, :n_nodes] = s[node_indices]
        
#         print(x_padded.shape, s_padded.shape) #torch.Size([8, 278, 32]) torch.Size([8, 278, 2])

#         # 5. Apply MinCut Pooling
#         x_pool, adj_pool, mc_loss, o_loss = dense_mincut_pool(x_padded, adj, s_padded)


#         print("x_pool and adj shapes are:", x.shape, x_part.shape, adj.shape, edge_index.shape, edge_weight.shape, x_pool.shape, adj_pool.shape)

#         # creating next 
#         # L2 normalize on GPU (( ss scaler did not work here))
#         adj_list = []
#         edge_index_list = []
#         edge_weight_list = []

#         node_offset = 0  # tracks global node indexing

#         for g in range(B):
#             mask_x = (batch == g)  # get nodes belonging to this graph
#             mask_part = (batch_part == g)
#             x_b = x[mask_x]
#             x_part_b = x_part[mask_part]

#             x_norm = torch.nn.functional.normalize(x_b, p=2, dim=1)
#             x_part_norm = torch.nn.functional.normalize(x_part_b, p=2, dim=1)
            
#             cos_mat_dist = x_norm @ x_part_norm.T
#             adj_2 = cos_mat_dist @ cos_mat_dist.T
            
#             # store dense adjacency
#             adj_list.append(adj_2)

#             # convert to edge index + weight
#             edge_index = (adj_2 > 0).nonzero(as_tuple=False).t()
#             edge_weight = adj_2[edge_index[0], edge_index[1]]

#             # shift local indices to global indices
#             edge_index_global = edge_index + node_offset

#             edge_index_list.append(edge_index_global)
#             edge_weight_list.append(edge_weight)

#             # update offset for next graph
#             node_offset += mask_x.sum().item()

#         # concatenate like PyG does
#         edge_index_new = torch.cat(edge_index_list, dim=1)  # shape [2, total_edges]
#         edge_weight_new = torch.cat(edge_weight_list, dim=0)  # shape [total_edges]

#         print("Next iteration:", x.shape, x_part.shape, adj_2.shape, edge_index_new.shape, edge_weight_new.shape) #adj_2 is from the last loop

#         # 6. Return softmax scores per node and losses
#         return torch.softmax(s, dim=-1), mc_loss, o_loss

#trying different way to indice adjacency matrix

# class Net(nn.Module):
#     def __init__(self, 
#                  mp_units,
#                  mp_act,
#                  in_channels, 
#                  n_clusters, 
#                  mlp_units=[],
#                  mlp_act="Identity",
#                  dropout=0.1):
#         super().__init__()
        
#         mp_act = getattr(nn, mp_act)(inplace=True)
#         mlp_act = getattr(nn, mlp_act)(inplace=True)

#         # Message Passing Layers

#         # i am doing only one MP and then dense_mincut_pool(), so below code not required.
#         # mp = [
#         #     (GraphConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
#         #     mp_act
#         # ]
        
#         # for i in range(len(mp_units) - 1):
#         #     mp.append((GraphConv(mp_units[i], mp_units[i + 1]), 'x, edge_index, edge_weight -> x'))
#         #     mp.append(mp_act)


#         # so instead this:
#         # First MP
#         self.mp1 = Sequential('x, edge_index, edge_weight', [
#             (GraphConv(in_channels, mp_units[0]), 'x, edge_index, edge_weight -> x'),
#             mp_act
#         ])

#         # Second MP
#         self.mp2 = Sequential('x, edge_index, edge_weight', [
#             (GraphConv(mp_units[0], mp_units[1]), 'x, edge_index, edge_weight -> x'),
#             mp_act
#         ])

#         # Projection for x_part
#         self.part_proj1 = nn.Linear(in_channels, mp_units[0])
#         self.part_proj2 = nn.Linear(mp_units[0], mp_units[1])



#         #below was for commenetd MP layer
#         #self.mp = Sequential('x, edge_index, edge_weight', mp)
        
#         #out_channels = mp_units[-1]

#         out_channels = mp_units[1] # picked second layer because heer we are using only 2 MP layer

#         # MLP Layers for Assignment Matrix
#         self.mlp = nn.Sequential()
#         for units in mlp_units:
#             self.mlp.append(nn.Linear(out_channels, units))
#             self.mlp.append(mlp_act)
#             if dropout > 0:
#                 self.mlp.append(nn.Dropout(p=dropout))
#             out_channels = units
#         self.mlp.append(nn.Linear(out_channels, n_clusters))

#     def forward(self, x, edge_index, edge_weight, batch):
#         # 1. Message Passing
#         x = self.mp(x, edge_index, edge_weight)
#         #print(x.shape)
#         # 2. Cluster Assignments
#         s = self.mlp(x)  # [num_nodes_total, num_clusters]

#         # 3. Convert sparse edge_index to dense adjacency with batching
#         adj = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)  # [batch_size, N, N]

#         # 4. Reshape node features and assignment matrix to match dense input
#         B = batch.max().item() + 1  # batch size
#         N = adj.size(-1)  # padded size (max num nodes in batch)
#         # Pad x and s to [B, N, ...]
#         # x_padded = torch.zeros((B, N, x.size(-1)), device=x.device)
#         # s_padded = torch.zeros((B, N, s.size(-1)), device=x.device)
#         x_padded = x.new_zeros((B, N, x.size(-1)))
#         s_padded = x.new_zeros((B, N, s.size(-1)))

#         for i in range(B):
#             node_indices = (batch == i).nonzero(as_tuple=True)[0]
#             n_nodes = node_indices.size(0)
#             x_padded[i, :n_nodes] = x[node_indices]
#             s_padded[i, :n_nodes] = s[node_indices]

#         # 5. Apply MinCut Pooling
#         x_pool, _, mc_loss, o_loss = dense_mincut_pool(x_padded, adj, s_padded)


#         ##Second pool

#         ######## Update adjacency using x_part (projected to match MP1 output) ##############
#         x_part_proj = self.part_proj1(x_part)
#         x_norm = F.normalize(x, p=2, dim=-1)
#         x_part_norm = F.normalize(x_part_proj, p=2, dim=-1)
#         cos_mat_dist = torch.matmul(x_norm, x_part_norm.t())
#         new_adj = cos_mat_dist @ cos_mat_dist.t()
#         edge_index_new = (new_adj > 0).nonzero(as_tuple=False).t()
#         edge_weight_new = new_adj[edge_index_new[0], edge_index_new[1]]

#         # === Second MP with updated adjacency ===
#         x2 = self.mp2(x, edge_index_new, edge_weight_new)
#         s2 = self.mlp(x2)

#         # Dense adj for second pool
#         adj2 = to_dense_adj(edge_index_new, edge_attr=edge_weight_new, batch=batch)
#         x2_padded = x2.new_zeros((B, N, x2.size(-1)))
#         s2_padded = x2.new_zeros((B, N, s2.size(-1)))
#         for i in range(B):
#             idx = (batch == i).nonzero(as_tuple=True)[0]
#             x2_padded[i, :idx.size(0)] = x2[idx]
#             s2_padded[i, :idx.size(0)] = s2[idx]

#         # Second pool
#         x_pool2, _, mc_loss2, o_loss2 = dense_mincut_pool(x2_padded, adj2, s2_padded)

#         # Combine losses
#         mc_loss = mc_loss1 + mc_loss2
#         o_loss = o_loss1 + o_loss2

#         # 6. Return softmax scores per node and losses
#         #return torch.softmax(s, dim=-1), mc_loss, o_loss
#         return torch.softmax(s2, dim=-1), mc_loss, o_loss


### Simple model##############

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import dense_mincut_pool

class DenseMinCutNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_clusters):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin_assign = nn.Linear(hidden_channels, out_clusters)

    def forward(self, x, edge_index, edge_weight, batch):
        # 1. Feature transformation
        x = F.relu(self.lin1(x))  # [N, hidden_channels]

        # 2. Dense batching
        x_dense, mask = to_dense_batch(x, batch)  # [B, N, hidden_channels]
        
        # 3. Assignment matrix
        s_dense = F.softmax(self.lin_assign(x_dense), dim=-1)  # [B, N, out_clusters]

        # 4. Adjacency matrix
        adj = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight)  # [B, N, N]

        # 5. MinCut Pooling
        x_pool, _, mc_loss, o_loss = dense_mincut_pool(x_dense, adj, s_dense, mask)  # Pooling done batch-wise

        # 6. Flatten soft assignments to match original behavior
        s_flat = s_dense[mask]  # shape: [total_num_nodes_across_batch, out_clusters]

        return s_flat, mc_loss, o_loss





############From protein_mincut_pool.py "https://github.com/pyg-team/pytorch_geometric/blob/master/examples/proteins_mincut_pool.py") ############

import os.path as osp
import time
from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGraphConv, GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch


class NetConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        num_nodes = ceil(0.5 * avg_num_nodes)
        self.pool1 = Linear(hidden_channels, num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = Linear(hidden_channels, num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s = self.pool1(x)
        x, adj, mc1, o1 = dense_mincut_pool(x, adj, s, mask)

        x = self.conv2(x, adj).relu()
        s = self.pool2(x)

        x, adj, mc2, o2 = dense_mincut_pool(x, adj, s)

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), mc1 + mc2, o1 + o2


