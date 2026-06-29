import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.data import Data
#from torch_geometric.nn import gcn_norm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from part_seg_new import get_query_feature_and_affinity_matrix_after_pruning
import numpy as np
import os


from tqdm import tqdm

import pickle
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SUPPORT_DICT_PATH = os.environ.get("SUPPORT_DICT_PATH", os.path.join(WORKSPACE_DIR, "pascal_val.pkl"))
with open(SUPPORT_DICT_PATH, 'rb') as f:
    supp_dict = pickle.load(f)


from torch.utils.data import DataLoader
from dataset_new import PartQueryDataset, custom_transform

# root directory pointing to training_data
dataset_root = os.environ.get("DATASET_ROOT", os.path.join(WORKSPACE_DIR, "data", "testing_data_pascal_MOHAN"))

# supp_dict is defined above

dataset = PartQueryDataset(root_dir=dataset_root, supp_dict=supp_dict, transform=custom_transform)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=8)
from torch_geometric.data import Data, Batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processed_data_dir = os.environ.get("PROCESSED_DATA_DIR", os.path.join(WORKSPACE_DIR, "new", "new_data", "testing_processed_pascal"))
os.makedirs(processed_data_dir, exist_ok=True)

idx = 0
# for single batch_size
for batch in dataloader:
    batched_data_list = []

    # the below code is to get the item
    for i in range(len(batch['query_image'])):
        query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist = get_query_feature_and_affinity_matrix_after_pruning(batch["support_image"][i], batch["support_part_mask"][i], 
                                                                                                     batch["support_full_mask"][i], 
                                                                                                     batch["query_image"][i], batch["query_part_mask"][i], 
                                                                                                     batch["query_full_mask"][i])

        
        
        # print(query_dict.keys(), support_dict.keys(), query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)
        # print(query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)

        # Get node features
        X = torch.tensor(query_dict['superpixel_features'][query_full_superpixels], dtype=torch.float32)
        X_part = torch.tensor(support_dict['superpixel_features'][support_part_superpixels], dtype=torch.float32)


        adj = cos_mat_dist @ cos_mat_dist.T
        adj = torch.tensor(adj, dtype=torch.float32)

        # Build graph edge_index and edge_weight
        edge_index = (adj > 0).nonzero(as_tuple=False).t()
        edge_weight = adj[edge_index[0], edge_index[1]]

        # Superpixel remapping
        id_map = {orig_id: new_id for new_id, orig_id in enumerate(query_full_superpixels)}
        segments = query_dict['superpixel_labels']

        # Create a mask of where superpixel labels are in query_full_superpixels
        mask = np.isin(query_dict['superpixel_labels'], query_full_superpixels)
        filtered_superpixels = np.where(mask, query_dict['superpixel_labels'], 0)
        segments = filtered_superpixels

        gt_query_part_seg = np.isin(segments, gt_query_part_superpixels)

        # Generate superpixel-level labels
        superpixel_gt_labels = np.full(len(id_map), -1)
        for orig_id, new_id in id_map.items():
            mask = segments == orig_id
            labels = gt_query_part_seg[mask]
            if len(labels) == 0:
                continue
            values, counts = np.unique(labels, return_counts=True)
            superpixel_gt_labels[new_id] = values[np.argmax(counts)]

        # Replace -1 with 0 (assumed background)
        superpixel_gt_labels[superpixel_gt_labels == -1] = 0

        # Create `y` tensor
        y = torch.tensor(superpixel_gt_labels, dtype=torch.long)

        #save the image (B*H*W) that is 8*224*224 but we are saving each in batch 
        img_t = batch['query_image'][i].cpu()
        img_s = batch['support_image'][i].cpu()


        # this below code is added to have batch infor in the pyG dataloader for the x_part also.
        # Number of nodes in x_part for this graph
        num_part_nodes = X_part.size(0)
        # For a single graph, batch_part is just all zeros
        batch_part = torch.zeros(num_part_nodes, dtype=torch.long)


        # Build PyG Data object (storing object_d and part_id which will be used in evaluation (optional for Training))
        #data = Data(x=X, x_part=X_part, batch_part=batch_part, edge_index=edge_index, edge_weight=edge_weight, y=y, object_id = batch['object_id'][i], part_id = batch['part_id'][i])

        data = Data(x=X, x_part=X_part, batch_part=batch_part, edge_index=edge_index, edge_weight=edge_weight, y=batch["query_part_mask"][i], support_y = batch["support_part_mask"][i], object_id = batch['object_id'][i], part_id = batch['part_id'][i])
        data.segments = torch.tensor(segments, dtype=torch.long)  # [H, W]
        data.query_full_superpixels = torch.tensor(query_full_superpixels, dtype=torch.long)
        data.gt_query_part_superpixels = torch.tensor(gt_query_part_superpixels, dtype=torch.long)
        data.support_part_superpixels = torch.tensor(support_part_superpixels, dtype=torch.long)
        data.image = img_t

        data.support_image = img_s

        segments_support = support_dict['superpixel_labels']

        mask_support_part = np.isin(support_dict['superpixel_labels'], support_part_superpixels) # a msked output true/false
        filtered_support_part_superpixels = np.where(mask_support_part, support_dict['superpixel_labels'], 0)
        segments_support = filtered_support_part_superpixels


        data.segments_support = torch.tensor(segments_support, dtype=torch.long)

        

        # Create binary mask (1 where label is in support_part_superpixels, else 0)
        segments_new = query_dict['superpixel_labels']
        binary_query_mask = np.isin(segments_new, gt_query_part_superpixels).astype(np.uint8)
        # Convert to tensor if needed
        plt.imsave("binary_query_mask.png", binary_query_mask, cmap="gray")

       

        # Save with matplotlib (preserves label values with a colormap)
        plt.imsave("support_pixel.png", batch["support_part_mask"][i], cmap='gray')
        plt.imsave("query_pixel.png", batch["query_part_mask"][i], cmap="gray")
        plt.imsave("superpixel_support.png", segments_support, cmap="gray")
        plt.imsave("superpixel_query.png", gt_query_part_seg, cmap="gray")

        #print(a)

        # Save data object
        torch.save(data, os.path.join(processed_data_dir, f"graph_{idx}.pt"))
        idx += 1

        #plt.imsave("saving_y.png", y.cpu().numpy(), cmap='gray')
       
        print("Batch completed (count started from 1):", idx)

