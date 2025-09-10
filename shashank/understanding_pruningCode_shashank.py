#LOAD A BATCH OF IMAGES (SUPPORT AND QUERY), ALONG WITH THEIR MASKS (SUPPORT PART MASK, AND QUERY FULL MASK)
#USE get_query_feature_and_affinity_matrix() TO EXTRACT FEATURES FROM BOTH SUPPORT AND QUERY IMAGES, AND THEN CREATE AN AFFINITY MATRIX USING COSINE DISTANCE AS SIMILARITY METRIC

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.data import Data
#from torch_geometric.nn import gcn_norm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from part_seg_shashank import get_query_feature_and_affinity_matrix
import numpy as np
import os


from tqdm import tqdm

import pickle
with open('/home/iiitb/Desktop/anant/GridRaster/part_ours_training/pascal_val.pkl', 'rb') as f:
    supp_dict = pickle.load(f)

def summarize_dict(d, indent=0, max_depth=3):
    """Recursively summarize nested dicts/lists/tensors/arrays."""
    spacing = "  " * indent

    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{spacing}{k}:")
            if isinstance(v, (dict, list)) and indent < max_depth:
                summarize(v, indent + 1, max_depth)
            else:
                try:
                    shape = v.shape if hasattr(v, "shape") else None
                    dtype = v.dtype if hasattr(v, "dtype") else type(v)
                    print(f"{spacing}  shape={shape}, dtype={dtype}")
                except Exception as e:
                    print(f"{spacing}  could not summarize ({e})")

    elif isinstance(d, list):
        print(f"{spacing}list of length {len(d)}")
        if len(d) > 0 and indent < max_depth:
            summarize(d[0], indent + 1, max_depth)

    else:
        shape = d.shape if hasattr(d, "shape") else None
        dtype = d.dtype if hasattr(d, "dtype") else type(d)
        print(f"{spacing}value: shape={shape}, dtype={dtype}")



# print(f"supp_dict: {summarize_dict(supp_dict)}")
# print(a)

# for k in supp_dict.keys():
#     print(k, supp_dict[k]['image_name'])


from torch.utils.data import DataLoader
from dataset_shashank import PartQueryDataset, custom_transform

# root directory pointing to training_data
#dataset_root = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/training_data_MOHAN"
dataset_root = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_pascal_MOHAN" # remeber to use correct support dict

# supp_dict is defined above

dataset = PartQueryDataset(root_dir=dataset_root, supp_dict=supp_dict, transform=custom_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

from torch_geometric.data import Data, Batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processed_data_dir = "test_processed_data_pascal"
os.makedirs(processed_data_dir, exist_ok=True)

###################################################################
import matplotlib.pyplot as plt

def to_numpy_img(t):
    """Convert tensor to numpy image safely."""
    if t.ndim == 3:  # (C, H, W) -> (H, W, C)
        return t.permute(1, 2, 0).cpu().numpy()
    elif t.ndim == 2:  # (H, W)
        return t.cpu().numpy()
    else:
        raise ValueError(f"Unexpected tensor shape {t.shape}")

def visualize_batch(batch, idx=0, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    query_img = to_numpy_img(batch['query_image'][idx])
    query_full = to_numpy_img(batch['query_full_mask'][idx])
    query_part = to_numpy_img(batch['query_part_mask'][idx])

    supp_img = to_numpy_img(batch['support_image'][idx])
    supp_full = to_numpy_img(batch['support_full_mask'][idx])
    supp_part = to_numpy_img(batch['support_part_mask'][idx])

    # print("inside visualize batch")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(query_img.astype("uint8"))
    axes[0, 0].set_title("Query Image")
    axes[0, 1].imshow(query_full)
    axes[0, 1].set_title("Query Full Mask")
    axes[0, 2].imshow(query_part)
    axes[0, 2].set_title("Query Part Mask")

    axes[1, 0].imshow(supp_img.astype("uint8"))
    axes[1, 0].set_title("Support Image")
    axes[1, 1].imshow(supp_full)
    axes[1, 1].set_title("Support Full Mask")
    axes[1, 2].imshow(supp_part)
    axes[1, 2].set_title("Support Part Mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
     # Save instead of show
    out_path = os.path.join(save_dir, f"batch_{idx}_before_processing.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved visualization to {out_path}")

import matplotlib.pyplot as plt
import os

def visualize_queryOrSupport(query_dict, save_path=None):
    """
    Visualize a query_dict with keys:
    ['original_image', 'superpixel_overlayed', 'superpixel_labels', 'superpixel_features']
    """
    original = query_dict["original_image"]
    overlay = query_dict["superpixel_overlayed"]
    labels = query_dict["superpixel_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    axes[1].imshow(overlay)
    axes[1].set_title("Superpixel Overlayed")


    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved query visualization to {save_path}")
    else:
        plt.show()

#####################################################



idx = 0
# for single batch_size
for batch in dataloader:
    batched_data_list = []

    # print(f"batch: {batch.keys()}")
    # visualize_batch(batch)
    print(f"object_id: {batch['object_id']}, part_id: {batch['part_id']}")

    # the below code is to get the item
    for i in range(len(batch['query_image'])):
        query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist = get_query_feature_and_affinity_matrix(batch["support_image"][i], batch["support_part_mask"][i], 
                                                                                                     batch["support_full_mask"][i], 
                                                                                                     batch["query_image"][i], batch["query_part_mask"][i], 
                                                                                                     batch["query_full_mask"][i])
        
        print(f"query_dict: {query_dict.keys()}, support_dict: {support_dict.keys()}")
        print(f"query_image_shape: {query_dict['original_image'].shape}, query_superpixels_shape: {query_dict['superpixel_overlayed'].shape}")
        print(f"query_superpixel_labels_shape: {query_dict['superpixel_labels'].shape}, query_superpixel_features_shape: {query_dict['superpixel_features'].shape}")

        print(f"support_image_shape: {support_dict['original_image'].shape}, support_superpixels_shape: {support_dict['superpixel_overlayed'].shape}")
        print(f"support_superpixel_labels_shape: {support_dict['superpixel_labels'].shape}, support_superpixel_features_shape: {support_dict['superpixel_features'].shape}")
        # visualize_queryOrSupport(query_dict, save_path="./visualizations/query_vis.png")
        # visualize_queryOrSupport(support_dict, save_path="./visualizations/support_vis.png")
        # visualize_support(support_dict, save_path="support_vis.png")
        
        # print(query_dict.keys(), support_dict.keys(), query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)
        # print(query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)
    

    break       #FOR DEBUGGING