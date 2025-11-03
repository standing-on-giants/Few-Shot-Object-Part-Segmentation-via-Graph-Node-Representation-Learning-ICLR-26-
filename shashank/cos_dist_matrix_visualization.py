import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch_geometric.data import Data
#from torch_geometric.nn import gcn_norm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from part_seg_shashank import get_query_feature_and_affinity_matrix_after_pruning, extract_cos_dist_mat_before_message_passing
import numpy as np
import os
from tqdm import tqdm

from model_shashank import Net, Net_second, Net_cos_mat
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt



import pickle
with open('/home/iiitb/Desktop/anant/GridRaster/part_ours_training/new_dict_val.pkl', 'rb') as f:
    supp_dict = pickle.load(f)


from torch.utils.data import DataLoader
from dataset_shashank import PartQueryDataset, custom_transform, PartQueryDataset_singleImage

# root directory pointing to training_data
#dataset_root = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_MOHAN"
dataset_root = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_MOHAN" # remember to use correct support dict

# supp_dict is defined above

# dataset = PartQueryDataset_singleImage(root_dir=dataset_root, supp_dict=supp_dict, transform=custom_transform)
###########################

dataset = PartQueryDataset_singleImage(
    query_image_path="/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_MOHAN/ADE_val_00000128_36_72/image.jpg",
    query_full_mask_path="/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_MOHAN/ADE_val_00000128_36_72/object_mask.png",
    query_part_mask_path="/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_MOHAN/ADE_val_00000128_36_72/part_mask.png",
    object_id=18,  # you now pass manually
    part_id=95,
    supp_dict=supp_dict,
    transform=custom_transform
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

from torch_geometric.data import Data, Batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

visualization_dir = "./cos_dist_mat_visualization"
os.makedirs(visualization_dir, exist_ok=True)

for data in dataloader:
    print(data["query_image"].shape)


#############################

idx = 0
# for single batch_size
for batch in dataloader:
    batched_data_list = []

    # the below code is to get the item
    for i in range(len(batch['query_image'])):
        query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist1 = get_query_feature_and_affinity_matrix_after_pruning(batch["support_image"][i], batch["support_part_mask"][i], 
                                                                                                     batch["support_full_mask"][i], 
                                                                                                     batch["query_image"][i], batch["query_part_mask"][i], 
                                                                                                     batch["query_full_mask"][i])

         # print(query_dict.keys(), support_dict.keys(), query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)
        # print(query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)

        # Get node features
        X = torch.tensor(query_dict['superpixel_features'][query_full_superpixels], dtype=torch.float32)
        X_part = torch.tensor(support_dict['superpixel_features'][support_part_superpixels], dtype=torch.float32)


        adj1 = cos_mat_dist1 @ cos_mat_dist1.T
        adj = adj1
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


        # this below code is added to have batch infor in the pyG dataloader for the x_part also.
        # Number of nodes in x_part for this graph
        num_part_nodes = X_part.size(0)
        # For a single graph, batch_part is just all zeros
        batch_part = torch.zeros(num_part_nodes, dtype=torch.long)


        # Build PyG Data object (storing object_d and part_id which will be used in evaluation (optional for Training))
        #data = Data(x=X, x_part=X_part, batch_part=batch_part, edge_index=edge_index, edge_weight=edge_weight, y=y, object_id = batch['object_id'][i], part_id = batch['part_id'][i])

        data = Data(x=X, x_part=X_part, batch_part=batch_part, edge_index=edge_index, edge_weight=edge_weight, y=batch["query_part_mask"][i], object_id = batch['object_id'][i], part_id = batch['part_id'][i])
        data.segments = torch.tensor(segments, dtype=torch.long)  # [H, W]
        data.query_full_superpixels = torch.tensor(query_full_superpixels, dtype=torch.long)
        data.gt_query_part_superpixels = torch.tensor(gt_query_part_superpixels, dtype=torch.long)
        data.image = img_t

        # Save data object
        torch.save(data, os.path.join(visualization_dir, f"graph_{idx}.pt"))
        idx += 1

        #plt.imsave("saving_y.png", y.cpu().numpy(), cmap='gray')
       
        # print("Batch completed (count started from 1):", idx)

        # Convert tensors to CPU numpy
        # query_img_np = batch["query_image"][i].permute(1, 2, 0).cpu().numpy()  # CxHxW -> HxWxC
        # support_img_np = batch["support_image"][i].permute(1, 2, 0).cpu().numpy()
        query_img_np = batch["query_image"][i].cpu().numpy()
        support_img_np = batch["support_image"][i].cpu().numpy()

        query_part_mask_np = batch["query_part_mask"][i].cpu().numpy()
        query_full_mask_np = batch["query_full_mask"][i].cpu().numpy()
        support_part_mask_np = batch["support_part_mask"][i].cpu().numpy()
        support_full_mask_np = batch["support_full_mask"][i].cpu().numpy()

        # Plot and save images + masks
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(query_img_np)
        axes[0, 0].set_title("Query Image")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(query_part_mask_np, cmap="gray")
        axes[0, 1].set_title("Query Part Mask")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(query_full_mask_np, cmap="gray")
        axes[0, 2].set_title("Query Full Mask")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(support_img_np)
        axes[1, 0].set_title("Support Image")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(support_part_mask_np, cmap="gray")
        axes[1, 1].set_title("Support Part Mask")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(support_full_mask_np, cmap="gray")
        axes[1, 2].set_title("Support Full Mask")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(visualization_dir, f"input_visualization_{idx}.png"))
        plt.close()

        break

    break


print(f"cos_mat_dist1.shape = {cos_mat_dist1.shape}")#, cos_mat_dist1 = {cos_mat_dist1}")


#TEST THIS ON A SINGLE IMAGE


# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_PATH = f"/home/iiitb/Desktop/anant/GridRaster/part_ours_training/shashank/cos_dist_mat_visualization/graph_{0}.pt"
MODEL_PATH = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/shashank/shashank_models/model_pruned_mp_[1024, 1024]_mlp_[512]_noAdjLearning_BS32_epoch_500_shashank/best_model.pth"

MP_ACT = 'ELU'
MLP_ACT = 'ReLU'
IN_CHANNELS = 1024
NUM_CLUSTERS = 2
MP_UNITS = [1024, 1024] # a list
MLP_UNITS = [512]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Net_cos_mat(
    mp_units=MP_UNITS,
    mp_act=MP_ACT,
    in_channels=IN_CHANNELS,
    n_clusters=NUM_CLUSTERS,
    mlp_units=MLP_UNITS,
    mlp_act=MLP_ACT
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

data = torch.load(GRAPH_PATH)
data = data.to(DEVICE)

# print(f"data = {data}")

with torch.no_grad():
    new_x_feature, out, _, _ = model(data.x, data.edge_index, data.edge_weight, batch=torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE))

# Move to CPU and convert to numpy
new_x_feature_cpu = new_x_feature.detach().cpu().numpy()
x_part_cpu = data.x_part.detach().cpu().numpy()

from sklearn.preprocessing import StandardScaler, normalize
ss = StandardScaler()
cos_mat_dist2 = normalize(ss.fit_transform(new_x_feature_cpu))@normalize(ss.fit_transform(x_part_cpu)).T
adj2 = cos_mat_dist2 @ cos_mat_dist2.T


print(f"cos_mat_dist2.shape = {cos_mat_dist2.shape}")#, cos_mat_dist2 = {cos_mat_dist2}")


# Plot cos_mat_dist1 and cos_mat_dist2
# fig, axes = plt.subplots(1, 3, figsize=(12, 5))
# im1 = axes[0].imshow(cos_mat_dist1, cmap="viridis")
# axes[0].set_title("cos_mat_dist1")
# plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# im2 = axes[1].imshow(cos_mat_dist2, cmap="viridis")
# axes[1].set_title("cos_mat_dist2")
# plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# axes[2].imshow(data["image"].cpu().numpy())
# axes[2].set_title("image in cos_dist_mat2")
# axes[2].axis("off")


# plt.tight_layout()
# plt.savefig(os.path.join(visualization_dir, f"cos_matrix_visualization_{idx}.png"))
# plt.close()



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def block_average(mat, block_h, block_w):
    H, W = mat.shape
    h_blocks = H // block_h
    w_blocks = W // block_w
    mat = mat[:h_blocks*block_h, :w_blocks*block_w]  # trim excess
    mat = mat.reshape(h_blocks, block_h, w_blocks, block_w)
    return mat.mean(axis=(1, 3))  # average over each block


# Generate random cosine similarity matrix
# cos_mat = np.random.uniform(-1, 1, (31, 11))
adj1_norm = (adj1 + 1) / 2  # normalize to [0, 1]
adj2_norm = (adj2 + 1) / 2

visualization_dir = "./cos_dist_matrix_visualizations"
os.makedirs(visualization_dir, exist_ok=True) 

# Target coarse grid size (width, height)
# target_size = (20, 11)

# Downsample matrix
# downsampled1 = cv2.resize(cos_dist_mat_norm, target_size, interpolation=cv2.INTER_AREA)
# downsampled_mat1 = block_average(cos_dist_mat1_norm, block_h=8, block_w=1)
# downsampled_mat2 = block_average(cos_dist_mat2_norm, block_h=8, block_w=1)

# # Plot
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# im1 = axes[0].imshow(downsampled_mat1, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
# fig.colorbar(im1, ax=axes[0], label="Block-averaged Cosine Similarity")
# axes[0].set_title("Coarse Grid Visualization")

# # Hide second subplot (or plot something else if needed)
# im2 = axes[1].imshow(downsampled_mat2, cmap='coolwarm', aspect='auto', vmin=0, vmax=1)
# fig.colorbar(im2, ax=axes[1], label="Block-averaged Cosine Similarity")
# axes[1].set_title("Coarse Grid Visualization")

# plt.tight_layout()
# output_path = os.path.join(visualization_dir, "cos_dist_mat_vis.png")
# plt.savefig(output_path)
# plt.close()

# print(f"Saved visualization to {output_path}")


# # Downsample with smaller block size (preserve variation)
# downsampled_mat1 = block_average(cos_dist_mat1_norm, block_h=2, block_w=1)
# downsampled_mat2 = block_average(cos_dist_mat2_norm, block_h=2, block_w=1)

# # Compute shared vmin and vmax
# global_vmin = min(downsampled_mat1.min(), downsampled_mat2.min())
# global_vmax = max(downsampled_mat1.max(), downsampled_mat2.max())

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# for ax, mat, title in zip(axes, [downsampled_mat1, downsampled_mat2],
#                           ["Coarse Grid (cos_mat_dist1)", "Coarse Grid (cos_mat_dist2)"]):
#     im = ax.imshow(mat, cmap='coolwarm', aspect='auto', vmin=global_vmin, vmax=global_vmax)

#     # Add gridlines
#     ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
#     ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
#     ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
#     ax.tick_params(which="both", bottom=False, left=False)
#     ax.set_title(title)

# # Add a single colorbar for both subplots (for consistency)
# fig.colorbar(im, ax=axes.ravel().tolist(), label="Block-averaged Cosine Similarity")

# plt.tight_layout()
# output_path = os.path.join(visualization_dir, "cos_dist_mat_vis.png")
# plt.savefig(output_path, dpi=300)
# plt.close()
# print(f"Saved visualization to {output_path}")

import numpy as np
import matplotlib.pyplot as plt

# --- Downsample (your existing step) ---
downsampled_mat1 = block_average(adj1_norm, block_h=3, block_w=3)
downsampled_mat2 = block_average(adj2_norm, block_h=3, block_w=3)

# --- Compute global vmin/vmax only for unmasked values ---
global_vmin = min(downsampled_mat1.min(), downsampled_mat2.min())
global_vmax = max(downsampled_mat1.max(), downsampled_mat2.max())

# --- Define Threshold ---
# threshold = (global_vmin + global_vmax) / 2  # adjust based on your data range
threshold = 0.501

# --- Mask low values ---
masked_mat1 = np.ma.masked_less(downsampled_mat1, threshold)
masked_mat2 = np.ma.masked_less(downsampled_mat2, threshold)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cmap = plt.cm.Reds  # single-color colormap
cmap.set_bad(color='white')  # white for masked cells

for ax, mat, title in zip(axes, [masked_mat1, masked_mat2],
                          ["Coarse Grid (cos_mat_dist1)", "Coarse Grid (cos_mat_dist2)"]):
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=global_vmin, vmax=global_vmax)

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False)
    ax.set_title(title)

# Shared colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), label="Cosine Similarity (> threshold)")

plt.tight_layout()
output_path = os.path.join(visualization_dir, "cos_dist_mat_highlighted.png")
plt.savefig(output_path, dpi=300)
plt.close()
print(f"Saved visualization to {output_path}")




