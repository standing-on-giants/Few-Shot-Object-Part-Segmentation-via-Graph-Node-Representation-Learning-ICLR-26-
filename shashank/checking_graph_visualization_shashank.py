import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import numpy as np

# Overlay helper
def overlay_mask_rgb(img, mask, alpha=0.5, color=(1, 0, 0)):
    """
    img: HxWx3 uint8
    mask: HxW binary
    color: tuple in [0,1] for RGB
    """
    imgf = img.astype(float) / 255.0
    c = np.array(color)[None,None,:]
    mask_bool = mask.astype(bool)
    imgf[mask_bool] = imgf[mask_bool] * (1-alpha) + c * alpha
    return (imgf * 255).astype(np.uint8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_PATH = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/shashank/shashank_data/train_processed_data_pruned_shashank/graph_0.pt"

# Load one graph
data = torch.load(GRAPH_PATH)
data = data.to(DEVICE)

print(f"data: {data}")

segments = data.segments.cpu().numpy()                                  # 2D label map of superpixels
query_full_superpixels = data.query_full_superpixels.cpu().numpy()      # list of superpixel IDs used

# id_map: superpixel ID → prediction label
# id_map = {sp_id: predicted[i] for i, sp_id in enumerate(query_full_superpixels)}

gt_query_part_superpixels = data.gt_query_part_superpixels.cpu().numpy()
# Ground Truth Part Mask
gt_mask = np.isin(segments, gt_query_part_superpixels).astype(np.uint8)

# Retrieve and prepare image
img_t = data.image  # shape [H, W]
img_np = img_t.cpu().numpy()

# If single channel, convert to RGB
if img_np.ndim == 2:
    img_gray = img_np
    img_rgb = np.stack([img_gray]*3, axis=-1)  # [H, W, 3]
else:
    # If it's [C, H, W], permute to [H, W, C]
    img_rgb = img_np.transpose(1, 2, 0)
    # If normalized in [0,1], scale to [0,255]:
    if img_rgb.max() <= 1.0:
        img_rgb = (img_rgb * 255).astype(np.uint8)


# Create overlays
gt_overlay = overlay_mask_rgb(img_rgb, gt_mask, alpha=0.4, color=(0,1,0))

# # Plot
# plt.figure(figsize=(12, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(img_rgb)
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(2, 2, 2)
# plt.imshow(gt_mask, cmap='gray')
# plt.title("Ground Truth Mask")
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.imshow(gt_overlay)
# plt.title("GT Overlay (green)")
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(pred_overlay)
# plt.title("Predicted Overlay (red)")
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# gt_full_mask = np.isin(segments, query_full_superpixels).astype(np.uint8)
# gt_full_overlay = overlay_mask_rgb(img_rgb, gt_full_mask, alpha=0.4, color=(0,1,0))
# plt.imshow(gt_full_overlay)

import os

# Create a figure and plot
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(gt_mask, cmap='gray')
plt.title("Ground Truth Mask")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(gt_overlay)
plt.title("GT Overlay (green)")
plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(pred_overlay)
# plt.title("Predicted Overlay (red)")
# plt.axis('off')

plt.tight_layout()

# Save the figure instead of showing it
output_dir = "./visualizations"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "comparison.png")
plt.savefig(save_path, bbox_inches="tight", dpi=300)
plt.close()  # Close to free memory

print(f"Saved comparison figure to {save_path}")

# Save GT full overlay as a separate image
gt_full_mask = np.isin(segments, query_full_superpixels).astype(np.uint8)
gt_full_overlay = overlay_mask_rgb(img_rgb, gt_full_mask, alpha=0.4, color=(0,1,0))

plt.figure(figsize=(6, 6))
plt.imshow(gt_full_overlay)
plt.axis("off")
gt_save_path = os.path.join(output_dir, "gt_full_overlay.png")
plt.savefig(gt_save_path, bbox_inches="tight", dpi=300)
plt.close()

print(f"Saved GT full overlay to {gt_save_path}")
