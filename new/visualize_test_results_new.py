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

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime  # Import datetime
from model_new import Net

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

data_path = os.environ.get("TEST_DATA_DIR", os.path.join(WORKSPACE_DIR, "new", "new_data", "testing_processed_ade_new"))
output_dir = os.environ.get("OUTPUT_PRED_DIR", os.path.join(WORKSPACE_DIR, "new", "new_predictions2"))
os.makedirs(output_dir, exist_ok=True)

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#MODEL_PATH = "/home/iiitb/Desktop/anant/GridRaster/saved_models/best_model.pth"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(WORKSPACE_DIR, "new", "new_models", "model_pruned_mp_[1024, 1024]_mlp_[512]_noAdjLearning_BS32_epoch_500_new/best_model.pth"))
MP_ACT = 'ELU'
MLP_ACT = 'ReLU'
IN_CHANNELS = 1024
NUM_CLUSTERS = 2
MP_UNITS = [1024, 1024] 
MLP_UNITS = [512]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Net(
    mp_units=MP_UNITS,
    mp_act=MP_ACT,
    in_channels=IN_CHANNELS,
    n_clusters=NUM_CLUSTERS,
    mlp_units=MLP_UNITS,
    mlp_act=MLP_ACT
).to(DEVICE)

#model = DenseMinCutNet(in_channels=1024, hidden_channels=64, out_clusters=2).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Loaded model for inference.")


for i in range(5923):
    data_raw_path = os.path.join(data_path, f"graph_{i}.pt")
    data = torch.load(data_raw_path)
    data = data.to(DEVICE)

    with torch.no_grad():
        out, _, _ = model(data.x, data.edge_index, data.edge_weight, batch=torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE))

    predicted = out.argmax(dim=1).cpu().numpy()


    segments = data.segments.cpu().numpy()                      # 2D label map of superpixels
    query_full_superpixels = data.query_full_superpixels.cpu().numpy()  # list of superpixel IDs used

    # id_map: superpixel ID → prediction label
    id_map = {sp_id: predicted[i] for i, sp_id in enumerate(query_full_superpixels)}

    # Reconstruct predicted mask
    pred_mask = np.zeros_like(segments, dtype=np.uint8)
    for sp_id, label in id_map.items():
        pred_mask[segments == sp_id] = label


    gt_query_part_superpixels = data.gt_query_part_superpixels.cpu().numpy()
    # Ground Truth Part Mask
    gt_mask = data.y.cpu().numpy()

    #print(f"query full superpixels number: {query_full_superpixels} and count {len(query_full_superpixels)} | query part superpixels number {gt_query_part_superpixels} and count {len(gt_query_part_superpixels)}")

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

    

    #################################################
    # Retrieve and prepare image
    img_s = data.support_image  # shape [H, W]
    img_s_np = img_s.cpu().numpy()

    # If single channel, convert to RGB
    if img_s_np.ndim == 2:
        img_gray_s = img_s_np
        img_rgb_s = np.stack([img_gray_s]*3, axis=-1)  # [H, W, 3]
    else:
        # If it's [C, H, W], permute to [H, W, C]
        img_rgb_s = img_s_np.transpose(1, 2, 0)
        # If normalized in [0,1], scale to [0,255]:
        if img_rgb_s.max() <= 1.0:
            img_rgb_s = (img_rgb_s * 255).astype(np.uint8)


    support_pixel_part_mask = data.support_y.cpu().numpy()

    # --- Create overlays ---
    gt_overlay = overlay_mask_rgb(img_rgb, gt_mask, alpha=0.4, color=(0, 1, 0))
    pred_overlay = overlay_mask_rgb(img_rgb, pred_mask, alpha=0.4, color=(1, 0, 0))

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 8))

    # Original image
    plt.subplot(2, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # GT mask
    plt.subplot(2, 4, 2)
    plt.imshow(gt_mask, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # GT overlay
    plt.subplot(2, 4, 3)
    plt.imshow(gt_overlay)
    plt.title("GT Overlay (green)")
    plt.axis("off")

    # Predicted overlay
    plt.subplot(2, 4, 4)
    plt.imshow(pred_overlay)
    plt.title("Predicted Overlay (red)")
    plt.axis("off")

    # Actual GT used for evaluation
    gt_query_part_superpixels = data.gt_query_part_superpixels.cpu().numpy()
    gt_mask2 = np.isin(segments, gt_query_part_superpixels).astype(np.uint8)
    overlay_gt = overlay_mask_rgb(img_rgb, gt_mask2, alpha=0.5, color=(0, 1, 0))

    plt.subplot(2, 4, 5)
    plt.imshow(overlay_gt)
    plt.title("GT Used for Evaluation")
    plt.axis("off")

    # Visual prompt (support part)
    segments_support = data.segments_support.cpu().numpy()
    support_part_superpixels = data.support_part_superpixels.cpu().numpy()
    support_mask = np.isin(segments_support, support_part_superpixels).astype(np.uint8)
    overlay_support = overlay_mask_rgb(img_rgb_s, support_mask, alpha=0.5, color=(0, 1, 0))

    support_pixel_overlay = overlay_mask_rgb(img_rgb_s, support_pixel_part_mask, alpha=0.5, color=(0, 1, 0))

    plt.subplot(2, 4, 6)
    plt.imshow(overlay_support)
    plt.title("Visual Prompt")
    plt.axis("off")

    # Support pixel mask
    plt.subplot(2, 4, 7)
    plt.imshow(support_pixel_overlay)
    plt.title("Support Pixel Mask")
    plt.axis("off")

    gt_full_mask = np.isin(segments, query_full_superpixels).astype(np.uint8)
    gt_full_overlay = overlay_mask_rgb(img_rgb, gt_full_mask, alpha=0.4, color=(0,1,0))
    #Query Full Mask
    plt.subplot(2, 4, 8)
    plt.imshow(gt_full_overlay)
    plt.title("Query full mask")
    plt.axis("off")

    plt.tight_layout()

    # --- Save figure with timestamp ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        output_dir, 
        f"ADE_objectID_{data.object_id}_partID_{data.part_id}_{timestamp}.png"
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)  # Close figure to free memory

    print(f"Saved visualization to {save_path}")



    ###################################################
    
#     # Create overlays
#     gt_overlay = overlay_mask_rgb(img_rgb, gt_mask, alpha=0.4, color=(0,1,0))
#     pred_overlay = overlay_mask_rgb(img_rgb, pred_mask, alpha=0.4, color=(1,0,0))

#     # Plot
#     plt.figure(figsize=(12, 6))

#     plt.subplot(4, 3, 1)
#     plt.imshow(img_rgb)
#     plt.title("Original Image")
#     plt.axis('off')

#     plt.subplot(4, 3, 2)
#     plt.imshow(gt_mask, cmap='gray')
#     plt.title("Ground Part Truth Mask")
#     plt.axis('off')

#     plt.subplot(4, 3, 3)
#     plt.imshow(gt_overlay)
#     plt.title("GT Part Overlay (green)")
#     plt.axis('off')

#     plt.subplot(4, 3, 6)
#     plt.imshow(pred_overlay)
#     plt.title("Predicted Part Overlay (red)")
#     plt.axis('off')

#     plt.subplot(4, 3, 5)
#     gt_full_mask = np.isin(segments, query_full_superpixels).astype(np.uint8)
#     gt_full_overlay = overlay_mask_rgb(img_rgb, gt_full_mask, alpha=0.4, color=(0,1,0))
#     plt.imshow(gt_full_overlay)
#     plt.title("Predicted Full Overlay (red)")
#     plt.axis('off')

#     plt.tight_layout()

#     # Create timestamp string
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Save the figure with timestamp
#     save_path = os.path.join(output_dir, f"ADE_objectID_{data.object_id}_partID_{data.part_id}_{timestamp}.png")
#     plt.savefig(save_path, bbox_inches="tight", dpi=300)
#     plt.close()



# # Create overlays
# gt_overlay = overlay_mask_rgb(img_rgb, gt_mask, alpha=0.4, color=(0,1,0))
# pred_overlay = overlay_mask_rgb(img_rgb, pred_mask, alpha=0.4, color=(1,0,0))

# # Plot
# plt.figure(figsize=(12, 6))

# plt.subplot(4, 3, 1)
# plt.imshow(img_rgb)
# plt.title("Original Image")
# plt.axis('off')

# plt.subplot(4, 3, 2)
# plt.imshow(gt_mask, cmap='gray')
# plt.title("Ground Truth Mask")
# plt.axis('off')

# plt.subplot(4, 3, 3)
# plt.imshow(gt_overlay)
# plt.title("GT Overlay (green)")
# plt.axis('off')

# plt.subplot(4, 3, 4)
# plt.imshow(pred_overlay)
# plt.title("Predicted Overlay (red)")
# plt.axis('off')


# # print actual GT
# gt_query_part_superpixels = data.gt_query_part_superpixels.cpu().numpy()
# gt_mask2 = np.isin(segments, gt_query_part_superpixels).astype(np.uint8)
# overlay_gt = overlay_mask_rgb(img_rgb, gt_mask2, alpha=0.5, color=(0, 1, 0))

# plt.subplot(2, 4, 5)
# plt.imshow(overlay_gt)
# plt.title("GT Used for Evaluation")
# plt.axis('off')



# # printing support part
# segments_support = data.segments_support.cpu().numpy() 
# support_part_superpixels = data.support_part_superpixels.cpu().numpy()
# support_mask = np.isin(segments_support, support_part_superpixels).astype(np.uint8)
# overlay_support = overlay_mask_rgb(img_rgb_s, support_mask, alpha=0.5, color=(0, 1, 0))

# plt.subplot(2, 4, 6)
# plt.imshow(overlay_support)
# plt.title("Visual Prompt")
# plt.axis('off')

# plt.subplot(2, 4, 7)
# plt.imshow(support_pixel_part_mask, cmap='gray')
# plt.title("Support Pixel Mask")
# plt.axis('off')


# plt.tight_layout()
# plt.show()

