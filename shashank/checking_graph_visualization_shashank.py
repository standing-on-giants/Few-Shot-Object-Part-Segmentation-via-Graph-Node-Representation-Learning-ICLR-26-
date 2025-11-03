import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import numpy as np
from model_shashank import Net, Net_second

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

for i in range(5923):

    GRAPH_PATH = f"/home/iiitb/Desktop/anant/GridRaster/part_ours_training/shashank/shashank_data/testing_processed_ade_new/graph_{i}.pt"
    MODEL_PATH = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/shashank/shashank_models/model_pruned_mp_[1024, 1024]_mlp_[512]_noAdjLearning_BS32_epoch_500_shashank/best_model.pth"

    MP_ACT = 'ELU'
    MLP_ACT = 'ReLU'
    IN_CHANNELS = 1024
    NUM_CLUSTERS = 2
    MP_UNITS = [1024, 1024] # a list
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

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load one graph
    data = torch.load(GRAPH_PATH)
    data = data.to(DEVICE)

    # print(f"data: {data}")

    with torch.no_grad():
        out, _, _ = model(data.x, data.edge_index, data.edge_weight, batch=torch.zeros(data.num_nodes, dtype=torch.long, device=DEVICE))

    predicted = out.argmax(dim=1).cpu().numpy()

    segments = data.segments.cpu().numpy()                                  # 2D label map of superpixels
    query_full_superpixels = data.query_full_superpixels.cpu().numpy()      # list of superpixel IDs used

    # id_map: superpixel ID → prediction label
    id_map = {sp_id: predicted[i] for i, sp_id in enumerate(query_full_superpixels)}

    # Reconstruct predicted mask
    pred_mask = np.zeros_like(segments, dtype=np.uint8)
    for sp_id, label in id_map.items():
        pred_mask[segments == sp_id] = label

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
    pred_overlay = overlay_mask_rgb(img_rgb, pred_mask, alpha=0.4, color=(1,0,0))

    # Save GT full overlay
    gt_full_mask = np.isin(segments, query_full_superpixels).astype(np.uint8)
    gt_full_overlay = overlay_mask_rgb(img_rgb, gt_full_mask, alpha=0.4, color=(0,1,0))

    import os

    # Create a figure and plot
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(gt_overlay)
    plt.title("GT Overlay (green)")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(pred_overlay)
    plt.title("Predicted Overlay (red)")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(gt_full_overlay)
    plt.title("GT Full Overlay")
    plt.axis('off')

    plt.tight_layout()

    # Save the figure instead of showing it
    output_dir = "./shashank_predictions"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"ADE_objectID_{data.object_id}_partID_{data.part_id}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close() 
