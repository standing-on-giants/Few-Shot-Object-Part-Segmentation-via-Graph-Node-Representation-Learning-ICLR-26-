# test.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader as GeoDataLoader
from model import Net              # or DenseMinCutNet
from graph_dataset import GraphPartDataset
from sklearn.metrics import normalized_mutual_info_score as NMI
from evaluator.evaluator import SimplePartSegEvaluator

# ----------------------
# Configs (match train.py)
# ----------------------
TEST_DATA_DIR   = "test_processed_data"
OUTPUT_DIR      = "test_outputs1"
MODEL_DIR       = "saved_models1"
MODEL_FILE      = os.path.join(MODEL_DIR, "best_model.pth")
BATCH_SIZE      = 8
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLUSTERS    = 2
# MP_UNITS        = [64]
# MLP_UNITS       = []
MP_ACT          = 'ELU'
MLP_ACT         = 'Identity'
IN_CHANNELS = 1024
NUM_CLUSTERS = [2, 2]
MP_UNITS = [[512,256, 128],[128,64]] # a list
MLP_UNITS = [[64,32],[32,16]]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# Dataset & Loader
# ----------------------
test_dataset = GraphPartDataset(TEST_DATA_DIR)
test_loader  = GeoDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------
# Model
# ----------------------
model = Net(
    mp_units=MP_UNITS,
    mp_act=MP_ACT,
    in_channels=IN_CHANNELS,
    n_clusters=NUM_CLUSTERS,
    mlp_units=MLP_UNITS,
    mlp_act=MLP_ACT
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()

# ----------------------
# Evaluator Setup
# ----------------------
evaluator = SimplePartSegEvaluator()

# ----------------------
# Inference Loop
# ----------------------
all_nmis = []
idx = 0
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(DEVICE)
        # Forward pass
        out, mc_loss, o_loss = model(batch.x, batch.x_part, batch.edge_index, batch.edge_weight, batch.batch, batch.batch_part)
        preds = out.argmax(dim=1).cpu().numpy()
        y_gt   = batch.y.cpu().numpy()

        # Compute NMI for this batch (node‑level)
        nmi = NMI(y_gt, preds)
        all_nmis.append(nmi)

        # Split into individual graphs
        data_list = batch.to_data_list()
        offset = 0
        for data in data_list:
            n = data.num_nodes
            node_preds = preds[offset:offset+n]
            offset += n

            # Reconstruct full-size masks
            segments = data.segments.cpu().numpy()
            full_sps = data.query_full_superpixels.cpu().numpy()
            gt_sps   = data.gt_query_part_superpixels.cpu().numpy()

            # Full GT mask
            gt_mask = np.isin(segments, gt_sps).astype(np.uint8)

            # Full predicted mask
            pred_mask = np.zeros_like(segments, dtype=np.uint8)
            for new_id, orig_id in enumerate(full_sps):
                pred_mask[segments == orig_id] = node_preds[new_id]

            # evaluation code b/w gt_mask and pred_mask
            #print(data) # Data(x=[91, 1024], edge_index=[2, 8281], y=[91], edge_weight=[8281], object_id=[1], part_id=[1], segments=[224, 224], query_full_superpixels=[91], gt_query_part_superpixels=[33], image=[224, 224])

            # steps to run evaluator
            # Evaluation step
            evaluator.process({
                'part_id': data.part_id,
                'gt_mask': torch.tensor(gt_mask),
                'pred_mask': torch.tensor(pred_mask)
            })

            # Original image tensor
            img_t = data.image.cpu().numpy()  # [H,W] or [C,H,W]

            # Save outputs
            out_path = os.path.join(OUTPUT_DIR, f"result_{idx}.pt")
            torch.save({
                'image': img_t,
                'segments': segments,
                'gt_mask': gt_mask,
                'pred_mask': pred_mask
            }, out_path)
            idx += 1

# ----------------------
# Summary
# ----------------------
print(f"\nAverage NMI over test set: {np.mean(all_nmis):.4f}")
print(f"Saved {idx} result files to '{OUTPUT_DIR}'")

# ----------------------
# Evaluate mIoU
# ----------------------
evaluator.evaluate()
