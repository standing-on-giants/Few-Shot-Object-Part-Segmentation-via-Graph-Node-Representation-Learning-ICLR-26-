# test.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader as GeoDataLoader
from model_shashank import Net, Net_second          # or DenseMinCutNet
from graph_dataset_shashank import GraphPartDataset
from sklearn.metrics import normalized_mutual_info_score as NMI

import matplotlib.pyplot as plt

import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluator.evaluator import SimplePartSegEvaluator

# ----------------------
# Configs (match train.py)
# ----------------------
TEST_DATA_DIR   = "shashank_data/testing_processed_ad"
#TEST_DATA_DIR   = "shashank_data/test_processed_data_pruned_shashank"
OUTPUT_DIR      = "shashank_models/OUTPUT_model_pruned_mp_[1024]_mlp_[512]_noAdjLearning_BS32_epoch_500_shashank"
MODEL_DIR       = "shashank_models/model_pruned_mp_[1024]_mlp_[512]_noAdjLearning_BS32_epoch_500_shashank"
#MODEL_DIR       = "shashank_models/model_1024_512_noAdjLearning"  #1st experiment
MODEL_FILE      = os.path.join(MODEL_DIR, "best_model.pth")
BATCH_SIZE      = 32
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MP_UNITS        = [64]
# MLP_UNITS       = []
MP_ACT          = 'ELU'
MLP_ACT         = 'ReLU'
IN_CHANNELS = 1024
NUM_CLUSTERS = 2
MP_UNITS = [1024] # a list
MLP_UNITS = [512]

#MP_UNITS = [1024, 1024] # a list
#MLP_UNITS = [512]
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
        #out, mc_loss, o_loss = model(batch.x, batch.x_part, batch.edge_index, batch.edge_weight, batch.batch, batch.batch_part)
        out, mc_loss, o_loss = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        preds = out.argmax(dim=1).cpu().numpy()

        #print(y_gt.shape, batch.num_graphs)

        y_gt  = batch.y.cpu().numpy()
        batch_size = batch.num_graphs
        height = 224
        width = 224

        # reshape to [batch_size, H, W]
        y_gt = y_gt.reshape(batch_size, height, width)


        # Compute NMI for this batch (node‑level)
        # nmi = NMI(y_gt, preds)
        # all_nmis.append(nmi)

        # Split into individual graphs
        data_list = batch.to_data_list()
        offset = 0
        pred_mask_list = []
        for i, data in enumerate(data_list):
            n = data.num_nodes
            node_preds = preds[offset:offset+n]
            offset += n

            # Reconstruct full-size masks
            segments = data.segments.cpu().numpy()
            full_sps = data.query_full_superpixels.cpu().numpy()
            gt_sps   = data.gt_query_part_superpixels.cpu().numpy()

            # part GT mask
            gt_mask = np.isin(segments, gt_sps).astype(np.uint8)
        
            # Full predicted mask
            pred_mask = np.zeros_like(segments, dtype=np.uint8)
            for new_id, orig_id in enumerate(full_sps):
                pred_mask[segments == orig_id] = node_preds[new_id]

            

            # evaluation code b/w gt_mask and pred_mask
            #print(data) # Data(x=[91, 1024], edge_index=[2, 8281], y=[91], edge_weight=[8281], object_id=[1], part_id=[1], segments=[224, 224], query_full_superpixels=[91], gt_query_part_superpixels=[33], image=[224, 224])

            # steps to run evaluator
            # Evaluation step
            #gt_mask = y_gt[i]

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
                'pred_mask': pred_mask,
                'gt_query_part_superpixels': data.gt_query_part_superpixels.cpu().numpy(),
                'query_full_superpixels': data.query_full_superpixels.cpu().numpy()
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
