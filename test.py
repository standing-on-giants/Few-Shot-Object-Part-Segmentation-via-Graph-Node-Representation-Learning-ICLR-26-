import os
import torch
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader as GeoDataLoader
from model import Net_second  # same model as training
from graph_dataset import GraphPartDataset  # your dataset
from tqdm import tqdm
from evaluator.evaluator import SimplePartSegEvaluator

# ----------------------
# Configs (match training!)
# ----------------------
BATCH_SIZE = 32
NUM_CLUSTERS = 2
MP_UNITS = [[1024, 1024], [1024, 1024]]
MLP_UNITS = [[512, 512], [512, 512]]
MP_ACT = 'ELU'
MLP_ACT = 'ReLU'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "model_mp_[[1024, 1024], [1024, 1024]]_mlp_[[512, 512], [512, 512]]_AdjLearning_BS32_epoch_500"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# ----------------------
# Load Dataset
# ----------------------
test_dataset = GraphPartDataset("/home/iiitb/Desktop/anant/GridRaster/test_processed_data")
test_loader = GeoDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------
# Load Model
# ----------------------
model = Net_second(
    mp_units=MP_UNITS,
    mp_act=MP_ACT,
    in_channels=1024,
    n_clusters=NUM_CLUSTERS,
    mlp_units=MLP_UNITS,
    mlp_act=MLP_ACT
).to(DEVICE)

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

criterion = CrossEntropyLoss()
evaluator = SimplePartSegEvaluator()

# ----------------------
# Evaluation Loop
# ----------------------
total_loss = 0
correct = 0
total_nodes = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        batch = batch.to(DEVICE)

        # forward
        out, mc_loss, o_loss = model(batch.x, batch.x_part, batch.edge_index, batch.edge_weight, batch.batch, batch.batch_part)
        y = batch.y

        # losses
        sup_loss = criterion(out, y)
        unsup_loss = mc_loss.abs() + o_loss.abs()
        loss = sup_loss + unsup_loss

        total_loss += loss.item()

        # accuracy
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total_nodes += y.size(0)

        evaluator.process({
                'part_id': batch.part_id,
                'gt_mask': torch.tensor(batch.gt_mask),
                'pred_mask': torch.tensor(batch.pred_mask)
            })

avg_loss = total_loss / len(test_loader)
accuracy = correct / total_nodes

print(f"\nTest Results:")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")

evaluator.evaluate()
