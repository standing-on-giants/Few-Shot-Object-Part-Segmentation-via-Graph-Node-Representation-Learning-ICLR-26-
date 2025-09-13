import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from model import Net, Net_second, DenseMinCutNet  # your model with forward(x, edge_index, edge_weight, batch)
from graph_dataset import GraphPartDataset  # assume you saved graphs and use torch.load()
from tqdm import tqdm
import torch.nn.functional as F

# ----------------------
# Configs
# ----------------------
EPOCHS = 500
#LEARNING_RATE = 1e-3
#LEARNING_RATE = 1e-2
LEARNING_RATE = 5e-4
CHECKPOINT_FREQ = 100
BATCH_SIZE = 32
NUM_CLUSTERS = 2 # used in one MP layer
#NUM_CLUSTERS = [2, 2]
#NUM_CLUSTERS = [2]
MP_UNITS = [[1024, 1024], [1024, 1024]] # used in one MP and pool first time it wa sjust [64]
#MP_UNITS = [[512,256, 128]] # a list
#MP_UNITS = [[512,256, 128],[128,64]] # a list( for 2 times)

MLP_UNITS = [[512], [512]] #so single MP, first time it was []
#MLP_UNITS = [[64,32]]
#MLP_UNITS = [[64,32],[32,16]] # for two times

MP_ACT = 'ELU'
#MLP_ACT = 'Identity'
MLP_ACT = 'ReLU'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model_mp_[[1024, 1024], [1024, 1024]]_mlp_[[512], [512]]_AdjLearning_BS32_epoch_500"
os.makedirs(MODEL_DIR, exist_ok=True)


# getting the graph training Dataset
dataset = GraphPartDataset("/home/iiitb/Desktop/anant/GridRaster/train_processed_data")  # load .pt files
loader = GeoDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------
# Model & Optimizer
# ----------------------
model = Net_second(
    mp_units=MP_UNITS,
    mp_act=MP_ACT,
    in_channels=1024,
    n_clusters=NUM_CLUSTERS,
    mlp_units=MLP_UNITS,
    mlp_act=MLP_ACT
).to(DEVICE)

# model = Net(
#     mp_units_list=MP_UNITS,
#     mp_act=MP_ACT,
#     in_channels=1024,
#     n_clusters_list=NUM_CLUSTERS,
#     mlp_units_list=MLP_UNITS,
#     mlp_act=MLP_ACT
# ).to(DEVICE)
#model = DenseMinCutNet(in_channels=1024, hidden_channels=64, out_clusters=2).to(DEVICE)

# for NetConv()
#model = Net(1024, 2).to(DEVICE)
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss()

#from torchviz import make_dot

# ----------------------
# Training Loop
# ----------------------
best_loss = float("inf")
best_model_weights = None
#alpha = 0.5  
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0
    count = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        
        #out, mc_loss, o_loss = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        # below code is not reqiired
        out, mc_loss, o_loss = model(batch.x, batch.x_part, batch.edge_index, batch.edge_weight, batch.batch, batch.batch_part)
        #make_dot(out, params=dict(model.named_parameters())).render("graph0", format="png")
        #print(a)
        
        # for NetConv
        #out, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)

        y = batch.y  # [num_nodes]

        sup_loss = criterion(out, y)
        unsup_loss = mc_loss.abs() + o_loss.abs()
        #unsup_loss = F.relu(mc_loss) + F.relu(o_loss)
        
        #unsup_loss = F.relu(mc_loss + o_loss)

        #loss = sup_loss + alpha * unsup_loss
        loss = sup_loss + unsup_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        count += 1

    avg_loss = total_loss / count
    avg_sup = total_sup_loss / count
    avg_unsup = total_unsup_loss / count

    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Sup: {avg_sup:.4f} | UnSup: {avg_unsup:.4f}")

    # Save checkpoint
    if epoch % CHECKPOINT_FREQ == 0:
        ckpt_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = os.path.join(MODEL_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_path)
        print(f"New best model found at epoch {epoch} with loss {best_loss:.4f}")

# ----------------------
# Final Save (this saves the final best model in last, it will keep in cache)
# ----------------------
# final_path = os.path.join(MODEL_DIR, "best_model.pth")
# torch.save(best_model_weights, final_path)
print(f"\nTraining completed. Best model saved to: {best_path}")
