import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from model_shashank import Net, Net_second, DenseMinCutNet  # your model with forward(x, edge_index, edge_weight, batch)
from graph_dataset_shashank import GraphPartDataset  # assume you saved graphs and use torch.load()
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np

# ----------------------
# Configs
# ----------------------
EPOCHS = 500
LEARNING_RATE = 1e-4
# LEARNING_RATE = 1e-2
# LEARNING_RATE = 5e-4
# LEARNING_RATE = 1e-4
# CHECKPOINT_FREQ = 100
BATCH_SIZE = 64
NUM_CLUSTERS = 2 # used in one MP layer
#NUM_CLUSTERS = [2, 2]
#NUM_CLUSTERS = [2]
MP_UNITS = [1024, 1024] # used in one MP and pool first time it wa sjust [64]
#MP_UNITS = [[512,256, 128]] # a list
#MP_UNITS = [[512,256, 128],[128,64]] # a list( for 2 times)

MLP_UNITS = [512] #so single MP, first time it was []
#MLP_UNITS = [[64,32]]
#MLP_UNITS = [[64,32],[32,16]] # for two times

MP_ACT = 'ELU'
#MLP_ACT = 'Identity'
MLP_ACT = 'ReLU'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "shashank_models/model_1024_1024_512_noAdjLearning"
os.makedirs(MODEL_DIR, exist_ok=True)


# getting the graph training Dataset
dataset = GraphPartDataset("/home/iiitb/Desktop/anant/GridRaster/part_ours_training/shashank/shashank_data/training_processed_ade")  # load .pt files
loader = GeoDataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------
# Model & Optimizer
# ----------------------

# net = Net(
#     mp_units=[1024],
#     mp_act='ELU',
#     in_channels=1024,
#     n_clusters=2,
#     mlp_units=[512],
#     mlp_act='ReLU'
# ).to(DEVICE)

#net.load_state_dict(torch.load("./shashank_models/model_pruned_mp_[1024]_mlp_[512]_noAdjLearning_BS32_epoch_500_shashank/best_model.pth"))

# checkpoint = torch.load("./shashank_models/model_pruned_mp_[1024]_mlp_[512]_AdjLearning_BS32_epoch_500_shashank/best_model.pth")
# print("Model keys:")
# for k in checkpoint.keys():
#     print(k)

# print("\n")


# print(f"net = {net}, net.mp = {net.mp}, net.mp[0] = {net.mp[0]}") #, net.mp[0][0] = {net.mp[0][0]}")
##first_mp_layer_weights = net.mp[0].lin_rel.weight
#first_mp_layer_bias = net.mp[0].lin_rel.bias

#first_mlp_layer_weights = net.mlp[0].weight
#first_mlp_layer_bias = net.mlp[0].bias

model = Net(
    mp_units=MP_UNITS,
    mp_act=MP_ACT,
    in_channels=1024,
    n_clusters=NUM_CLUSTERS,
    mlp_units=MLP_UNITS,
    mlp_act=MLP_ACT
).to(DEVICE)

# Copy weights only for first MP layer
#model.mp_layers[0][0].lin_rel.weight.data.copy_(first_mp_layer_weights.data)
#model.mp_layers[0][0].lin_rel.bias.data.copy_(first_mp_layer_bias.data)

# Copy weights for first MLP layer
#model.mlp_layers[0][0].weight.data.copy_(first_mlp_layer_weights.data)
#model.mlp_layers[0][0].bias.data.copy_(first_mlp_layer_bias.data)



# model = DenseMinCutNet(in_channels=1024, hidden_channels=64, out_clusters=2).to(DEVICE)

# for NetConv()
#model = Net(1024, 2).to(DEVICE)
#optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)


#optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = CrossEntropyLoss(ignore_index=-1)

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
        
        out, mc_loss, o_loss = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        # below code is not reqiired
        #out, mc_loss, o_loss = model(batch.x, batch.x_part, batch.edge_index, batch.edge_weight, batch.batch, batch.batch_part)
        #make_dot(out, params=dict(model.named_parameters())).render("graph0", format="png")
        #print(a)
        
        # for NetConv
        #out, mc_loss, o_loss = model(data.x, data.edge_index, data.batch)

        #print(batch)
        #preds = out.argmax(dim=1).cpu().numpy()
        preds = out

        #y = batch.y  
        y = batch.y.cpu()
        batch_size = batch.num_graphs
        height = 224
        width = 224
        # reshape to [batch_size, H, W]
        y = y.reshape(batch_size, height, width)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)
        y_binary = (y != 0).long()

        # print(preds.shape, y.shape)
        # import matplotlib.pyplot as plt
        # plt.imsave("imggg.png", y[0].cpu())
        # plt.imsave("imggg2.png", y_binary[0].cpu())

        # print(torch.unique(y_binary[0]))
        # print(torch.unique(y[0]))
        
        #print(preds.shape, y_binary.shape)
        #print(a)

        # Split into individual graphs
        data_list = batch.to_data_list()
    
        offset = 0
        pred_mask_list = []
        for data in data_list:
            n = data.num_nodes
            node_preds = preds[offset:offset+n] # (n, 2) eg: ..., torch.Size([200, 2]), ... (total = 32 = Batch Size)
            offset += n

            # Reconstruct full-size masks
            segments = data.segments.cpu().numpy() #224*224
            full_sps = data.query_full_superpixels.cpu().numpy() #n
            #gt_sps   = data.gt_query_part_superpixels.cpu().numpy()

            # Full GT mask
            #gt_mask = np.isin(segments, gt_sps).astype(np.uint8)

            # Full predicted mask

            H, W = segments.shape
            pred_mask = np.zeros((H, W, 2), dtype=np.float32)  # Initialize with zeros

            for new_id, orig_id in enumerate(full_sps):
                #print(node_preds[new_id].shape, new_id, orig_id)
                pred_mask[segments == orig_id] = node_preds[new_id].detach().cpu().numpy()

                # import matplotlib.pyplot as plt
                # plt.imsave("mask_y.png", y[0].cpu())
                # plt.imsave("mask_binary_y.png", y_binary[0].cpu())
                # #plt.imsave("pred_mask.png", pred_mask)
                # print(np.unique(pred_mask), pred_mask.shape)
                
            pred_mask_list.append(pred_mask)


        pred_mask_all = np.stack(pred_mask_list, axis=0)
        # Convert to tensor
        pred_mask_all = torch.tensor(pred_mask_all, dtype=torch.float, device=DEVICE) # Shape: [32, 224, 224, 2]
        pred_mask_all = pred_mask_all.permute(0, 3, 1, 2)  # [32, 2, 224, 224]


        #print(pred_mask_all.shape, y_binary.shape)

        #print(torch.unique(y_binary))

        #print(a)

        y_ignore = y_binary.clone()
        y_ignore[y_ignore == 0] = -1


        #sup_loss = criterion(pred_mask_all, y_ignore)
        unsup_loss = mc_loss.abs() + o_loss.abs()
        #loss = sup_loss + unsup_loss
        loss = unsup_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        #total_sup_loss += sup_loss.item()
        #total_unsup_loss += unsup_loss.item()
        count += 1

    avg_loss = total_loss / count
    #avg_sup = total_sup_loss / count
    #avg_unsup = total_unsup_loss / count

    #print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Sup: {avg_sup:.4f} | UnSup: {avg_unsup:.4f}")
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

    # Save checkpoint
    # if epoch % CHECKPOINT_FREQ == 0:
    #     ckpt_path = os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.pth")
    #     torch.save(model.state_dict(), ckpt_path)
    #     print(f"Saved checkpoint: {ckpt_path}")

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
