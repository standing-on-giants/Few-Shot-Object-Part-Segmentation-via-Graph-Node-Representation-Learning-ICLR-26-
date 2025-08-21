import pickle
with open('part_ours_training/new_dict.pkl', 'rb') as f:
    supp_dict = pickle.load(f)


from torch.utils.data import DataLoader
from dataset import PartQueryDataset, custom_transform

# root directory pointing to training_data
dataset_root = "part_ours_training/data/training_data_MOHAN"

# supp_dict is defined above

dataset = PartQueryDataset(root_dir=dataset_root, supp_dict=supp_dict, transform=custom_transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=8)


import numpy as np
import os


from part_seg import get_query_feature_and_affinity_matrix
from model import Net

from tqdm import tqdm

import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.data import Data


# for single batch_size
for batch in dataloader:
       
    for i in range(len(batch['query_image'])):
        query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist = get_query_feature_and_affinity_matrix(batch["support_image"][i], batch["support_part_mask"][i], 
                                                                                                     batch["support_full_mask"][i], 
                                                                                                     batch["query_image"][i], batch["query_part_mask"][i], 
                                                                                                     batch["query_full_mask"][i])
        
        # Save the outputs
        #save_path = f"./saved_features/item_{i+11}.npz"
        mask = np.isin(query_dict.item()['superpixel_labels'], query_full_superpixels)
        filtered_superpixels = np.where(mask, query_dict.item()['superpixel_labels'], 0)

        # Prepare your input graph
        X = torch.tensor(query_dict.item()['superpixel_features'][query_full_superpixels], dtype=torch.float32)
        adj = cos_mat_dist @ cos_mat_dist.T
        adj = torch.tensor(adj, dtype=torch.float32)

        # Superpixel ID remapping
        id_map = {orig_id: new_id for new_id, orig_id in enumerate(query_full_superpixels)}

        # Convert to sparse format
        edge_index = (adj > 0).nonzero(as_tuple=False).t()
        edge_weight = adj[edge_index[0], edge_index[1]]

        # Ground truth part labels for evaluation
        #y = torch.tensor(gt_query_part_superpixels, dtype=torch.long)

        # Create PyG Data object
        #data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight, y=y)
        data = Data(x=X, edge_index=edge_index, edge_weight=edge_weight)

        # Normalize edge weights
        # from torch_geometric.nn import gcn_norm
        data.edge_index, data.edge_weight = gcn_norm(
            data.edge_index, data.edge_weight, data.num_nodes,
            add_self_loops=False, dtype=data.x.dtype
        )

        ITER = 5000
        NUM_CLUSTERS = 2    

        from model import Net   

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        model = Net(mp_units=[64], mp_act="ELU", in_channels=1024, n_clusters=2).to(device)
        # --- TRAINING ---
        # model = DenseMinCutNet(
        #     in_channels=X.size(1),
        #     hidden_channels=16,
        #     out_clusters=NUM_CLUSTERS,
        #     num_nodes=X.size(0)
        # )
        model = model.to(device)
        #print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        segments = filtered_superpixels
        gt_query_part_seg = np.isin(query_dict.item()['superpixel_labels'], gt_query_part_superpixels)

        # Convert part_seg (pixel-wise) --> node-level labels (superpixel-wise)
        # this code makes the value =1 for the unique labels so for all the labels from the full mask superpixel it will make 1 to those superpixel which is part
        superpixel_gt_labels = np.full(len(id_map), -1)  # Initialize
        #print(superpixel_gt_labels)
        for orig_id, new_id in id_map.items():
            mask_full = segments == orig_id
            labels_in_part_seg = gt_query_part_seg[mask_full]
            if len(labels_in_part_seg) == 0:
                continue
            values, counts = np.unique(labels_in_part_seg, return_counts=True)
            superpixel_gt_labels[new_id] = values[np.argmax(counts)]  # majority vote
        # print(superpixel_gt_labels, len(superpixel_gt_labels))

        # this below code is to get the pixel level part_mask from the binary label given to the superpixel belonging to parts
        labels_gt = np.full_like(segments, fill_value=-1)
        sp_to_label = dict(zip(query_full_superpixels, superpixel_gt_labels))
        for sp_id, label in sp_to_label.items():
            labels_gt[segments == sp_id] = label
        labels_gt[labels_gt == -1] = 0  # background

        # Training loop
        losses, sup_losses, unsup_losses = [], [], []
        model.train()
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(ITER)):
            optimizer.zero_grad()
            s, mc_loss, o_loss = model(data.x, data.edge_index, data.edge_weight)
            

            y = torch.from_numpy(superpixel_gt_labels).long().to(s.device)


            # s: [num_superpixels, 2]
            #c = s.argmax(dim=1).cpu().numpy()  # [num_superpixels]
            c = s.argmax(dim=1)  # stays as torch.Tensor

            
            #print(s.shape, logits.shape, c.shape, y.shape)
            
            sup_loss = criterion(s, y)

            unsup_loss = mc_loss + o_loss
            loss = unsup_loss + sup_loss
            loss = mc_loss + o_loss
            loss.backward()

            optimizer.step()

            #loss.item()
            losses.append(loss.item())
            sup_losses.append(sup_loss.item())
            unsup_losses.append(unsup_loss.item())
      
        
        break # just to print one element in the batch
        
    break #to just print first batch


      
         

os.makedirs('overlay_output', exist_ok=True)
#Save model
model_path = os.path.join('overlay_output', 'mincut_model_final.pth')
#torch.save(model.state_dict(), model_path)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': total_loss
}, model_path)

print(f"\nModel saved to {model_path}")
