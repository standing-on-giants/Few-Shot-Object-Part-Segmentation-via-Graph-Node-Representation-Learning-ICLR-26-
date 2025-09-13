#LOAD A BATCH OF IMAGES (SUPPORT AND QUERY), ALONG WITH THEIR MASKS (SUPPORT PART MASK, AND QUERY FULL MASK)
#USE get_query_feature_and_affinity_matrix() TO EXTRACT FEATURES FROM BOTH SUPPORT AND QUERY IMAGES, AND THEN CREATE AN AFFINITY MATRIX USING COSINE DISTANCE AS SIMILARITY METRIC

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.data import Data
#from torch_geometric.nn import gcn_norm
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from part_seg_shashank import get_query_feature_and_affinity_matrix_before_pruning, get_query_feature_and_affinity_matrix_after_pruning
import numpy as np
import os


from tqdm import tqdm

import pickle
with open('/home/iiitb/Desktop/anant/GridRaster/part_ours_training/pascal_val.pkl', 'rb') as f:
    supp_dict = pickle.load(f)

def summarize_dict(d, indent=0, max_depth=3):
    """Recursively summarize nested dicts/lists/tensors/arrays."""
    spacing = "  " * indent

    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{spacing}{k}:")
            if isinstance(v, (dict, list)) and indent < max_depth:
                summarize(v, indent + 1, max_depth)
            else:
                try:
                    shape = v.shape if hasattr(v, "shape") else None
                    dtype = v.dtype if hasattr(v, "dtype") else type(v)
                    print(f"{spacing}  shape={shape}, dtype={dtype}")
                except Exception as e:
                    print(f"{spacing}  could not summarize ({e})")

    elif isinstance(d, list):
        print(f"{spacing}list of length {len(d)}")
        if len(d) > 0 and indent < max_depth:
            summarize(d[0], indent + 1, max_depth)

    else:
        shape = d.shape if hasattr(d, "shape") else None
        dtype = d.dtype if hasattr(d, "dtype") else type(d)
        print(f"{spacing}value: shape={shape}, dtype={dtype}")



# print(f"supp_dict: {summarize_dict(supp_dict)}")
# print(a)

# for k in supp_dict.keys():
#     print(k, supp_dict[k]['image_name'])


from torch.utils.data import DataLoader
from dataset_shashank import PartQueryDataset, custom_transform

# root directory pointing to training_data
#dataset_root = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/training_data_MOHAN"
dataset_root = "/home/iiitb/Desktop/anant/GridRaster/part_ours_training/data/testing_data_pascal_MOHAN" # remeber to use correct support dict

# supp_dict is defined above

dataset = PartQueryDataset(root_dir=dataset_root, supp_dict=supp_dict, transform=custom_transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

from torch_geometric.data import Data, Batch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processed_data_dir = "test_processed_data_pascal"
os.makedirs(processed_data_dir, exist_ok=True)

###################################################################
import matplotlib.pyplot as plt

def to_numpy_img(t):
    """Convert tensor to numpy image safely."""
    if t.ndim == 3:  # (C, H, W) -> (H, W, C)
        return t.permute(1, 2, 0).cpu().numpy()
    elif t.ndim == 2:  # (H, W)
        return t.cpu().numpy()
    else:
        raise ValueError(f"Unexpected tensor shape {t.shape}")

def visualize_batch(batch, idx=0, save_dir="visualizations"):
    os.makedirs(save_dir, exist_ok=True)

    query_img = to_numpy_img(batch['query_image'][idx])
    query_full = to_numpy_img(batch['query_full_mask'][idx])
    query_part = to_numpy_img(batch['query_part_mask'][idx])

    supp_img = to_numpy_img(batch['support_image'][idx])
    supp_full = to_numpy_img(batch['support_full_mask'][idx])
    supp_part = to_numpy_img(batch['support_part_mask'][idx])

    # print("inside visualize batch")

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(query_img.astype("uint8"))
    axes[0, 0].set_title("Query Image")
    axes[0, 1].imshow(query_full)
    axes[0, 1].set_title("Query Full Mask")
    axes[0, 2].imshow(query_part)
    axes[0, 2].set_title("Query Part Mask")

    axes[1, 0].imshow(supp_img.astype("uint8"))
    axes[1, 0].set_title("Support Image")
    axes[1, 1].imshow(supp_full)
    axes[1, 1].set_title("Support Full Mask")
    axes[1, 2].imshow(supp_part)
    axes[1, 2].set_title("Support Part Mask")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
     # Save instead of show
    out_path = os.path.join(save_dir, f"batch_{idx}_before_processing.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved visualization to {out_path}")

import matplotlib.pyplot as plt
import os

def visualize_queryOrSupport(query_dict, save_path=None):
    """
    Visualize a query_dict with keys:
    ['original_image', 'superpixel_overlayed', 'superpixel_labels', 'superpixel_features']
    """
    original = query_dict["original_image"]
    overlay = query_dict["superpixel_overlayed"]
    labels = query_dict["superpixel_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    axes[1].imshow(overlay)
    axes[1].set_title("Superpixel Overlayed")


    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved query visualization to {save_path}")
    else:
        plt.show()




def visualizeEffectOfPruning(support_dict, support_part_blendedImage_before, query_full_blendedImage_before, query_part_blendedImage_before,
                            query_dict, support_part_blendedImage_after, query_full_blendedImage_after, query_part_blendedImage_after, 
                            save_path=None):
    
    #CALLED AS
    # visualizeEffectOfPruning(support_dict, support_part_blendedImage_before, query_full_blendedImage_before, query_part_blendedImage_before,
    #                              query_dict, support_part_blendedImage_after, query_full_blendedImage_after, query_part_blendedImage_after, 
    #                              save_path="./visualizations/pruningEffect")      



    fig, axes = plt.subplots(2, 4, figsize=(15, 5))

    axes[0][0].imshow(support_dict["original_image"])
    axes[0][0].set_title("Support image")

    axes[0][1].imshow(support_part_blendedImage_before)
    axes[0][1].set_title("support part before pruning")

    axes[0][2].imshow(query_full_blendedImage_before)
    axes[0][2].set_title("query full before pruning")

    axes[0][3].imshow(query_part_blendedImage_before)
    axes[0][3].set_title("query part before pruning")

    axes[1][0].imshow(query_dict["original_image"])
    axes[1][0].set_title("query image")

    axes[1][1].imshow(support_part_blendedImage_after)
    axes[1][1].set_title("support part after pruning")

    axes[1][2].imshow(query_full_blendedImage_after)
    axes[1][2].set_title("query full after pruning")

    axes[1][3].imshow(query_part_blendedImage_after)
    axes[1][3].set_title("query part after pruning")


    # for ax in axes:
        # ax.axis("off")
    for ax in axes.flat:
        ax.axis("off")


    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization of the effect of Pruning to {save_path}")
    else:
        plt.show()

#####################################################

from torchvision import transforms
from PIL import Image
import cv2

def resize_image(img):
    """Resize the input image using torchvision transforms
    """
    to_pil_image = transforms.ToPILImage()
    resize_transform = transforms.Resize(size=(224, 224))

    return resize_transform(to_pil_image(img))

def display_overlap_images(img_list, alpha=None, save=False, path=None):
    """Provided the image list and corresponding alphas, output the overlapped image
    """
    if alpha is None:
        alpha = [1]
        for i in range(1, len(img_list)):
            alpha.append(0.5)
    
    if save and path is None:
        path = 'overlap_image'
        
    if len(img_list) < 2:
        raise("At least 2 images are required for overlapping")
    

    blended_image = Image.blend(Image.fromarray(img_list[0]).convert("RGB"), Image.fromarray(img_list[1]).convert("RGB"), alpha[1])
    for i in range(2, len(img_list)):
        blended_image = Image.blend(blended_image, Image.fromarray(img_list[i]).convert("RGB"), alpha[i])
    
    
    
    #blended_image = Image.blend(blended_image, Image.fromarray(superpixel_intensity_rgb).convert("RGB"), alpha[i])


    blended_image = np.asarray(blended_image)
    #print(blended_image.shape)

    plt.axis('off')
    plt.tight_layout()
    plt.imshow(blended_image)
    
    if save:
        cv2.imwrite(path, cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))
    
    #cv2.imwrite("/home/iiitb/Desktop/anant/GridRaster/parts_ours/blended_image.png", cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR))

    #Added code by Anant
    return blended_image


#############################################



idx = 0
# for single batch_size
for batch in dataloader:
    batched_data_list = []

    # print(f"batch: {batch.keys()}")
    # visualize_batch(batch)
    print(f"object_id: {batch['object_id']}, part_id: {batch['part_id']}")
    # print("length of batch = ", len(batch))
    # print(batch)
    # print(a)

    # the below code is to get the item
    for i in range(len(batch['query_image'])):
        # print("length = ", len(batch['query_image']))
        # visualize_queryOrSupport(query_dict, save_path="./visualizations/query_vis_beforePruning.png")
        # visualize_queryOrSupport(support_dict, save_path="./visualizations/support_vis_beforePruning.png")


        query_dictBefore, support_dictBefore, query_full_superpixelsBefore, support_part_superpixelsBefore, gt_query_part_superpixelsBefore, cos_mat_distBefore = get_query_feature_and_affinity_matrix_before_pruning(batch["support_image"][i], batch["support_part_mask"][i], 
                                                                                                     batch["support_full_mask"][i], 
                                                                                                     batch["query_image"][i], batch["query_part_mask"][i], 
                                                                                                     batch["query_full_mask"][i])


        query_dictAfter, support_dictAfter, query_full_superpixelsAfter, support_part_superpixelsAfter, gt_query_part_superpixelsAfter, cos_mat_distAfter = get_query_feature_and_affinity_matrix_after_pruning(batch["support_image"][i], batch["support_part_mask"][i], 
                                                                                                     batch["support_full_mask"][i], 
                                                                                                     batch["query_image"][i], batch["query_part_mask"][i], 
                                                                                                     batch["query_full_mask"][i])




        support_part_blendedImage_before = display_overlap_images([support_dictBefore['original_image'], np.isin(support_dictBefore['superpixel_labels'], support_part_superpixelsBefore), support_dictBefore['superpixel_overlayed']], alpha=[0, 0.8, 0.2], save=True, path="./visualizations/overlapDisplay3.png")
        query_full_blendedImage_before = display_overlap_images([query_dictBefore['original_image'], np.isin(query_dictBefore['superpixel_labels'], query_full_superpixelsBefore), query_dictBefore['superpixel_overlayed']], alpha=[0, 0.8, 0.2], save=True, path="./visualizations/overlapDisplay3.png")
        query_part_blendedImage_before = display_overlap_images([query_dictBefore['original_image'], np.isin(query_dictBefore['superpixel_labels'], gt_query_part_superpixelsBefore), query_dictBefore['superpixel_overlayed']], alpha=[0, 0.8, 0.2], save=True, path="./visualizations/overlapDisplay3.png")


        support_part_blendedImage_after = display_overlap_images([support_dictAfter['original_image'], np.isin(support_dictAfter['superpixel_labels'], support_part_superpixelsAfter), support_dictAfter['superpixel_overlayed']], alpha=[0, 0.8, 0.2], save=True, path="./visualizations/overlapDisplay3.png")
        query_full_blendedImage_after = display_overlap_images([query_dictAfter['original_image'], np.isin(query_dictAfter['superpixel_labels'], query_full_superpixelsAfter), query_dictAfter['superpixel_overlayed']], alpha=[0, 0.8, 0.2], save=True, path="./visualizations/overlapDisplay3.png")
        query_part_blendedImage_after = display_overlap_images([query_dictAfter['original_image'], np.isin(query_dictAfter['superpixel_labels'], gt_query_part_superpixelsAfter), query_dictAfter['superpixel_overlayed']], alpha=[0, 0.8, 0.2], save=True, path="./visualizations/overlapDisplay3.png")

        visualizeEffectOfPruning(support_dictBefore, support_part_blendedImage_before, query_full_blendedImage_before, query_part_blendedImage_before,
                                 query_dictBefore, support_part_blendedImage_after, query_full_blendedImage_after, query_part_blendedImage_after, 
                                 save_path="./visualizations/pruningEffect")


        # visualizeEffectOfPruning(query_dict, , save_path="./visualizations/query_effectOfPruning")



        # visualize_queryOrSupport(query_dict, save_path="./visualizations/query_vis_afterPruning.png")
        # visualize_queryOrSupport(support_dict, save_path="./visualizations/support_vis_afterPruning.png")
        
        # print(f"query_dict: {query_dict.keys()}, support_dict: {support_dict.keys()}")
        # print(f"query_image_shape: {query_dict['original_image'].shape}, query_superpixels_shape: {query_dict['superpixel_overlayed'].shape}")
        # print(f"query_superpixel_labels_shape: {query_dict['superpixel_labels'].shape}, query_superpixel_features_shape: {query_dict['superpixel_features'].shape}")

        # print(f"support_image_shape: {support_dict['original_image'].shape}, support_superpixels_shape: {support_dict['superpixel_overlayed'].shape}")
        # print(f"support_superpixel_labels_shape: {support_dict['superpixel_labels'].shape}, support_superpixel_features_shape: {support_dict['superpixel_features'].shape}")
        # visualize_queryOrSupport(query_dict, save_path="./visualizations/query_vis.png")
        # visualize_queryOrSupport(support_dict, save_path="./visualizations/support_vis.png")
        
        # print(query_dict.keys(), support_dict.keys(), query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)
        # print(query_full_superpixels.shape, support_part_superpixels.shape, gt_query_part_superpixels.shape, cos_mat_dist.shape)

        # print(f"query_full_superpixels.shape = {query_full_superpixels.shape}")
        # print(f"query_full_superpixels = {query_full_superpixels}")
        # print(f"support_part_superpixels.shape = {support_part_superpixels.shape}")
        # print(f"support_part_superpixels = {support_part_superpixels}")
        # print(f"gt_query_part_superpixels.shape = {gt_query_part_superpixels.shape}")
        # print(f"gt_query_part_superpixels = {gt_query_part_superpixels}")

        # print(f"cos_mat_dist.shape = {cos_mat_dist.shape}")
        # print(f"cos_mat_dist = {cos_mat_dist}")

    

    # break       #FOR DEBUGGING