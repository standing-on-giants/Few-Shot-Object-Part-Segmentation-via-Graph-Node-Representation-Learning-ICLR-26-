from transformers import AutoImageProcessor, Dinov2Model
import torch
import cv2
import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.cluster import SpectralBiclustering, SpectralClustering
from tqdm.notebook import tqdm
from PIL import Image
from scipy.special import softmax

from glob import glob
import os
import gc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import copy

import transformers, tokenizers
transformers.__version__, tokenizers.__version__

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



def display_images_in_grid(images, rows, cols, figsize=(20, 30), cmap='gray'):
   """Displays a grid of images

   Args:
       images: A list or array of images as NumPy arrays.
       rows: The number of rows in the grid.
       cols: The number of columns in the grid.
       figsize: The size of the figure in inches (width, height).
       cmap: The colormap to use for grayscale images.
   """

   num_images = len(images)
   if num_images > rows * cols:
       #for python>3.6
       #raise ValueError(f"Number of images ({num_images}) exceeds grid size ({rows}x{cols}).")

       #for python<3.6
       raise ValueError("Number of images ({}) exceeds grid size ({}x{}).".format(num_images, rows, cols))

   fig, axes = plt.subplots(rows, cols, figsize=figsize)
   axes = axes.flatten()  # Ensure axes are 1D for easier iteration

   for i, image in enumerate(images):
       ax = axes[i]
       ax.imshow(image, cmap=cmap)  # Use cmap for grayscale images
       ax.axis('off')  # Turn off axes for cleaner display

   plt.tight_layout(pad=0.4, h_pad=0)  # Adjust spacing between subplots
   plt.show()

def resize_image(img):
    """Resize the input image using torchvision transforms
    """
    to_pil_image = transforms.ToPILImage()
    resize_transform = transforms.Resize(size=(224, 224))

    return resize_transform(to_pil_image(img))


image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
model = Dinov2Model.from_pretrained("facebook/dinov2-large").to(device)

def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))

def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:img_id_to_img[final_result_list[0]['scene_id'], final_result_list[0]['image_id']]] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


from math import ceil

@torch.no_grad()
def dino_dct(img, processor, trunc_ratio=1):
    feats = model(**processor(images=img, return_tensors="pt").to(device))
    # print(list(feats.keys()))
    patch_feats = feats['last_hidden_state'][:, 1:, :].reshape(-1, 16, 16, 1024).cpu().detach().numpy()
    patch_feats = torch.tensor(np.transpose(patch_feats, (0, -1, -2, -3))).to(device)
    patch_feats_freq = dct_2d(patch_feats)
    
    patch_feats_freq_low_pass = torch.zeros_like(patch_feats_freq)
    patch_feats_freq_low_pass[:, :, :ceil(16*trunc_ratio), :ceil(16*trunc_ratio)] = patch_feats_freq[:, :, :ceil(16*trunc_ratio), :ceil(16*trunc_ratio)]
    # print(patch_feats_freq_low_pass)
    patch_feats = idct_2d(patch_feats_freq_low_pass).cpu().numpy()
    patch_feats = torch.tensor(np.transpose(patch_feats, (0, -1, -2, -3)).reshape(-1, 256, 1024)).to(device)
    # print(patch_feats)
    avg_feats = torch.mean(patch_feats, 1)[0]
    # patch_feats_fft = dct(feats['x_norm_patchtokens'].transpose(-2,-1), norm='ortho')[:, :, :ceil(256*trunc_ratio)]
    # patch_feats = idct(patch_feats_fft, norm='ortho').transpose(-2,-1)
    image_features = feats['last_hidden_state'][:, 0, :]
    # print(type(patch_feats))
    image_features /= image_features.norm(dim=-1, keepdim=True)
    # image_features = image_features.tolist()
    # final_img_features.extend(image_features)
    # final_img_filepaths.extend((list(file_paths)))
    # patch_img_features.extend(patch_feats.tolist())
    
    return image_features.squeeze(), patch_feats, avg_feats

def get_superpixel_features(model, image_processor, img_path=None, img=None, n_superpixels=256):
    """Get features for superpixels of the image
    """
    # to_pil_image = transforms.ToPILImage()
    # resize_transform = transforms.Resize(size=(224, 224))
    
    def overlapImageContour(img, contour): 
        width = contour.shape[0] 
        height = contour.shape[1] 

        for i in range(width): 
            for j in range(height): 
                if(contour[i][j] == 255): 
                    for k in range(img.shape[2]): 
                        img[i][j][k] = 255 

        return img 

    def SEEDS(img_rgb, prior = 2, histogram_bins = 5, doubleStep = False, numLevels = 2, numSuperpixels = 256, numIterations = 10, outputPath = './result.jpg'): 
        # Read image in color mode 
        height = img_rgb.shape[0]
        width = img_rgb.shape[1] 
        numChannels = None 

        if(len(img_rgb.shape) == 2):
            numChannels = 1 
        else: 
            numChannels = img_rgb.shape[2] 
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Create SEEDS class object 
        seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, numChannels, numSuperpixels, numLevels, prior, histogram_bins, doubleStep) 
        seeds.iterate(img, numIterations) 

        # Get the contour lavel mask 
        contourLabelMask = seeds.getLabelContourMask()
        labels = seeds.getLabels()

        # Now, overlap the label mask over the image
        contour_img = overlapImageContour(img, contourLabelMask) 
        

        # Save the image
        # cv2.imwrite(outputPath, img) 
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), labels, cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    
    if img_path is None and img is None:
        raise Exception("img_path and img both cannot be None")
    
    if img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return_dict = {}
    
    # resize image for Dino
    resized_image = resize_image(img)
    
    img_rgb = image_processor(resized_image, return_tensors='pt')
    mms_0_255 = MinMaxScaler((0, 255))
    # convert image to 0, 255 to get an image output
    input_img_h = mms_0_255.fit_transform(img_rgb.pixel_values[0].permute(1, 2, 0).reshape(-1, 3)).reshape(224, 224, 3).astype(np.uint8)
    return_dict['original_image'] = input_img_h.copy()
    
    images = SEEDS(input_img_h, numSuperpixels=n_superpixels, numIterations=50)
    superpixel_labels = images[1]
    # print("Unique Superpixels", np.unique(superpixel_labels).shape)
    # make sure that superpixel indices are serialized
    for i, superpixel in enumerate(np.unique(superpixel_labels)):
        superpixel_labels[superpixel_labels == superpixel] = i
        
    # return_dict['superpixel_labels'] = images[1]
    return_dict['superpixel_overlayed'] = images[2]
    # actual_n_superpixels = return_dict['superpixel_labels'].max() + 1
    
    
    h, w, c = input_img_h.shape
    rows, cols = 16, 16
    grid_h, grid_w = h//16, w//16
    grid_size = grid_h*grid_w

    n_superpixels = superpixel_labels.max() + 1

    # dictionary that contains info about the patch and percentage patch covered for each superpixel
    superpixel_weight_table = {}

    # calculate the per superpixel coverage of each patch
    for i in range(n_superpixels):
        superpixel_mask = (superpixel_labels == i).astype(np.uint8)
        superpixel_weights = {}
        
        for row in range(rows):
            for col in range(cols):
                coverage = superpixel_mask[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w].sum()/grid_size
                # coverage = patch.sum()/grid_size
                
                if coverage > 0:
                    superpixel_weights[row*cols + col] = coverage
        
        superpixel_weight_table[i] = superpixel_weights
            
    return_dict['superpixel_labels'] = superpixel_labels
    # print(np.unique(superpixel_labels).shape)
            
    # get patch features
    op = model(img_rgb.pixel_values.to(device))
    op = op.last_hidden_state[0, 1:]
    # op_reshaped = op.reshape(16, 16, -1)
    
    #dino dct
    # op = dino_dct(resized_image, image_processor)
    # op = op[1]
    
    superpixel_weights = []
    for superpixel in superpixel_weight_table.keys():
        coverage = superpixel_weight_table[superpixel]
        
        # weight of ith patch, weight_i = coverage_i/sum(coverage_i)
        # each superpixel will have a weight vector (256, ) which can be 
        # multiplied with the feature map (256, 1024) to get a feature vector
        # (1024, ) for each superpixel
        total_coverage = np.array(list(coverage.values())).sum()
        weights = np.zeros((256,), dtype=np.float32)
        for patch in coverage.keys():
            weights[patch] = coverage[patch]/total_coverage
            
        superpixel_weights.append(weights)
    
    # get superpixel weights in the vector space of dino features
    superpixel_weights = torch.tensor(np.array(superpixel_weights), dtype=torch.float32, device=device, requires_grad=False)
    
    return_dict['superpixel_features'] = (superpixel_weights @ op).detach().cpu().numpy()
    del op, superpixel_weights
    torch.cuda.empty_cache()
    gc.collect()
    return return_dict

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

def get_centroid_distances(correct_pose_output, gt_output):
    """Get a matrix with L2 distance between the centroids
    eg. matrix[i, j] will give the distance between ith real image superpixel
    and jth rendered image superpixel centroids
    """
    def get_sup_centroid(op_dict):
        """Get the centroids for each superpixel
        """
        centroids = []
        sups = np.unique(op_dict['superpixel_labels'])
        for sup in sups:
            # get pixels belonging to a particular superpixel selected
            sup_image = op_dict['superpixel_labels'] == sup
            sup_pixels = np.argwhere(sup_image)
            
            # average out the pixel coordinates to get the centroid
            centroid_opp = np.sum(sup_pixels, axis=0)//sup_pixels.shape[0]
            centroid = [centroid_opp[1], centroid_opp[0]]
            
            centroids.append(centroid)
            
        return np.array(centroids)
        
    correct_pose_centroids = get_sup_centroid(correct_pose_output)
    
    gt_centroids = get_sup_centroid(gt_output)
    
    # find the difference for x and y, use them to calculate L2
    x_dist = gt_centroids[:, 0].reshape(-1, 1) - correct_pose_centroids[:, 0]
    y_dist = gt_centroids[:, 1].reshape(-1, 1) - correct_pose_centroids[:, 1]
    
    total_dist = np.sqrt(np.square(x_dist) + np.square(y_dist))/(np.sqrt(2)*224)
    
    #print(f"Centroid distance matrix shape {total_dist.shape}")
    
    # num_real_superpixel x num_rend_superpixel matrix
    return total_dist

#latest implementation

import numpy as np
import cv2

def divide_boundary_superpixels_with_partitions(scale_dict1, part_mask, boundary_superpixels):
    """
    Divides boundary superpixels into two regions based on overlap with part_mask 
    and assigns new labels to the overlapping region. Reflects changes in superpixel_overlayed.

    Parameters:
        scale_dict (dict): A dictionary containing keys:
            - 'superpixel_labels': 2D array of superpixel labels.
            - 'superpixel_overlayed': 3D array to visualize superpixel boundaries.
        part_mask (numpy.ndarray): Binary mask (0 and 1) for regions of interest.
        boundary_superpixels (list): List of boundary superpixel labels.

    Returns:
        dict: Updated scale_dict with partitioned superpixel labels and overlay.
    """
    # Extract superpixel labels and overlay visualization
    superpixel_labels = scale_dict1['superpixel_labels']
    superpixel_overlayed = scale_dict1['superpixel_overlayed']
    height, width = superpixel_labels.shape

    # Initialize counter for new labels
    new_label_counter = 784

    # Loop over each boundary superpixel
    for i, superpixel_label in enumerate(boundary_superpixels, start=1):
        # Mask for the current superpixel
        superpixel_mask = superpixel_labels == superpixel_label

        # Overlapping and non-overlapping regions
        overlap_mask = superpixel_mask & (part_mask > 0)
        non_overlap_mask = superpixel_mask & ~(part_mask > 0)

        # Skip if no overlap
        if not np.any(overlap_mask):
            continue

        # Assign a new label to the overlapping region
        new_label = new_label_counter + i
        superpixel_labels[overlap_mask] = new_label

        # Update superpixel_overlayed to show the partition boundary
        combined_mask = np.zeros_like(superpixel_mask, dtype=np.uint8)
        combined_mask[overlap_mask] = 1
        combined_mask[non_overlap_mask] = 2

        # Find boundary between the partitions
        boundary = cv2.Canny((combined_mask * 127).astype(np.uint8), 50, 150)
        superpixel_overlayed[boundary > 0] = [255, 255, 255]  # White boundary lines

    # Update scale_dict with the modified labels and overlay
    scale_dict1['superpixel_labels'] = superpixel_labels
    scale_dict1['superpixel_overlayed'] = superpixel_overlayed

    return scale_dict1

import numpy as np

def inner_and_outer_superpixel_mask(superpixel_labels, mask):
    """
    Filters superpixels into inner and boundary superpixels based on overlap with a mask.

    Parameters:
        superpixel_labels (numpy.ndarray): A 2D array of superpixel labels.
        mask (numpy.ndarray): A binary mask where 1 indicates part regions and 0 otherwise.

    Returns:
        inner_superpixels (list): Superpixels completely covered by the mask.
        boundary_superpixels (list): Superpixels partially covered by the mask.
    """
    # Get unique superpixel labels
    unique_superpixels = np.unique(superpixel_labels)
    
    # Initialize lists to store inner and boundary superpixels
    inner_superpixels = []
    boundary_superpixels = []

    # Iterate over each unique superpixel label
    for superpixel_label in unique_superpixels:
        # Create a binary mask for the current superpixel
        superpixel_mask = superpixel_labels == superpixel_label
        
        # Total number of pixels in the current superpixel
        total_superpixel_pixels = np.sum(superpixel_mask)
        
        # Number of overlapping pixels with the given mask
        overlapping_pixels = np.sum(mask[superpixel_mask] > 0)

        # Classify the superpixel
        if overlapping_pixels == total_superpixel_pixels:
            # All pixels of the superpixel are covered by the mask
            inner_superpixels.append(superpixel_label)
        elif overlapping_pixels > 0:
            # Partial overlap: Boundary superpixel
            boundary_superpixels.append(superpixel_label)

    return inner_superpixels, boundary_superpixels

def create_dino_feature_for_added_labels(scale_dict1):
    return_dict = {}

    #copy image in img_rgb
    img = scale_dict1['original_image']

    # resize image for Dino
    resized_image = resize_image(img)
    img_rgb = image_processor(resized_image, return_tensors='pt')
    mms_0_255 = MinMaxScaler((0, 255))

    # convert image to 0, 255 to get an image output
    input_img_h = mms_0_255.fit_transform(img_rgb.pixel_values[0].permute(1, 2, 0).reshape(-1, 3)).reshape(224, 224, 3).astype(np.uint8)
    return_dict['original_image'] = input_img_h.copy()

    #below is the added code
    superpixel_labels = scale_dict1['superpixel_labels']

    for i, superpixel in enumerate(np.unique(superpixel_labels)):
        superpixel_labels[superpixel_labels == superpixel] = i
        
    # return_dict['superpixel_labels'] = images[1]
    return_dict['superpixel_overlayed'] = scale_dict1['superpixel_overlayed']


    h, w, c = input_img_h.shape
    rows, cols = 16, 16
    grid_h, grid_w = h//16, w//16
    grid_size = grid_h*grid_w

    n_superpixels = superpixel_labels.max() + 1

    # dictionary that contains info about the patch and percentage patch covered for each superpixel
    superpixel_weight_table = {}

    # calculate the per superpixel coverage of each patch
    for i in range(n_superpixels):
        superpixel_mask = (superpixel_labels == i).astype(np.uint8)
        superpixel_weights = {}
        
        for row in range(rows):
            for col in range(cols):
                coverage = superpixel_mask[row*grid_h:(row+1)*grid_h, col*grid_w:(col+1)*grid_w].sum()/grid_size
                # coverage = patch.sum()/grid_size
                
                if coverage > 0:
                    superpixel_weights[row*cols + col] = coverage
        
        superpixel_weight_table[i] = superpixel_weights
            
    return_dict['superpixel_labels'] = superpixel_labels
    # print(np.unique(superpixel_labels).shape)
            
    # get patch features
    op = model(img_rgb.pixel_values.to(device))
    op = op.last_hidden_state[0, 1:]
    # op_reshaped = op.reshape(16, 16, -1)
    
    #for dinoDCT
    # op = dino_dct(resized_image, image_processor)
    # op = op[1]

    superpixel_weights = []
    for superpixel in superpixel_weight_table.keys():
        coverage = superpixel_weight_table[superpixel]
        
        # weight of ith patch, weight_i = coverage_i/sum(coverage_i)
        # each superpixel will have a weight vector (256, ) which can be 
        # multiplied with the feature map (256, 1024) to get a feature vector
        # (1024, ) for each superpixel
        total_coverage = np.array(list(coverage.values())).sum()
        weights = np.zeros((256,), dtype=np.float32)
        for patch in coverage.keys():
            weights[patch] = coverage[patch]/total_coverage
            
        superpixel_weights.append(weights)

    # get superpixel weights in the vector space of dino features
    superpixel_weights = torch.tensor(np.array(superpixel_weights), dtype=torch.float32, device=device, requires_grad=False)

    return_dict['superpixel_features'] = (superpixel_weights @ op).detach().cpu().numpy()
    del op, superpixel_weights
    torch.cuda.empty_cache()
    gc.collect()

    return return_dict





def get_features_and_affinity_matrix_before_pruning(support_dict, query_dict, support_part_mask, query_full_mask, query_part_mask, airplane_name, out_path=None, **kwargs):
    if out_path is not None:
        #out_path_real = out_path + f"/{airplane_name}_real_pred_parts.jpg"
        out_path_real = out_path + "/{}_real_pred_parts.jpg".format(airplane_name)
    else:
        out_path_real = None
    
    # if multi_Scale is selected
    if 'multi_scale' in kwargs and kwargs['multi_scale']:
        pass

    else:
        if out_path is not None:
            #out_path_rend = out_path + f"/{airplane_name}_rendered_parts.jpg"
            out_path_rend = out_path + "/{}_rendered_parts.jpg".format(airplane_name)
        else:
            out_path_rend = None

        # #latest code implementation
        # inner_superpixel, bound_superpixel = inner_and_outer_superpixel_mask(
        #     correct_pose_output['superpixel_labels'],
        #     np.asarray(resize_image(part_mask_rend))
        # )
        
        # correct_pose_output = divide_boundary_superpixels_with_partitions(correct_pose_output, np.asarray(resize_image(part_mask_rend)), bound_superpixel)
        # correct_pose_output = create_dino_feature_for_added_labels(scale_dict1=correct_pose_output)
        
        # get the superpixels corresponding to the parts
        support_part_superpixels= np.unique(support_dict['superpixel_labels'][np.asarray(resize_image(support_part_mask)) > 0])

        #returning query_part mask superpixel which will be used as GT
        gt_query_part_superpixels = np.unique(query_dict['superpixel_labels'][np.asarray(resize_image(query_part_mask)) > 0])

        
        #below is added real_sups_seg code
        if 'use_seg_mask' in kwargs and kwargs['use_seg_mask']:
            
            # inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
            #         gt_output['superpixel_labels'],
            #         np.asarray(resize_image(seg_mask_real))
            #     )
            # gt_output = divide_boundary_superpixels_with_partitions(gt_output, np.asarray(resize_image(seg_mask_real)), bound_superpixel_gt)
            # gt_output = create_dino_feature_for_added_labels(scale_dict1=gt_output)
            query_full_superpixels = np.unique(query_dict['superpixel_labels'][np.asarray(resize_image(query_full_mask)) > 0])
            
        else:
            
            # inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
            #         gt_output['superpixel_labels'],
            #         np.asarray(resize_image(seg_mask_real))
            #     )
            # gt_output = divide_boundary_superpixels_with_partitions(gt_output, np.asarray(resize_image(seg_mask_real)), bound_superpixel_gt)
            # gt_output = create_dino_feature_for_added_labels(scale_dict1=gt_output)
            query_full_superpixels = np.unique(query_dict['superpixel_labels'])
            
        
        # get the cosine distance matrix
        ss = StandardScaler()
        cos_mat_dist = normalize(ss.fit_transform(query_dict['superpixel_features']))@normalize(ss.fit_transform(support_dict['superpixel_features'])).T


        # get the superpixel distances and subtract them from the cosine distance matrix
        if 'use_distance_info' in kwargs and kwargs['use_distance_info']:
            if 'distance_lambda' not in kwargs:
                raise("distance_lambda needs to be provided to use superpixel distance information")
            cos_mat_dist -= kwargs['distance_lambda']*get_centroid_distances(support_dict, query_dict)
        
        # select only the required parts
        cos_mat_dist = cos_mat_dist[:, support_part_superpixels]

        if 'use_seg_mask' in kwargs and kwargs['use_seg_mask']:
            cos_mat_dist = cos_mat_dist[query_full_superpixels]

        # apply column-wise softmax
        if 'softmax' in kwargs and kwargs['softmax']:
            # print(cluster_labels.shape)
            cos_mat_dist = softmax(cos_mat_dist, axis=0)
      
    # returning query_dict and the affinity matrix   
    return query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist


def get_query_feature_and_affinity_matrix_before_pruning(support_image, support_part_mask, support_full_mask, query_image, query_part_mask, query_full_mask):
    #ONE LINE DESCRIPTION: 
        #INPUT: GETS SUPPORT IMAGES + MASK, QUERY IMAGES + MASK
        #OUTPUT: SUPPORT SUPERPIXELS AND THEIR FEATURES, QUERY SUPERPIXELS AND THEIR FEATURES, AND THE AFFINITY MATRIX



    # MULTI SCALE CASE

    n_sups = 1024
    correct_pose_sups = 1024
    airplane_name = "GraphTraining"

    # Get the rendered plane and part mask
    #support_image, support_part_mask, support_full_mask = support_image, support_mask, support_full_mask

    # Get the real image and segmentation mask
    #query_image, query_full_mask = query_image, query_full_mask

    # Get the result dictionary for the real image
    query_dict1 = get_superpixel_features(model=model, image_processor=image_processor, img=query_image, n_superpixels=n_sups)

    # Get the result dictionary for the rendered image
    support_dict1 = get_superpixel_features(model=model, image_processor=image_processor, img=support_image, n_superpixels=correct_pose_sups)

    # print(f"query_dict: {query_dict.keys()}, support_dict: {support_dict.keys()}")
    # print(f"query_image_shape: {query_dict['original_image'].shape}, query_superpixels_shape: {query_dict['superpixel_overlayed'].shape}")
    # print(f"query_superpixel_labels_shape: {query_dict['superpixel_labels'].shape}, query_superpixel_features_shape: {query_dict['superpixel_features'].shape}")

    # print(f"support_image_shape: {support_dict['original_image'].shape}, support_superpixels_shape: {support_dict['superpixel_overlayed'].shape}")
    # print(f"support_superpixel_labels_shape: {support_dict['superpixel_labels'].shape}, support_superpixel_features_shape: {support_dict['superpixel_features'].shape}")

    # # from understanding_pruningCode_new import visualize_queryOrSupport
    # visualize_queryOrSupport(query_dict, save_path="./visualizations/queryBeforeFn_vis.png")
    # visualize_queryOrSupport(support_dict, save_path="./visualizations/supportBeforeFn_vis.png")

    # Output path setup
    #out_path = f'./outputs/Experiment_1'
    out_path = './outputs/Experiment_1'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # Get the predicted front_mask
    query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist = get_features_and_affinity_matrix_before_pruning(
        support_dict1, 
        query_dict1, 
        support_part_mask, 
        query_full_mask, 
        query_part_mask,
        airplane_name, 
        out_path=out_path, 
        use_seg_mask=True, 
        softmax=True, 
        use_distance_info=True, 
        distance_lambda=3, 
        multi_scale=False, 
        scale_list=correct_pose_sups
    )

    # different1 = np.any(query_dict['superpixel_overlayed'] != query_dict1['superpixel_overlayed'])
    # different2 = np.any(query_dict['original_image'] != query_dict1['original_image'])
    # different3 = np.any(query_dict['superpixel_labels'] != query_dict1['superpixel_labels'])
    # different4 = np.any(query_dict['superpixel_features'] != query_dict1['superpixel_features'])
    # different5 = np.any(support_dict['superpixel_overlayed'] != support_dict['superpixel_overlayed'])
    # different6 = np.any(support_dict['original_image'] != support_dict['original_image'])
    # different7 = np.any(support_dict['superpixel_labels'] != support_dict['superpixel_labels'])
    # different8 = np.any(support_dict['superpixel_features'] != support_dict['superpixel_features'])
    
    # print(f"{different1}, {different2}, {different3}, {different4}, {different5}, {different6}, {different7}, {different8}")

        
    return query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist













































def get_features_and_affinity_matrix_after_pruning(support_dict, query_dict, support_part_mask, query_full_mask, query_part_mask, airplane_name, out_path=None, **kwargs):
    #GIVEN SUPPORT AND QUERY IMAGES/MASKS, ALONG WITH SUPERPIXELS AND THEIR LABELS, THIS FUNCTION RETURNS AN AFFINITY MATRIX REPRESENTING SIMILARITY BTW SUPERPIXEL FEATURES



    if out_path is not None:
        #out_path_real = out_path + f"/{airplane_name}_real_pred_parts.jpg"
        out_path_real = out_path + "/{}_real_pred_parts.jpg".format(airplane_name)
    else:
        out_path_real = None
    
    # if multi_Scale is selected
    if 'multi_scale' in kwargs and kwargs['multi_scale']:
        pass

    else:
        if out_path is not None:
            #out_path_rend = out_path + f"/{airplane_name}_rendered_parts.jpg"
            out_path_rend = out_path + "/{}_rendered_parts.jpg".format(airplane_name)
        else:
            out_path_rend = None

        # #latest code implementation
        # inner_superpixel, bound_superpixel = inner_and_outer_superpixel_mask(
        #     correct_pose_output['superpixel_labels'],                             #REPLACE WITH SUPP_DICT
        #     np.asarray(resize_image(part_mask_rend))                              #REPLACE WITH SUPPORT_PART_MASK
        # )
        
        # correct_pose_output = divide_boundary_superpixels_with_partitions(correct_pose_output, np.asarray(resize_image(part_mask_rend)), bound_superpixel)
        # correct_pose_output = create_dino_feature_for_added_labels(scale_dict1=correct_pose_output)

        #latest code implementation
        inner_superpixel, bound_superpixel = inner_and_outer_superpixel_mask(
            support_dict['superpixel_labels'],                                      
            np.asarray(resize_image(support_part_mask))                              
        )

        # print(f"inner_superpixel.shape = {len(inner_superpixel)}")
        # print(f"inner_superpixel = {inner_superpixel}")

        # print(f"bound_superpixel.shape = {len(bound_superpixel)}")
        # print(f"bound_superpixel = {bound_superpixel}")
        
        support_dict = divide_boundary_superpixels_with_partitions(support_dict, np.asarray(resize_image(support_part_mask)), bound_superpixel)
        support_dict = create_dino_feature_for_added_labels(scale_dict1=support_dict)
        
        # get the superpixels corresponding to the parts
        support_part_superpixels= np.unique(support_dict['superpixel_labels'][np.asarray(resize_image(support_part_mask)) > 0])     #FIRST FINDS THE SUPERPIXEL LABELS THAT LIE IN THE PART MASK, AND THEN TAKES ONLY THE UNIQUE ONES AMONG THEM

        
        #below is added real_sups_seg code
        if 'use_seg_mask' in kwargs and kwargs['use_seg_mask']:
            
            # inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
            #         gt_output['superpixel_labels'],                           #REPLACE WITH QUERY_DICT
            #         np.asarray(resize_image(seg_mask_real))                   #REPLACE WITH QUERY_IMAGE
            #     )
            # gt_output = divide_boundary_superpixels_with_partitions(gt_output, np.asarray(resize_image(seg_mask_real)), bound_superpixel_gt)
            # gt_output = create_dino_feature_for_added_labels(scale_dict1=gt_output)

            inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
                    query_dict['superpixel_labels'],                           #REPLACE WITH QUERY_DICT
                    np.asarray(resize_image(query_full_mask))                   #REPLACE WITH QUERY_IMAGE
                )
            query_dict = divide_boundary_superpixels_with_partitions(query_dict, np.asarray(resize_image(query_full_mask)), bound_superpixel_gt)
            query_dict = create_dino_feature_for_added_labels(scale_dict1=query_dict)

            query_full_superpixels = np.unique(query_dict['superpixel_labels'][np.asarray(resize_image(query_full_mask)) > 0])          #DIFFERENCE BTW IF AND ELSE CONDITION (HERE, THE QUERY_FULL_SUPERPIXELS WILL ONLY BE FROM THE QUERY FULL MASK)
            
        else:
            
            # inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
            #         gt_output['superpixel_labels'],                                     #REPLACE WITH QUERY_DICT
            #         np.asarray(resize_image(query_full_mask))                             #REPLACE WITH QUERY_IMAGE
            #     )
            # gt_output = divide_boundary_superpixels_with_partitions(query_dict, np.asarray(resize_image(query_full_mask)), bound_superpixel_gt)
            # gt_output = create_dino_feature_for_added_labels(scale_dict1=query_dict)

            inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
                    query_dict['superpixel_labels'],                                     #REPLACE WITH QUERY_DICT
                    np.asarray(resize_image(query_full_mask))                             #REPLACE WITH QUERY_IMAGE
                )
            query_dict = divide_boundary_superpixels_with_partitions(query_dict, np.asarray(resize_image(query_full_mask)), bound_superpixel_gt)
            query_dict = create_dino_feature_for_added_labels(scale_dict1=query_dict)

            query_full_superpixels = np.unique(query_dict['superpixel_labels'])                                                        # DIFFERENCE FROM IF CONDITION (HERE, THE QUERY_FULL_SUPERPIXELS WILL INCLUDE THE WHOLE SUPERPIXELS)

        
        # get the cosine distance matrix
        ss = StandardScaler()
        cos_mat_dist = normalize(ss.fit_transform(query_dict['superpixel_features']))@normalize(ss.fit_transform(support_dict['superpixel_features'])).T       #normalize(query) @ normalize(support).T finds the dot product btw each query and support superpixel. ss.fit_transform performs the z-normalisation, while normalise() takes the unit vector along the dimension, or the vector with magnitude one along the superpixel's feature direction 


        # get the superpixel distances and subtract them from the cosine distance matrix
        if 'use_distance_info' in kwargs and kwargs['use_distance_info']:
            if 'distance_lambda' not in kwargs:
                raise("distance_lambda needs to be provided to use superpixel distance information")
            cos_mat_dist -= kwargs['distance_lambda']*get_centroid_distances(support_dict, query_dict)
        
        # select only the required parts (ONLY THE ONES FROM SUPPORT_MASK)
        # print(f"cos_mat_dist.shape initially: {cos_mat_dist.shape}")
        cos_mat_dist = cos_mat_dist[:, support_part_superpixels]
        # print(f"cos_mat_dist.shape after providing support part info: {cos_mat_dist.shape}")

        if 'use_seg_mask' in kwargs and kwargs['use_seg_mask']:
            # print(f"cos_mat_dist.shape initially: {cos_mat_dist.shape}")
            cos_mat_dist = cos_mat_dist[query_full_superpixels]             #CONSIDERING ONLY THE SUPERPIXELS BELONGING TO THE QUERY FULL MASK
            # print(f"cos_mat_dist.shape after providing query object mask info: {cos_mat_dist.shape}")

        # print("Shape:", cos_mat_dist.shape)
        # print("Min:", cos_mat_dist.min())
        # print("Max:", cos_mat_dist.max())
        # print("Mean:", cos_mat_dist.mean())
        # print("Std:", cos_mat_dist.std())
        # print("Any NaN?:", np.isnan(cos_mat_dist).any())
        # print("Any Inf?:", np.isinf(cos_mat_dist).any())

        # apply column-wise softmax
        if 'softmax' in kwargs and kwargs['softmax']:       #(WE CAN STILL HAVE NEGATIVE VALUES HERE, BUT THEY'LL ALL CONTRIBUTE TO A MAGNITUDE OF 1, BUT AFTER SOFTMAX, WE'LL ONLY HAVE POSITIVE VALUES)
            # print(cluster_labels.shape)
            # print(f"cos_mat_dist.shape initially: {cos_mat_dist.shape}")
            cos_mat_dist = softmax(cos_mat_dist, axis=0)                    #FOR A GIVEN SUPPORT SUPERPIXEL, FIND THE PROBABILITY DISTRIBUTION OVER THE QUERY SUPERPIXELS REGARDING WHICH ONE MIGHT CORRESPOND TO THIS SUPPORT SUPERPIXEL
            # print(f"cos_mat_dist.shape after taking softmax: {cos_mat_dist.shape}")
        

        #returning query_part mask superpixel which will be used as GT
        query_dict_copy = query_dict.copy()
        inner_superpixel_part_gt, bound_superpixel_part_gt = inner_and_outer_superpixel_mask(
                    query_dict_copy['superpixel_labels'],                                     
                    np.asarray(resize_image(query_part_mask))                             
                )
        query_dict_part = divide_boundary_superpixels_with_partitions(query_dict_copy, np.asarray(resize_image(query_part_mask)), bound_superpixel_gt)
        query_dict_part = create_dino_feature_for_added_labels(scale_dict1=query_dict_part)
        gt_query_part_superpixels = np.unique(query_dict_part['superpixel_labels'][np.asarray(resize_image(query_part_mask)) > 0])    

      
    # returning query_dict and the affinity matrix


    return query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist

import numpy as np


def get_query_feature_and_affinity_matrix_after_pruning(support_image, support_part_mask, support_full_mask, query_image, query_part_mask, query_full_mask):
    #ONE LINE DESCRIPTION: 
        #INPUT: GETS SUPPORT IMAGES + MASK, QUERY IMAGES + MASK
        #OUTPUT: SUPPORT SUPERPIXELS AND THEIR FEATURES, QUERY SUPERPIXELS AND THEIR FEATURES, AND THE AFFINITY MATRIX



    # MULTI SCALE CASE

    n_sups = 1024
    correct_pose_sups = 1024
    airplane_name = "GraphTraining"

    # Get the rendered plane and part mask
    #support_image, support_part_mask, support_full_mask = support_image, support_mask, support_full_mask

    # Get the real image and segmentation mask
    #query_image, query_full_mask = query_image, query_full_mask

    # Get the result dictionary for the real image
    query_dict1 = get_superpixel_features(model=model, image_processor=image_processor, img=query_image, n_superpixels=n_sups)

    # Get the result dictionary for the rendered image
    support_dict1 = get_superpixel_features(model=model, image_processor=image_processor, img=support_image, n_superpixels=correct_pose_sups)

    # print(f"query_dict: {query_dict.keys()}, support_dict: {support_dict.keys()}")
    # print(f"query_image_shape: {query_dict['original_image'].shape}, query_superpixels_shape: {query_dict['superpixel_overlayed'].shape}")
    # print(f"query_superpixel_labels_shape: {query_dict['superpixel_labels'].shape}, query_superpixel_features_shape: {query_dict['superpixel_features'].shape}")

    # print(f"support_image_shape: {support_dict['original_image'].shape}, support_superpixels_shape: {support_dict['superpixel_overlayed'].shape}")
    # print(f"support_superpixel_labels_shape: {support_dict['superpixel_labels'].shape}, support_superpixel_features_shape: {support_dict['superpixel_features'].shape}")

    # # from understanding_pruningCode_new import visualize_queryOrSupport
    # visualize_queryOrSupport(query_dict, save_path="./visualizations/queryBeforeFn_vis.png")
    # visualize_queryOrSupport(support_dict, save_path="./visualizations/supportBeforeFn_vis.png")

    # Output path setup
    #out_path = f'./outputs/Experiment_1'
    out_path = './outputs/Experiment_1'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # print(f"query_dict shapes before pruning: {query_dict1['original_image'].shape}, {query_dict1['superpixel_overlayed'].shape}, {query_dict1['superpixel_labels'].shape}, {query_dict1['superpixel_values'].shape}")

    # print(f"support_dict1['superpixel_labels].shape: {support_dict1['superpixel_labels'].shape}")

    # Get the predicted front_mask
    query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist = get_features_and_affinity_matrix_after_pruning(
        support_dict1, 
        query_dict1, 
        support_part_mask, 
        query_full_mask, 
        query_part_mask,
        airplane_name, 
        out_path=out_path, 
        use_seg_mask=True, 
        softmax=True, 
        use_distance_info=True, 
        distance_lambda=3, 
        multi_scale=False, 
        scale_list=correct_pose_sups
    )


    # print(f"support_part_superpixels: {support_part_superpixels}")
    

    # print(f"query_dict shapes before pruning: {query_dict['original_image'].shape}, {query_dict['superpixel_overlayed'].shape}, {query_dict['superpixel_labels'].shape}, {query_dict['superpixel_values'].shape}")
    # print(a)
    # different1 = np.any(query_dict['superpixel_overlayed'] != query_dict1['superpixel_overlayed'])
    # different2 = np.any(query_dict['original_image'] != query_dict1['original_image'])
    # different3 = np.any(query_dict['superpixel_labels'] != query_dict1['superpixel_labels'])
    # different4 = np.any(query_dict['superpixel_features'] != query_dict1['superpixel_features'])
    # different5 = np.any(support_dict['superpixel_overlayed'] != support_dict['superpixel_overlayed'])
    # different6 = np.any(support_dict['original_image'] != support_dict['original_image'])
    # different7 = np.any(support_dict['superpixel_labels'] != support_dict['superpixel_labels'])
    # different8 = np.any(support_dict['superpixel_features'] != support_dict['superpixel_features'])
    
    # print(f"{different1}, {different2}, {different3}, {different4}, {different5}, {different6}, {different7}, {different8}")

        
    return query_dict, support_dict, query_full_superpixels, support_part_superpixels, gt_query_part_superpixels, cos_mat_dist




































def get_cos_dist_mat_helper(support_dict, query_dict, support_part_mask, query_full_mask, query_part_mask, airplane_name, out_path=None, **kwargs):
    #GIVEN SUPPORT AND QUERY IMAGES/MASKS, ALONG WITH SUPERPIXELS AND THEIR LABELS, THIS FUNCTION RETURNS AN AFFINITY MATRIX REPRESENTING SIMILARITY BTW SUPERPIXEL FEATURES



    if out_path is not None:
        #out_path_real = out_path + f"/{airplane_name}_real_pred_parts.jpg"
        out_path_real = out_path + "/{}_real_pred_parts.jpg".format(airplane_name)
    else:
        out_path_real = None
    
    # if multi_Scale is selected
    if 'multi_scale' in kwargs and kwargs['multi_scale']:
        pass

    else:
        if out_path is not None:
            #out_path_rend = out_path + f"/{airplane_name}_rendered_parts.jpg"
            out_path_rend = out_path + "/{}_rendered_parts.jpg".format(airplane_name)
        else:
            out_path_rend = None

        # #latest code implementation
        # inner_superpixel, bound_superpixel = inner_and_outer_superpixel_mask(
        #     correct_pose_output['superpixel_labels'],                             #REPLACE WITH SUPP_DICT
        #     np.asarray(resize_image(part_mask_rend))                              #REPLACE WITH SUPPORT_PART_MASK
        # )
        
        # correct_pose_output = divide_boundary_superpixels_with_partitions(correct_pose_output, np.asarray(resize_image(part_mask_rend)), bound_superpixel)
        # correct_pose_output = create_dino_feature_for_added_labels(scale_dict1=correct_pose_output)

        #latest code implementation
        inner_superpixel, bound_superpixel = inner_and_outer_superpixel_mask(
            support_dict['superpixel_labels'],                                      
            np.asarray(resize_image(support_part_mask))                              
        )

        # print(f"inner_superpixel.shape = {len(inner_superpixel)}")
        # print(f"inner_superpixel = {inner_superpixel}")

        # print(f"bound_superpixel.shape = {len(bound_superpixel)}")
        # print(f"bound_superpixel = {bound_superpixel}")
        
        support_dict = divide_boundary_superpixels_with_partitions(support_dict, np.asarray(resize_image(support_part_mask)), bound_superpixel)
        support_dict = create_dino_feature_for_added_labels(scale_dict1=support_dict)
        
        # get the superpixels corresponding to the parts
        support_part_superpixels= np.unique(support_dict['superpixel_labels'][np.asarray(resize_image(support_part_mask)) > 0])     #FIRST FINDS THE SUPERPIXEL LABELS THAT LIE IN THE PART MASK, AND THEN TAKES ONLY THE UNIQUE ONES AMONG THEM

        
        #below is added real_sups_seg code
        if 'use_seg_mask' in kwargs and kwargs['use_seg_mask']:
            
            # inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
            #         gt_output['superpixel_labels'],                           #REPLACE WITH QUERY_DICT
            #         np.asarray(resize_image(seg_mask_real))                   #REPLACE WITH QUERY_IMAGE
            #     )
            # gt_output = divide_boundary_superpixels_with_partitions(gt_output, np.asarray(resize_image(seg_mask_real)), bound_superpixel_gt)
            # gt_output = create_dino_feature_for_added_labels(scale_dict1=gt_output)

            inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
                    query_dict['superpixel_labels'],                           #REPLACE WITH QUERY_DICT
                    np.asarray(resize_image(query_full_mask))                   #REPLACE WITH QUERY_IMAGE
                )
            query_dict = divide_boundary_superpixels_with_partitions(query_dict, np.asarray(resize_image(query_full_mask)), bound_superpixel_gt)
            query_dict = create_dino_feature_for_added_labels(scale_dict1=query_dict)

            query_full_superpixels = np.unique(query_dict['superpixel_labels'][np.asarray(resize_image(query_full_mask)) > 0])          #DIFFERENCE BTW IF AND ELSE CONDITION (HERE, THE QUERY_FULL_SUPERPIXELS WILL ONLY BE FROM THE QUERY FULL MASK)
            
        else:
            
            # inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
            #         gt_output['superpixel_labels'],                                     #REPLACE WITH QUERY_DICT
            #         np.asarray(resize_image(query_full_mask))                             #REPLACE WITH QUERY_IMAGE
            #     )
            # gt_output = divide_boundary_superpixels_with_partitions(query_dict, np.asarray(resize_image(query_full_mask)), bound_superpixel_gt)
            # gt_output = create_dino_feature_for_added_labels(scale_dict1=query_dict)

            inner_superpixel_gt, bound_superpixel_gt = inner_and_outer_superpixel_mask(
                    query_dict['superpixel_labels'],                                     #REPLACE WITH QUERY_DICT
                    np.asarray(resize_image(query_full_mask))                             #REPLACE WITH QUERY_IMAGE
                )
            query_dict = divide_boundary_superpixels_with_partitions(query_dict, np.asarray(resize_image(query_full_mask)), bound_superpixel_gt)
            query_dict = create_dino_feature_for_added_labels(scale_dict1=query_dict)

            query_full_superpixels = np.unique(query_dict['superpixel_labels'])                                                        # DIFFERENCE FROM IF CONDITION (HERE, THE QUERY_FULL_SUPERPIXELS WILL INCLUDE THE WHOLE SUPERPIXELS)


             
        
        # get the cosine distance matrix
        ss = StandardScaler()
        cos_mat_dist = normalize(ss.fit_transform(query_dict['superpixel_features']))@normalize(ss.fit_transform(support_dict['superpixel_features'])).T       #normalize(query) @ normalize(support).T finds the dot product btw each query and support superpixel. ss.fit_transform performs the z-normalisation, while normalise() takes the unit vector along the dimension, or the vector with magnitude one along the superpixel's feature direction 


        # get the superpixel distances and subtract them from the cosine distance matrix
        if 'use_distance_info' in kwargs and kwargs['use_distance_info']:
            if 'distance_lambda' not in kwargs:
                raise("distance_lambda needs to be provided to use superpixel distance information")
            cos_mat_dist -= kwargs['distance_lambda']*get_centroid_distances(support_dict, query_dict)
        
        # select only the required parts (ONLY THE ONES FROM SUPPORT_MASK)
        # print(f"cos_mat_dist.shape initially: {cos_mat_dist.shape}")
        cos_mat_dist = cos_mat_dist[:, support_part_superpixels]
        # print(f"cos_mat_dist.shape after providing support part info: {cos_mat_dist.shape}")

        if 'use_seg_mask' in kwargs and kwargs['use_seg_mask']:
            # print(f"cos_mat_dist.shape initially: {cos_mat_dist.shape}")
            cos_mat_dist = cos_mat_dist[query_full_superpixels]             #CONSIDERING ONLY THE SUPERPIXELS BELONGING TO THE QUERY FULL MASK
            # print(f"cos_mat_dist.shape after providing query object mask info: {cos_mat_dist.shape}")

        # print("Shape:", cos_mat_dist.shape)
        # print("Min:", cos_mat_dist.min())
        # print("Max:", cos_mat_dist.max())
        # print("Mean:", cos_mat_dist.mean())
        # print("Std:", cos_mat_dist.std())
        # print("Any NaN?:", np.isnan(cos_mat_dist).any())
        # print("Any Inf?:", np.isinf(cos_mat_dist).any())

        # apply column-wise softmax
        if 'softmax' in kwargs and kwargs['softmax']:       #(WE CAN STILL HAVE NEGATIVE VALUES HERE, BUT THEY'LL ALL CONTRIBUTE TO A MAGNITUDE OF 1, BUT AFTER SOFTMAX, WE'LL ONLY HAVE POSITIVE VALUES)
            # print(cluster_labels.shape)
            # print(f"cos_mat_dist.shape initially: {cos_mat_dist.shape}")
            cos_mat_dist = softmax(cos_mat_dist, axis=0)                    #FOR A GIVEN SUPPORT SUPERPIXEL, FIND THE PROBABILITY DISTRIBUTION OVER THE QUERY SUPERPIXELS REGARDING WHICH ONE MIGHT CORRESPOND TO THIS SUPPORT SUPERPIXEL
            # print(f"cos_mat_dist.shape after taking softmax: {cos_mat_dist.shape}")
        

        #returning query_part mask superpixel which will be used as GT
        query_dict_copy = query_dict.copy()
        inner_superpixel_part_gt, bound_superpixel_part_gt = inner_and_outer_superpixel_mask(
                    query_dict_copy['superpixel_labels'],                                     
                    np.asarray(resize_image(query_part_mask))                             
                )
        query_dict_part = divide_boundary_superpixels_with_partitions(query_dict_copy, np.asarray(resize_image(query_part_mask)), bound_superpixel_gt)
        query_dict_part = create_dino_feature_for_added_labels(scale_dict1=query_dict_part)
        gt_query_part_superpixels = np.unique(query_dict_part['superpixel_labels'][np.asarray(resize_image(query_part_mask)) > 0]) 
      
    # returning query_dict and the affinity matrix
    return cos_mat_dist





def extract_cos_dist_mat_before_message_passing(support_image, support_part_mask, support_full_mask, query_image, query_part_mask, query_full_mask):
    #RETURNS THE COS DISTANCE MATRIX GIVEN SUPPORT IMAGE + PART MASK/FULL MASK, QUERY IMAGE + PART MASK/FULL MASK

    # MULTI SCALE CASE

    n_sups = 1024
    correct_pose_sups = 1024
    airplane_name = "GraphTraining"

    # Get the result dictionary for the real image
    query_dict1 = get_superpixel_features(model=model, image_processor=image_processor, img=query_image, n_superpixels=n_sups)

    # Get the result dictionary for the rendered image
    support_dict1 = get_superpixel_features(model=model, image_processor=image_processor, img=support_image, n_superpixels=correct_pose_sups)

    # Output path setup
    #out_path = f'./outputs/Experiment_1'
    out_path = './outputs/Experiment_1'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # Get the predicted front_mask
    cos_mat_dist = get_cos_dist_mat_helper(
        support_dict1, 
        query_dict1, 
        support_part_mask, 
        query_full_mask, 
        query_part_mask,
        airplane_name, 
        out_path=out_path, 
        use_seg_mask=True, 
        softmax=True, 
        use_distance_info=True, 
        distance_lambda=3, 
        multi_scale=False, 
        scale_list=correct_pose_sups
    )

        
    return cos_mat_dist

