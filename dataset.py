import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np

def custom_transform(img):
    """
    Resize to (224, 224), convert to grayscale, and return as numpy.ndarray
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("L")  # convert to grayscale
    img = img.resize((224, 224), resample=Image.BILINEAR)
    img_np = np.array(img)  # shape: (224, 224), dtype: uint8

    return img_np

class PartQueryDataset(Dataset):
    def __init__(self, root_dir, supp_dict, transform=None):
        self.root_dir = root_dir
        self.folder_names = os.listdir(root_dir)
        self.folder_paths = [os.path.join(root_dir, folder) for folder in self.folder_names]
        self.supp_dict = supp_dict
        self.transform = transform

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]
        folder_name = os.path.basename(folder_path)

        # Extract object_id and part_id from the folder name
        try:
            parts = folder_name.split('_')
            object_id = int(parts[-2])
            part_id = int(parts[-1])
        except ValueError:
            raise RuntimeError(f"Cannot parse object_id and part_id from folder name: {folder_name}")

        # Load image and masks
        query_image = Image.open(os.path.join(folder_path, 'image.jpg')).convert("RGB")
        query_full_mask = Image.open(os.path.join(folder_path, 'object_mask.png')).convert("L")
        query_part_mask = Image.open(os.path.join(folder_path, 'part_mask.png')).convert("L")

        # Apply transform (if any)
        if self.transform:
            query_image = self.transform(query_image)
            query_full_mask = self.transform(query_full_mask)
            query_part_mask = self.transform(query_part_mask)
        else:
            # Default: convert to tensor
            query_image = torch.from_numpy(np.array(query_image)).permute(2, 0, 1).float() / 255.0
            query_full_mask = torch.from_numpy(np.array(query_full_mask)).unsqueeze(0).float()
            query_part_mask = torch.from_numpy(np.array(query_part_mask)).unsqueeze(0).float()

        # Get support from supp_dict using part_id
        support_info = self.supp_dict.get(part_id, None)
        if support_info is None:
            raise KeyError(f"Support data for part_id {part_id} not found in supp_dict.")

        # support_image = support_info['image']
        # support_full_mask = support_info['obj_mask']
        # support_part_mask = support_info['part_mask']

        # Apply transform to support data
        support_image = self.transform(support_info["image"])
        support_full_mask = self.transform(support_info["obj_mask"])
        support_part_mask = self.transform(support_info["part_mask"])

        return {
            "query_image": query_image,
            "query_full_mask": query_full_mask,
            "query_part_mask": query_part_mask,
            "object_id": object_id,
            "part_id": part_id,
            "support_image": support_image,
            "support_full_mask": support_full_mask,
            "support_part_mask": support_part_mask
        }
