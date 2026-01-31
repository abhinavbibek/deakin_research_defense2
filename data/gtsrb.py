import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from .prefetch import prefetch_transform

class GTSRB(Dataset):
    """German Traffic Sign Recognition Benchmark (GTSRB).
    
    Args:
        root (string): Root directory of dataset.
        transform (callable, optional): A function/transform that takes in an PIL image and returns
            a transformed version.
        train (bool): If True, creates dataset from training set, otherwise creates from test set.
        prefetch (bool): If True, remove ``ToTensor`` and ``Normalize`` in
            ``transform["remaining"]``, and turn on prefetch mode.
    """
    
    def __init__(self, root, transform=None, train=True, prefetch=False):
        self.train = train
        self.pre_transform = transform["pre"]
        self.primary_transform = transform["primary"]
        if prefetch:
            self.remaining_transform, self.mean, self.std = prefetch_transform(
                transform["remaining"]
            )
        else:
            self.remaining_transform = transform["remaining"]
        self.prefetch = prefetch
        
        # Load GTSRB data
        root = os.path.expanduser(root)
        if train:
            pkl_file = os.path.join(root, "train.pkl")
        else:
            pkl_file = os.path.join(root, "test.pkl")
        
        # Ensure file exists before loading to give clear error
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"GTSRB pickle file not found at {pkl_file}. Please ensure dataset is prepared as pickles.")

        with open(pkl_file, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")
        
        # GTSRB data in pickle is expected to be dict with 'images' and 'labels'
        # Adjust based on how CIFAR pickle was handled if needed, but standard GTSRB pickles usually have this structure
        self.data = np.array(data_dict["images"]) 
        self.targets = np.array(data_dict["labels"])
        
        # Verification of shape
        # GTSRB images are usually resized to 32x32 for these benchmarks
        # if they come as list of arrays/strings, we might need adjustments.
        # Assuming pre-processed pickles similar to CIFAR structure.
        
        # If data is BHWC
        if self.data.ndim == 4 and self.data.shape[3] == 3:
            pass
        # If data is flattened or needs reshaping, logic would go here.
        # For now assuming standard preprocessed 32x32 pickle format.
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Handle cases where image might be a path string or bytes (depends on pickle creation)
        # But assuming np.ndarray from __init__
        
        if not isinstance(img, Image.Image):
             img = Image.fromarray(img.astype(np.uint8))
        
        # Pre-processing
        if self.pre_transform is not None:
            img = self.pre_transform(img)
        # Primary transformation
        img = self.primary_transform(img)
        # Remaining transformation
        img = self.remaining_transform(img)
        if self.prefetch:
            img = np.rollaxis(np.array(img, dtype=np.uint8), 2)
            img = torch.from_numpy(img)
        
        item = {"img": img, "target": target}
        return item
    
    def __len__(self):
        return len(self.data)
