import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

# Data transformation
transform = transforms.Compose(
    [
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TripletDataset(Dataset):
    """Dataset to generate image triplets: anchor, positive, negative."""

    def __init__(self, hf_dataset, triplets, transform=None):
        """
        Initialize the TripletDataset.

        Args:
            hf_dataset (Dataset): HuggingFace dataset object.
            triplets (list): List of triplets (anchor_idx, positive_idx, negative_idx).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.hf_dataset = hf_dataset
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor = self.hf_dataset[self.triplets[idx][0]]["image"]
        positive = self.hf_dataset[self.triplets[idx][1]]["image"]
        negative = self.hf_dataset[self.triplets[idx][2]]["image"]
        label = self.hf_dataset[self.triplets[idx][0]]["label"]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative, label
