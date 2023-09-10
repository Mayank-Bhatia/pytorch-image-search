import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


def generate_triplets(hf_dataset, num_triplets=10000):
    """
    Generate a list of triplets (anchor, positive, negative) indices.

    Args:
        hf_dataset (Dataset): HuggingFace dataset object.
        num_triplets (int): Number of triplets to generate.

    Returns:
        list: List of triplets (anchor_idx, positive_idx, negative_idx).
    """
    triplets = []
    num_classes = len(np.unique(hf_dataset["label"]))
    class_indices = [
        np.where(np.array(hf_dataset["label"]) == i)[0].tolist()
        for i in range(num_classes)
    ]

    for _ in range(num_triplets):
        anchor_class = np.random.randint(0, num_classes)
        negative_class = (
            anchor_class + np.random.randint(1, num_classes)
        ) % num_classes

        anchor_idx = np.random.choice(class_indices[anchor_class])
        positive_idx = np.random.choice(class_indices[anchor_class])
        negative_idx = np.random.choice(class_indices[negative_class])

        while positive_idx == anchor_idx:
            positive_idx = np.random.choice(class_indices[anchor_class])

        triplets.append([int(anchor_idx), int(positive_idx), int(negative_idx)])

    return triplets


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


class SiameseNetwork(nn.Module):
    """Siamese Network for generating embeddings of images."""

    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
        )

        sample_output = self.convnet(torch.rand(1, 3, 100, 100))
        flattened_size = sample_output.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


class TripletLoss(nn.Module):
    """Triplet Loss for training the Siamese Network."""

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """Compute the Triplet Loss.

        Args:
            anchor (Tensor): Anchor image embeddings.
            positive (Tensor): Positive image (same class as anchor) embeddings.
            negative (Tensor): Negative image (different class from anchor) embeddings.
        Returns:
            Tensor: Triplet loss value.
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def train_siamese_network(model, dataloader, loss_fn, optimizer, device, epochs=5):
    """
    Training loop for the Siamese Network.

    Args:
        model (SiameseNetwork): Siamese network model.
        dataloader (DataLoader): DataLoader for the training data.
        loss_fn (TripletLoss): Triplet loss function.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): Device ("cuda" or "cpu").
        epochs (int): Number of training epochs.

    Returns:
        list: Loss history.
    """
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for (anchor, positive, negative, _), _ in dataloader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            optimizer.zero_grad()

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        logging.info(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "siamese_network_model.pth")

    return loss_history


def evaluate_siamese_network(model, dataloader, device):
    """
    Evaluate the Siamese Network on test data.

    Args:
        model (SiameseNetwork): Trained Siamese network model.
        dataloader (DataLoader): DataLoader for the test data.
        device (torch.device): Device ("cuda" or "cpu").

    Returns:
        float: Test accuracy.
    """
    model.load_state_dict(torch.load("siamese_network_model.pth"))
    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    with torch.no_grad():
        for batch in dataloader:
            images, labels_batch = batch[0:3], batch[3]
            images = torch.stack(images).to(device)
            embedded = model(images)
            embeddings.append(embedded.cpu().numpy())
            labels.append(labels_batch.cpu().numpy())

    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)

    correct_predictions = 0
    total_predictions = len(embeddings)

    for idx, anchor_embedding in enumerate(embeddings):
        distances = np.linalg.norm(embeddings - anchor_embedding, axis=1)
        distances[idx] = float("inf")
        closest_idx = np.argmin(distances)

        if labels[idx] == labels[closest_idx]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy


def main():
    LR = 0.001
    EPOCHS = 20
    N_TRIPLETS = 5000

    transform = transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using {device}")

    dataset = load_dataset("Matthijs/snacks")
    logging.info(f"Loaded data: {dataset}")

    train_triplets = generate_triplets(dataset["train"], num_triplets=N_TRIPLETS * 2)
    val_triplets = generate_triplets(dataset["validation"], num_triplets=N_TRIPLETS)
    test_triplets = generate_triplets(dataset["test"], num_triplets=N_TRIPLETS)
    logging.info(
        f"Generated {len(train_triplets)}, {len(val_triplets)}, {len(test_triplets)} triplets for train, val, test"
    )

    train_dataset = TripletDataset(
        dataset["train"], train_triplets, transform=transform
    )
    val_dataset = TripletDataset(
        dataset["validation"], val_triplets, transform=transform
    )
    test_dataset = TripletDataset(dataset["test"], test_triplets, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    siamese_net = SiameseNetwork()
    siamese_net.to(device)
    triplet_loss = TripletLoss()
    optimizer = optim.Adam(siamese_net.parameters(), lr=LR)
    logging.info(
        f"Initialized siamese network with Adam optimizer on triplet loss: {siamese_net}"
    )

    logging.info("Starting training...")
    train_siamese_network(
        siamese_net, train_loader, triplet_loss, optimizer, device, epochs=EPOCHS
    )

    logging.info("Evaluating on validation set...")
    val_accuracy = evaluate_siamese_network(siamese_net, val_loader, device)
    logging.info(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    logging.info("Evaluating on test set...")
    test_accuracy = evaluate_siamese_network(siamese_net, test_loader, device)
    logging.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
