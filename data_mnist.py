import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_mnist_datasets(valid_size=0.1, random_seed=42):
    # Define transformations for the training and validation sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Create a random permutation of indices for splitting the dataset
    num_train = len(mnist_dataset)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    # Calculate the split index
    split = int(np.floor(valid_size * num_train))

    # Split the indices into training and validation sets
    train_indices, valid_indices = indices[split:], indices[:split]

    train_dataset = Subset(mnist_dataset, train_indices)
    valid_dataset = Subset(mnist_dataset, valid_indices)

    return train_dataset, valid_dataset
