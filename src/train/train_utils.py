import os
import glob

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import tqdm

from preprocess import tensorize_image, tensorize_mask

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
AUG_IMAGE = os.path.join(DATA_DIR, 'aug_images')
AUG_MASK = os.path.join(DATA_DIR, 'aug_masks')
VIP_AUG_IMG = os.path.join(DATA_DIR, 'vip_aug_img')
VIP_AUG_MASK = os.path.join(DATA_DIR, 'vip_aug_mask')
###############################

input_shape = (224, 224)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    # get sample
    def __getitem__(self, idx):
        image_file = self.image_paths[idx]
        mask_file = self.mask_paths[idx]

        # Make absolute paths
        image_file = r"{}".format(os.path.abspath(image_file))
        mask_file = r"{}".format(os.path.abspath(mask_file))

        image = tensorize_image([image_file], input_shape)
        target = tensorize_mask([mask_file], input_shape, n_class=2)

        # Remove batch dimension, Because dataloader puts batch dimension
        image = image.squeeze(0)
        target = target.squeeze(0)

        return image, target

    def __len__(self):
        return len(self.image_paths)


def prepare_images_and_masks_path():

    image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
    image_path_list.sort()

    mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
    mask_path_list.sort()

    aug_image_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
    aug_mask_path_list  = glob.glob(os.path.join(AUG_MASK,  '*'))


    vip_aug_img = glob.glob(os.path.join(VIP_AUG_IMG, '*'))
    vip_aug_mask = glob.glob(os.path.join(VIP_AUG_MASK, '*'))

    total_images = image_path_list + aug_image_path_list + vip_aug_img
    
    total_labels = mask_path_list + aug_mask_path_list + vip_aug_mask

    total_images = image_path_list + vip_aug_img
    total_labels = mask_path_list + vip_aug_mask

    return total_images, total_labels


def train_model(config_dict, train_loader, val_loader, best_model_path, epochs=10, device='cuda'):
    model = config_dict["model"].to(device)
    criterion = config_dict["criterion"]
    optimizer = config_dict["optimizer"]

    # Store loss and accuracy for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, targets in tqdm.tqdm(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Calculate average training loss for the epoch
        train_losses.append(running_loss / len(train_loader))

        # Evaluate model after each epoch
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Save the model if validation loss improves
        current_val_loss = val_losses[-1]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_best_model(
                model=config_dict['model'],
                val_loss=best_val_loss,
                save_path=best_model_path
            )

        print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_losses[-1]:.4f} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.2f}%")

    return train_losses, val_losses, val_accuracies

# Modify evaluate_model function to return loss and accuracy
def evaluate_model(model, val_loader, device='cuda'):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            _, batch_label_2 = torch.max(targets.data, 1)
            total += targets.numel()
            correct += (predicted.data == batch_label_2.data).sum().float()

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total * 2
    return avg_loss, accuracy

# Function to plot training loss and accuracy graphs
def plot_metrics(train_losses, val_losses, val_accuracies):

    print("type of train losses..: ", type(val_accuracies))
    print("type of train loss..: ", type(val_accuracies[0]))

    val_accuracies_list = []
    for val_ac in val_accuracies:
        val_accuracies_list.append(val_ac.cpu())


    epochs = range(1, len(train_losses) + 1)

    # Plotting training and validation loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies_list, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def save_best_model(model, val_loss, save_path):
    """
    Save the best model during training.

    Args:
        model (torch.nn.Module): The model to save.
        val_loss (float): The best validation loss achieved so far.
        save_path (str): The file path to save the model checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model, save_path)
    print(f"Best model saved to {save_path} with validation loss {val_loss:.4f}")

def save_metrics(metrics, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
    print(f"Metrics saved to {save_path}")

def save_plot(train_losses, val_losses, val_accuracies, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    val_accuracies_list = []
    for val_ac in val_accuracies:
        val_accuracies_list.append(val_ac.cpu())

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend()
    plt.title("Training and Validation Losses")
    plt.savefig(save_path)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(val_accuracies_list, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracies")
    plt.title("Validation Accuracy")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")