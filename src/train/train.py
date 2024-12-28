import sys
import os

import torch.optim as optim

# Add the root directory (where 'src' is located) to sys.path
sys.path.append(r'C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src')

from models.model_fcn import FCN
from models.model_segnet import SegNet
from models.model_unet import UNet
from models.model_deeplabv3 import DeepLabV3

from train_utils import *

######### Parameters #########
valid_size = 0.15
test_size  = 0.15
B = 10  # batch size
epochs = 6
cuda = True
input_shape = (224, 224)
n_classes = 2
save_path = "saved_trained_models"  # Directory to save models
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
###############################

image_paths, mask_paths = prepare_images_and_masks_path()

print("Number of images found: ", len(image_paths))

# Initialize dataset
dataset = Dataset(
    image_paths=image_paths,
    mask_paths=mask_paths,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
)

# Split the dataset into three parts (train 70%, test 15%, validation 15%)
test_size = 0.15
val_size = 0.15

test_amount = int(len(dataset) * test_size)
val_amount = int(len(dataset) * val_size)

train_set, val_set, test_set = torch.utils.data.random_split(
    dataset, 
    [(len(dataset) - (test_amount + val_amount)), test_amount, val_amount]
)

# Saving test images and masks paths to evaluate after training
test_image_paths = [dataset.image_paths[idx] for idx in test_set.indices]
test_mask_paths = [dataset.mask_paths[idx] for idx in test_set.indices]

# Define dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=B,
    shuffle=True,
)

val_dataloader = torch.utils.data.DataLoader(
    val_set,
    batch_size=B,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=B,
    shuffle=True,
)

# Paths for saving lists
test_list_path = r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\data\test_list.txt"

save_list(test_list_path, test_image_paths, test_mask_paths)

# Print dataset size
print("Number of images in test set:", len(test_set))

# Call Models
unet = UNet(input_shape, n_classes)
fcn = FCN(n_classes)
segnet = SegNet(n_classes)
deeplabv3 = DeepLabV3(n_classes)

fcn_config = {
    "model": fcn,
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": optim.Adam(fcn.parameters(), lr=1e-4, weight_decay=1e-6),
}

unet_config = {
    "model": unet,
    "criterion": nn.BCELoss(),
    "optimizer": optim.Adam(unet.parameters(), lr=1e-4, weight_decay=1e-6),
}

segnet_config = {
    "model": segnet,
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": optim.Adam(segnet.parameters(), lr=1e-4, weight_decay=1e-6),
}

deeplabv3_config = {
    "model": deeplabv3,
    "criterion": nn.CrossEntropyLoss(),
    "optimizer": optim.Adam(deeplabv3.parameters(), lr=1e-4, weight_decay=1e-6),
}

models_config = [segnet_config, fcn_config, unet_config, deeplabv3_config]

for i, config_dict in enumerate(models_config):
    model_name = config_dict['model'].__class__.__name__

    # Create a directory to save the model
    model_output_dir = os.path.join(save_path, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    best_model_path = os.path.join(model_output_dir, f"{model_name}_best_model.pth")
    plot_path = os.path.join(model_output_dir, f"{model_name}_metrics")
    metrics_file_path = os.path.join(model_output_dir, f"{model_name}_metrics.json")

    print(f"\nTraining Model {i+1} ({model_name})")
    train_losses, val_losses, val_accuracies = train_model(
        config_dict, 
        train_dataloader, 
        val_dataloader, 
        best_model_path, 
        epochs=epochs, 
        device='cuda'
    )

    # Log epoch-wise data
    epoch_metrics = []
    for epoch in range(len(train_losses)):
        epoch_metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_losses[epoch],
            "val_loss": val_losses[epoch],
            "val_accuracy": val_accuracies[epoch].item() if hasattr(val_accuracies[epoch], 'item') else val_accuracies[epoch],
        })

    # Calculate summary metrics
    best_val_loss = min(val_losses)
    best_epoch = val_losses.index(best_val_loss) + 1
    final_val_accuracy = val_accuracies[-1].item() if hasattr(val_accuracies[-1], 'item') else val_accuracies[-1]

    summary_metrics = {
        "Best Validation Loss": best_val_loss,
        "Best Epoch": best_epoch,
        "Final Validation Accuracy": final_val_accuracy,
        "Total Epochs": len(train_losses),
    }

    # Combine all metrics into a single JSON
    all_metrics = {
        "epoch_metrics": epoch_metrics,
        "summary_metrics": summary_metrics,
    }

    # Save plots and metrics
    save_plot(train_losses, val_losses, val_accuracies, plot_path)
    save_metrics(all_metrics, metrics_file_path)

    print(f"Model {model_name} training completed.")
