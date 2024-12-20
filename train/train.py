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
epochs = 5
cuda = True
input_shape = (224, 224)
n_classes = 2
save_path = "saved_trained_models"  # Directory to save models
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
###############################

image_paths, mask_paths = prepare_images_and_masks_path()

print("Number of images found: ", len(image_paths))

dataset = Dataset(image_paths, mask_paths)

# lets split the dataset into three parts (train 70%, test 15%, validation 15%)
test_size = 0.15
val_size = 0.15

test_amount, val_amount = int(dataset.__len__() * test_size), int(dataset.__len__() * val_size)

# this function will automatically randomly split your dataset but you could also implement the split yourself
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [
            (dataset.__len__() - (test_amount + val_amount)), 
            test_amount, 
            val_amount
])

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

print("Number of images in train set: ", len(train_set))
print("Number of images in validation set: ", len(val_set))
print("Number of images in test set: ", len(test_set))


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
    plot_path = os.path.join(model_output_dir, f"{model_name}_metrics.png")
    metrics_file_path = os.path.join(model_output_dir, f"{model_name}_metrics.txt")

    print(f"\nTraining Model {i+1} ({model_name})")
    train_losses, val_losses, val_accuracies = train_model(config_dict, 
                                                           train_dataloader, 
                                                           val_dataloader, 
                                                           best_model_path, 
                                                           epochs=epochs, 
                                                           device='cuda')


    # Plot training results for this model
    plot_metrics(train_losses, val_losses, val_accuracies)

    best_val_loss = min(val_losses)

    # Save the metrics and plot at the end of training
    save_plot(train_losses, val_losses, val_accuracies, plot_path)
    metrics = {
        "Best Validation Loss": best_val_loss,
        "Final Validation Accuracy": val_accuracies[-1]
    }
    save_metrics(metrics, metrics_file_path)