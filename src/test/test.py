from test_utils import read_model, test_model_with_loader, read_test_images_masks_from_path
import sys
import os
import torch

from constant import IMAGE_DIR, MASK_DIR
from train.train_utils import prepare_images_and_masks_path, Dataset, tensorize_image, tensorize_mask

# Add the root directory (where 'src' is located) to sys.path
sys.path.append(r'C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src')

# Define normalization parameters (same as during training)
mean = (0.485, 0.456, 0.406)  # Example mean
std = (0.229, 0.224, 0.225)   # Example std

test_images_masks_path = r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\data\test_list.txt"

# Prepare paths (you can modify this to fetch the paths dynamically or use a different method)
image_paths, mask_paths = read_test_images_masks_from_path(test_images_masks_path)

# Create the dataset and apply the same normalization
test_dataset = Dataset(image_paths, mask_paths, mean=mean, std=std)

# Create the DataLoader for the test set
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,  # Use the batch size used during training
    shuffle=False  # No need to shuffle for testing
)

# Load the pre-trained models
segnet_model_folder_path = r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src\saved_trained_models\SegNet"
segnet_model_path = os.path.join(segnet_model_folder_path, "SegNet_best_model.pth")
segnet_model = read_model(segnet_model_path)

fcn_model_folder_path = r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src\saved_trained_models\FCN"
fcn_model_path = os.path.join(fcn_model_folder_path, "FCN_best_model.pth")
fcn_model = read_model(fcn_model_path)

unet_model_folder_path = r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src\saved_trained_models\UNet"
unet_model_path = os.path.join(unet_model_folder_path, "UNet_best_model.pth")
unet_model = read_model(unet_model_path)

deeplabv3_model_folder_path = r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src\saved_trained_models\DeepLabV3"
deeplabv3_model_path = os.path.join(deeplabv3_model_folder_path, "DeepLabV3_best_model.pth")
deeplabv3_model = read_model(deeplabv3_model_path)

# List of models to test
models_list = [[segnet_model, segnet_model_folder_path], [fcn_model, fcn_model_folder_path], [unet_model, unet_model_folder_path], [deeplabv3_model, deeplabv3_model_folder_path]]

model_results = {}
# Evaluate each model
for model_list in models_list:
    model = model_list[0]
    model_path = model_list[1]
    print("Model: ", model.__class__.__name__)
    print("Test with loader")
    
    # Test the model with the DataLoader
    result = test_model_with_loader(model, model_path, test_dataloader)
    print(result)
    model_results[model.__class__.__name__] = result

save_path = os.path.join(r"C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src\saved_trained_models", "test_results.txt")
with open(save_path, "w") as f:
    for key, value in model_results.items():
        f.write(f"{key}: \n{value}\n")

                  