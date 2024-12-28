import torch
from PIL import Image
import numpy as np
import os
import sys
import cv2
import tqdm

# Add the root directory (where 'src' is located) to sys.path
sys.path.append(r'C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src')

from preprocess import tensorize_image, tensorize_mask

def read_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

def test_model_with_loader(model, model_path, test_loader, device = "cuda"):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    model = model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for i, (data, targets, image_path, real_mask_path) in enumerate(tqdm.tqdm(test_loader)):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            _, batch_label_2 = torch.max(targets.data, 1)
            total += targets.numel()
            correct += (predicted.data == batch_label_2.data).sum().float()

            # Convert the output to a numpy array
            result = outputs.argmax(axis = 1)
            result = result.cpu()
            result_np = result.detach().numpy()
            result_np = np.squeeze(result_np, axis = 0)

            img = cv2.imread(image_path[0])
            result_np = cv2.resize(result_np.astype(np.uint8), (1920, 1208))
            #result_np = denormalize(result_np, mean, std)

            images_folder = os.path.join(model_path, "test_images")
            os.makedirs(images_folder, exist_ok=True)
            save_outputs_on_images(result_np, img, images_folder, i)

    accuracy = 100 * correct / total * 2
    total = total / len(test_loader)
    info_dict = {
        "total": total,
        "accuracy": accuracy
    }
    return info_dict

def denormalize(image, mean, std):
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return image * std + mean

def test_manually(model, model_path, test_image, test_mask, name = "Default", device = "cuda"):
    model = model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        data, target = tensorize_image(test_image, (224, 224), True), tensorize_mask(test_mask, (224, 224), 2, True)
        data, target = data.to(device), target.to(device)

        outputs = model(data)

        _, predicted = torch.max(outputs.data, 1)
        _, batch_label_2 = torch.max(target.data, 1)
        total += target.numel()
        correct += (predicted.data == batch_label_2.data).sum().float()

        # Convert the output to a numpy array
        result = outputs.argmax(axis = 1)
        result = result.cpu()
        result_np = result.detach().numpy()
        result_np = np.squeeze(result_np, axis = 0)

        img = cv2.imread(test_image[0])
        result_np = cv2.resize(result_np.astype(np.uint8), (1920, 1208))

        images_folder = model_path + "/test_images"
        os.makedirs(images_folder, exist_ok=True)
        save_outputs_on_images(result_np, img, images_folder, name)

    accuracy = 100 * correct / total * 2
    info_dict = {
        "total": total,
        "accuracy": float(accuracy)
    }
    return info_dict

def save_outputs_on_images(output, data, model_folder_path, iteration):
    # Create a mask
    mask = np.zeros_like(data)
    if output.shape == mask.shape[:2]:
        mask[output == 1] = 255
    else:
        return

    copy_image = data.copy()
    copy_image[output == 1, :] = (255, 0, 125)
    opac_image=(copy_image/2+data/2).astype(np.uint8)

    saved_path = model_folder_path + f"/{iteration}_predict.png"
    cv2.imwrite(saved_path, opac_image.astype(np.uint8))

def read_test_images_masks_from_path(test_list_path):
    test_image_paths, test_mask_paths = [], []
    with open(test_list_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Strip leading/trailing whitespace and split by comma
            line = line.strip()
            if line:  # Ensure line is not empty
                parts = line.split(",")
                if len(parts) == 2:
                    image_path, mask_path = parts
                    # Make the paths absolute (if they are not already)
                    image_path = os.path.abspath(image_path)
                    mask_path = os.path.abspath(mask_path)

                    test_image_paths.append(image_path)
                    test_mask_paths.append(mask_path)
                else:
                    print(f"Skipping invalid line: {line}")
    return test_image_paths, test_mask_paths
        