
from test_utils import read_model, test_manually
import sys
import os
from constant import IMAGE_DIR, MASK_DIR

# Add the root directory (where 'src' is located) to sys.path
sys.path.append(r'C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src')

model_folder_path = "./saved_trained_models/SegNet"

model_path = model_folder_path + "/SegNet_best_model.pth"

segnet_model = read_model(model_path)

image_path = os.path.join(IMAGE_DIR, 'cfc_004593.jpg')
mask_path = os.path.join(MASK_DIR, 'cfc_004593.png')

test_results = test_manually(segnet_model, model_folder_path, [image_path], [mask_path])

print(test_results)