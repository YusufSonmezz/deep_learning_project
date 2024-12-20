import os
import sys

# Add the root directory (where 'src' is located) to sys.path
sys.path.append(r'C:\Users\Yusuf\Desktop\YL\1. Donem\Derin Ogrenme\Proje\src')

print("constant root dir: ",os.getcwd())

# Path to jsons
JSON_DIR = '../data/jsons'

# Path to mask
MASK_DIR  = '../data/masks'
if not os.path.exists(MASK_DIR):
    os.mkdir(MASK_DIR)

# Path to output images
IMAGE_OUT_DIR = '../data/masked_images'
if not os.path.exists(IMAGE_OUT_DIR):
    os.mkdir(IMAGE_OUT_DIR)

# Path to original images
IMAGE_DIR = '../data/images'


# In order to visualize masked-image(s), change "False" with "True"
VISUALIZE = False

# Bacth size
BACTH_SIZE = 4

# Input dimension
HEIGHT = 224
WIDTH = 224

# Number of class, for this task it is 2: Non-drivable area and Driviable area
N_CLASS= 2

# Path to augmentation images
AUG_IMAGE = '../data/aug_images'
if not os.path.exists(AUG_IMAGE):
    os.mkdir(AUG_IMAGE)

# Path to augmentation images
AUG_MASK = '../data/aug_masks'
if not os.path.exists(AUG_MASK):
    os.mkdir(AUG_MASK)

VIP_AUG_IMG = '../data/vip_aug_img'
VIP_AUG_MASK = '../data/vip_aug_mask'