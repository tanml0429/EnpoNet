

import os
from pathlib import Path
# dir = "/data/tml/mixed_polyp/LDTest"
# dir = "/data/tml/mixed_polyp_v5_format2/LDTest/images"
dir = "/data/tml/mixed_polyp_unlabeled/LDTest"

num_images = 0
for root, dirs, files in os.walk(dir):
    for file in files:
        if Path(file).suffix in ['.jpg', '.png', '.jpeg']:
            print(file)
            num_images += 1
            pass

print(f"Number of images: {num_images}")