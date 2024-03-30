import imghdr
import os
import shutil

import Augmentor
import cv2

source_file_dir = "D:\\python_projects\\augmentor\\6_test"

converted_folder_path = source_file_dir + "/converted"
output_folder_path = source_file_dir + "/output"
if os.path.exists(converted_folder_path) or os.path.exists(output_folder_path):
    try:
        shutil.rmtree(converted_folder_path)
    except OSError as e:
        pass
    try:
        shutil.rmtree(output_folder_path)
    except OSError as e:
        pass
os.makedirs(converted_folder_path)

# convert non-RGB to RGB and save image file to JPEG format
for root, _, files in os.walk(source_file_dir):
    for file in files:
        file_path = os.path.join(root, file)
        if not imghdr.what(file_path) in ['jpeg', 'jpg', 'png']:
            continue
        new_file_path = converted_folder_path + '/' + file
        cv2.imwrite(new_file_path, cv2.imread(file_path), [cv2.IMWRITE_JPEG_QUALITY, 100])

# set source_directory and output_directory
p = Augmentor.Pipeline(source_directory=converted_folder_path, output_directory=output_folder_path)

p.rotate_without_crop(0.5, max_left_rotation=20, max_right_rotation=20, expand=True)
p.skew_top_bottom(probability=0.3, magnitude=0.5)
p.skew_left_right(probability=0.3, magnitude=0.5)
p.random_brightness(0.1, 0.3, 2.5)

p.status()

p.sample(10 * len(p.augmentor_images))

# transforms = torchvision.transforms.Compose([
#     p.torch_transform(),
#     torchvision.transforms.ToTensor(),
# ])

# delete converted img file
try:
    shutil.rmtree(converted_folder_path)
except OSError as e:
    print(e)
