import torch
from torchvision import transforms
from PIL import Image
import os


def down_sampled_img(input_folder, output_folder, image_size_scale):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = transforms.Compose([
        transforms.Resize((int(1024 * image_size_scale / 100), int(1024 * image_size_scale / 100))),
        transforms.ToTensor()
    ])

    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            img_resized = transform(img)

            #converting tensor to image
            img_resized_pil = transforms.ToPILImage()(img_resized)

            #saving the images
            img_resized_pil.save(os.path.join(output_folder, filename))



input_folder = r'D:\seminar\dataset\train\normal'
output_folder = r'D:\seminar\dataset\train\down_img'
image_size_scale = 50

down_sampled_img(input_folder, output_folder, image_size_scale)