import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2


class PetNoseDataset(Dataset):
    def __init__(self, img_dir, labels_file, transform=None):
        self.img_labels = self.read_labels(labels_file)
        self.img_dir = img_dir
        self.transform = transform

    def read_labels(self, annotations_file):
        labels = {}
        with open(annotations_file, "r") as file:
            for line in file:
                parts = line.strip().split(',"(')
                if len(parts) == 2:
                    image_name = parts[0].strip()
                    keypoints_str = parts[1].strip(')"')
                    keypoints = tuple(map(int, keypoints_str.split(',')))
                    labels[image_name] = keypoints
        return labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, keypoints = list(self.img_labels.items())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)


        if image is None:
            raise RuntimeError(f"Failed to read image {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Original image size
        original_width, original_height = image.shape[1], image.shape[0]

        original_size = (original_width, original_height)

        # Resize image
        if self.transform:
            image = self.transform(image)

        # Rescale keypoints
        x_scale = 224 / original_width
        y_scale = 224 / original_height
        keypoints = torch.tensor([keypoints[0] * x_scale, keypoints[1] * y_scale])

        return image, keypoints, original_size


# Image transformation
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
