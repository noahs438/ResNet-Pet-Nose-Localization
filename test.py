import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PetNoseDataset
from model import NoseNet
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Test the NoseNet model')
parser.add_argument('--num-images', type=int, default=44, help='Number of images to process and visualize')
args = parser.parse_args()

model = NoseNet().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to denormalize images
def denormalize(tensor, means, stds):
    denorm = transforms.Normalize(
        [-m / s for m, s in zip(means, stds)],
        [1 / s for s in stds]
    )
    return denorm(tensor)


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

test_image_dir = './data/images/164/'
test_labels_file = './data/images/164/labels.txt'
test_dataset = PetNoseDataset(test_image_dir, test_labels_file, transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def euclidean_distance(predicted, true):
    return np.linalg.norm(predicted - true)


distances = []

for i, (images, true_keypoints, original_sizes) in enumerate(test_loader):
    if args.num_images is not None and i >= args.num_images:
        break
    images = images.to(device)
    predicted_keypoints = model(images)

    # Denormalize the image for visualization - Otherwise in weird colors
    img = denormalize(images[0], means, stds).cpu()
    img = transforms.ToPILImage()(img)

    dist = euclidean_distance(predicted_keypoints.detach().cpu().numpy(), true_keypoints.detach().numpy())
    distances.append(dist)

    plt.imshow(img)
    plt.scatter(*true_keypoints[0].detach().numpy(), color='green', label='Ground Truth')  # Green for ground truth
    plt.scatter(*predicted_keypoints[0].detach().cpu().numpy(), color='red', label='Prediction')  # Red for prediction
    plt.legend()
    plt.show()

min_distance = np.min(distances)
mean_distance = np.mean(distances)
max_distance = np.max(distances)
std_distance = np.std(distances)

print(f'Min Distance: {min_distance}')
print(f'Mean Distance: {mean_distance}')
print(f'Max Distance: {max_distance}')
print(f'Standard Deviation: {std_distance}')
