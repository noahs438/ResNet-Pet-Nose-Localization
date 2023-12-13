import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PetNoseDataset
import time
from model import NoseNet
import torch.nn as nn

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def rescale_keypoints(keypoints, original_size):
    original_width, original_height = original_size
    x_scale = original_width / 224
    y_scale = original_height / 224
    rescaled_keypoints = torch.tensor([keypoints[0] * x_scale, keypoints[1] * y_scale])
    return rescaled_keypoints


def train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    epoch_losses = []
    epoch_val_losses = []

    for epoch in range(num_epochs):
        # Start time for the epoch
        epoch_start_time = time.time()

        # Training Phase
        model.train()
        train_loss = 0
        num_train_samples = 0
        for images, keypoints, _ in train_loader:
            images = images.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images).to(device)

            # Print model outputs for debugging
            # print("Model outputs:", outputs)

            loss = criterion(outputs, keypoints)
            loss.backward()

            # Implement gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            num_train_samples += images.size(0)

        avg_train_loss = train_loss / num_train_samples
        epoch_losses.append(avg_train_loss)

        # RESIZING CAUSING BIZARRE VALIDATION LOSS
        # # Validation Phase
        # model.eval()
        # val_loss = 0
        # num_samples = 0
        # with torch.no_grad():
        #     for images, true_keypoints, original_sizes in test_loader:
        #         images = images.to(device)
        #         true_keypoints = true_keypoints.to(device)
        #
        #         # Get predictions
        #         predicted_keypoints = model(images)
        #
        #         # Rescale predicted keypoints
        #         rescaled_keypoints = []
        #         for i in range(len(predicted_keypoints)):
        #             width, height = original_sizes[i * 2].item(), original_sizes[i * 2 + 1].item()
        #             rescaled_keypoints.append(rescale_keypoints(predicted_keypoints[i], (width, height)))
        #
        #         rescaled_keypoints = torch.stack(rescaled_keypoints).to(device)
        #
        #         # Compute loss
        #         loss = criterion(rescaled_keypoints, true_keypoints)
        #         val_loss += loss.item()
        #         num_samples += images.size(0)
        #
        # avg_val_loss = val_loss / num_samples
        # epoch_val_losses.append(avg_val_loss)

        scheduler.step()

        # End time for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f'Epoch [{epoch + 1}/{num_epochs}], Duration: {epoch_duration:.2f} sec, Training Loss: {avg_train_loss:.4f}')
        # , Validation Loss: {avg_val_loss:.4f}

    # Save the model
    torch.save(model.state_dict(), 'model.pth')

    # Plot the training and validation losses
    plt.plot(epoch_losses, label='Training Loss')
    # plt.plot(epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    # plt.legend()
    plt.savefig('loss.png')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PetNet')
    parser.add_argument('-b', type=int, default=16, help='batch size')
    parser.add_argument('-e', type=int, default=20, help='number of epochs')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    train_image_dir = './data/images-original/images/'
    train_labels_file = 'train_noses.3.txt'

    test_image_dir = './data/images-original/images/'
    test_labels_file = 'test_noses.txt'

    train_dataset = PetNoseDataset(train_image_dir, train_labels_file, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True)

    test_dataset = PetNoseDataset(test_image_dir, test_labels_file, transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = NoseNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)


    # Re-add test_loader to train function if want to use validation loss
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.e)
