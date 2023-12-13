import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights


def aggregate_votes(vote_maps):
    device = vote_maps.device

    n, c, h, w = vote_maps.size()
    x_coords = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, w).expand(n, 1, h, w)
    y_coords = torch.linspace(-1, 1, h, device=device).view(1, 1, h, 1).expand(n, 1, h, w)

    # Sum over all dimensions except the batch dimension
    x_center = torch.sum(vote_maps * x_coords, dim=[1, 2, 3]) / torch.sum(vote_maps, dim=[1, 2, 3])
    y_center = torch.sum(vote_maps * y_coords, dim=[1, 2, 3]) / torch.sum(vote_maps, dim=[1, 2, 3])

    # Ensure x_center and y_center have an extra dimension for stacking
    x_center = x_center.unsqueeze(1)
    y_center = y_center.unsqueeze(1)

    return torch.stack([x_center, y_center], dim=2).squeeze(1)


# GRADIENTS NULLED OUT?? - VERSION 1
# class NoseNet(nn.Module):
#     def __init__(self):
#         super(NoseNet, self).__init__()
#         self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#         self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
#         self.voting_layer = nn.Linear(512, 2)
#
#     def forward(self, x):
#         feature_maps = self.resnet(x)
#         keypoint_location = self.voting_layer(feature_maps)
#         # vote_maps = torch.sigmoid(vote_maps)  # Apply sigmoid for normalization
#         # keypoint_location = aggregate_votes(vote_maps)
#         return keypoint_location

# NOSENET VERSION 2
class NoseNet(nn.Module):
    def __init__(self):
        super(NoseNet, self).__init__()
        self.resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # Add adaptive pooling
        self.voting_layer = nn.Linear(512, 2)  # Output 2 values for x and y coordinates

    def forward(self, x):
        feature_maps = self.resnet(x)
        pooled_features = self.adaptive_pool(feature_maps)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Flatten the features
        keypoints = self.voting_layer(pooled_features)
        return keypoints
