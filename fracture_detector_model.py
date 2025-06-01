# fracture_detector_model.py

import torch.nn as nn
from torchvision import models

def get_fracture_model():
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model
