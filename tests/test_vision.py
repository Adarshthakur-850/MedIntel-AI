import torch
import numpy as np
from ml.vision.models import build_vision_model
from ml.vision.dataset import get_vision_transforms

def test_vision_model_forward():
    model = build_vision_model(arch="resnet18", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 1)

def test_vision_feature_extraction():
    model = build_vision_model(arch="resnet18", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    features = model.extract_features(dummy_input)
    assert features.shape == (2, 512)

def test_vision_transforms():
    train_tf, val_tf = get_vision_transforms(img_size=224)
    assert train_tf is not None
    assert val_tf is not None
