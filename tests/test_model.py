import torch
from src.model import SmallCNN

def test_model_output_shape():
    model = SmallCNN(num_classes=9)
    out = model(torch.randn(4, 3, 28, 28))
    assert out.shape == (4, 9)
