
import torch
import torch.nn as nn


class QNN(nn.Module):
  "DNN to approximate the Q-function"
  def __init__(self):
    super(QNN, self).__init__()
    self.model = torch.nn.Sequential(
      torch.nn.Conv2d(3, 32, 5),
      torch.nn.MaxPool2d((2,2)),
      torch.nn.ReLU(),
      torch.nn.Conv2d(32,64, 5),
      torch.nn.MaxPool2d((2,2)),
      torch.nn.ReLU(),
      torch.nn.Conv2d(64,128, 5),
      torch.nn.MaxPool2d((2,2)),
      torch.nn.ReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear(12800, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, 13))

  def forward(self, x):
    return self.model(x)


