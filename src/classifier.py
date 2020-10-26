
from torch import nn
import torch
import numpy as np

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        dropout_rate = 0.3

        self.do_loss = nn.CrossEntropyLoss()

        self.the_layer = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(input_size, input_size),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(input_size, input_size),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(input_size, input_size),
                nn.GELU(),
                nn.Linear(input_size, num_classes))

        
    def forward(self, x, y):
        x_th = torch.tensor(x)
        return self.the_layer(x_th)
        
    def loss_func(self, logits, labels):
        return self.do_loss(logits, torch.tensor(labels))
        
    def test_forward(self, input):
        logits = self.the_layer(torch.tensor(input)).detach().numpy()
        predictions = np.argmax(logits, axis=1)
        return predictions
