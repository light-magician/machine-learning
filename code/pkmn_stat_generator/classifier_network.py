import torch
import torch.nn as nn
import torch.optim as optim

# neural net for classification
class Classifier(nn.Module):
    def __init__(self, input_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear