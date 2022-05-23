import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gumbel import gumbel_softmax

class ChartPredictor2d(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, temp=1e-1):
        super(ChartPredictor2d, self).__init__()


        self.fc1_y = nn.Linear(x_dim, h_dim)
        self.fc2_y = nn.Linear(h_dim, h_dim)
        self.fc3_y = nn.Linear(h_dim, y_dim)

        self.relu = nn.ReLU()

        self.temperature = temp

    def forward(self, x):

        y = self.fc1_y(x)
        y = self.relu(y)
        y = self.fc2_y(y)
        y = self.relu(y)
        logit = self.fc3_y(y)
        probs = F.softmax(logit, dim=-1)

        return gumbel_softmax(logit, self.temperature), probs


class ChartPredictor3d(nn.Module):
    def __init__(self, x_dim, h_dim, y_dim, tempe=1e-1, gumbel=True):
        super(ChartPredictor3d, self).__init__()

        self.fc1_y = nn.Linear(x_dim, h_dim)
        self.fc2_y = nn.Linear(h_dim, h_dim)
        self.fc3_y = nn.Linear(h_dim, y_dim)

        s_X_dim=128
        self._hyper_bias_1 = nn.Linear(s_X_dim, h_dim, bias=False)
        self._hyper_gate_1 = nn.Linear(s_X_dim, h_dim)
        
        self._hyper_bias_2 = nn.Linear(s_X_dim, h_dim, bias=False)
        self._hyper_gate_2 = nn.Linear(s_X_dim, h_dim)

        self._hyper_bias_3 = nn.Linear(s_X_dim, y_dim, bias=False)
        self._hyper_gate_3 = nn.Linear(s_X_dim, y_dim)

        self.gumbel = gumbel
        self.temperature = tempe
    
    def forward(self, x, z):

        gate_1 = torch.sigmoid(self._hyper_gate_1(z).unsqueeze(1))
        bias_1 = self._hyper_bias_1(z).unsqueeze(1)
        gate_2 = torch.sigmoid(self._hyper_gate_2(z).unsqueeze(1))
        bias_2 = self._hyper_bias_2(z).unsqueeze(1)
        gate_3 = torch.sigmoid(self._hyper_gate_3(z).unsqueeze(1))
        bias_3 = self._hyper_bias_3(z).unsqueeze(1)

        y = F.relu(self.fc1_y(x) * gate_1 + bias_1)
        y = F.relu(self.fc2_y(y) * gate_2 + bias_2)
        logit = self.fc3_y(y) * gate_3 + bias_3
        probs = F.softmax(logit, dim=-1)

        if self.gumbel:
            return gumbel_softmax(logit, self.temperature), probs
        else:
            return logit

class ChartGenerator(nn.Module):
    def __init__(self, input_dim, y_dim):
        super(ChartGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, y_dim)
        
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, 1)