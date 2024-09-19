import numpy as np
import torch
from torch import nn
import  transformer as F

class MSTCRB(torch.nn.Module):
    def __init__(self):
        super(MSTCRB, self).__init__()

        self.linear1 = torch.nn.Linear(3, 11)
        self.linear2 = torch.nn.Linear(11, 21)
        self.linear3 = torch.nn.Linear(21, 1)
        self.attention_layer = nn.Linear(11, 1)
        self.conv = nn.Conv1d(101, 4, kernel_size=1)
        self.Encoder = F.Encoder(105, 210, 3, 6, 0.3)
        self.x1_linear1 = torch.nn.Linear(105, 105)
        self.conv1 = nn.Conv1d(101, 4, kernel_size=85)
        self.BN = nn.BatchNorm1d(4)
        self.BN1 = nn.BatchNorm1d(101)
        self.dropout = nn.Dropout(0.5)
        self.dropout1 = nn.Dropout(0.2)
        self.x0_linear1 = torch.nn.Linear(84, 21)
        self.conv1dx0 = nn.Conv1d(101, 4, kernel_size=65)
        self.conv1dx1 = nn.Conv1d(99, 4, kernel_size=2)
        self.conv1dx2 = nn.Conv1d(101, 4, kernel_size=81)
        self.outputconv1dx2 = nn.Conv1d(101, 4, kernel_size=1)
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 21)
        self.fc3 = torch.nn.Linear(21, 1)
        self.softmax = torch.nn.Softmax(dim=2)


    def forward(self, data,epoch):
        #data:
        # 0：101,84-kmer
        # 1：101,3-NCP
        # 2：101,11-DPCP
        # 3: 101,21 -KNF
        # 4: 101,101-pair
        DPCP = data["DPCP"].to(torch.float32)
        NCP = data["NCP"].to(torch.float32)
        DPCP = self.BN1(DPCP)
        NCP = self.BN1(NCP)
        NCP = self.linear1(NCP)
        NCP = torch.relu(NCP)
        attention_weights = torch.sigmoid(self.attention_layer(NCP))
        x0 = attention_weights * NCP + (1 - attention_weights) * DPCP
        x0 = self.conv(x0)
        x0 = torch.sigmoid(x0)
        x0 = self.linear2(x0)
        #x0 = self.BN(x0)
        x0 = torch.sigmoid(x0)
        x0 = self.dropout(x0)

        Kmer = data["Kmer"].to(torch.float32)
        knf = data["knf"].to(torch.float32)
        x1 = torch.cat([Kmer, knf], dim=2)
        x1 = self.Encoder(x1)
        x1 = self.x1_linear1(x1)
        x1 = self.softmax(x1)
        x1 = self.conv1(x1)
        x1 = torch.relu(x1)
        x1 = self.dropout1(x1)

        Y = data["Y"].to(torch.float32)
        x2 = data["pair"].to(torch.float32)
        seqx = x0 + x1
        x2 = self.conv1dx2(x2)
        x2_gate = torch.sigmoid(x2)
        x2 = self.BN(x2)
        x2 = torch.relu(x2)
        x_gate = x2_gate *seqx
        x_end = x_gate+x2
        x = self.linear3(x_end)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.mean(x, dim=1)
        x = torch.sigmoid(x)
        return x
