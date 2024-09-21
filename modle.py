import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import  transformer as F1


class MSTCRB(torch.nn.Module):
    def __init__(self):
        super(MSTCRB, self).__init__()

        self.linear1 = torch.nn.Linear(3, 11)
        self.attention_layer = nn.Linear(11, 1)
        self.conv11 = nn.Conv1d(in_channels=11, out_channels=102, kernel_size=7, stride=1, padding=0)
        self.Encoder = F1.Encoder(105, 210, 3, 6, 0.3)
        self.conv12 = nn.Conv1d(in_channels=105, out_channels=102, kernel_size=7, stride=1, padding=0)
        self.BN = nn.BatchNorm1d(101)
        self.conv13 = nn.Conv1d(in_channels=101, out_channels=102, kernel_size=7, stride=1, padding=0)


        self.conv1 = nn.Conv1d(in_channels=21, out_channels=102, kernel_size=7, stride=1, padding=0)

        self.pool = nn.AvgPool1d(kernel_size=5)

        self.dropout1 = nn.Dropout(0.8)

        self.lstm = nn.LSTM(input_size=102, hidden_size=120, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(120 * 2 * (99 // 5), 102)
        self.fc11 = nn.Linear(4560, 102)
        self.dropout2 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(102, 2)

        self.fc13 = nn.Linear(1938, 512)
        self.fc14 = nn.Linear(512, 264)
        self.fc15 = nn.Linear(264, 102)

        self.linear2 = nn.Linear(3, 21)


    def forward(self, data,epoch):
        #data:
        # 0：101,84-kmer
        # 1：101,3-NCP
        # 2：101,11-DPCP
        # 3: 101,21 -KNF
        # 4: 101,101-pair

        DPCP = data["DPCP"].to(torch.float32)
        NCP = data["NCP"].to(torch.float32)
        DPCP = self.BN(DPCP)
        NCP = self.linear1(NCP)
        NCP = torch.relu(NCP)
        attention_weights = torch.sigmoid(self.attention_layer(NCP))
        x0 = attention_weights * NCP + (1 - attention_weights) * DPCP
        x0 = x0.transpose(1, 2)
        x0 = F.relu(self.conv11(x0))
        x0 = self.pool(x0)
        x0 = self.dropout2(x0)
        # x0 = x0.transpose(1, 2)
        # x0, _ = self.lstm(x0)
        # x0 = x0.contiguous().view(x0.size(0), -1)
        # x0 = F.relu(self.fc11(x0))
        # x0 = self.dropout2(x0)
        # x0 = self.fc2(x0)

        Kmer = data["Kmer"].to(torch.float32)
        knf = data["knf"].to(torch.float32)
        x1 = torch.cat([Kmer, knf], dim=2)
        x1 = self.Encoder(x1)
        x1 = x1.transpose(1, 2)
        x1 = F.relu(self.conv12(x1))
        x1 = self.pool(x1)
        x1 = self.dropout2(x1)
        # x1 = x0 + x1
        # x1 = x1.transpose(1, 2)
        # x1, _ = self.lstm(x1)
        # x1 = x1.contiguous().view(x1.size(0), -1)
        # x1 = F.relu(self.fc11(x1))
        # x1 = self.dropout2(x1)
        # x1 = self.fc2(x1)

        x01 = x0 + x1
        x2 = data["pair"].to(torch.float32)
        x2 = x2.transpose(1, 2)
        x2 = F.relu(self.conv13(x2))
        x2 = self.pool(x2)
        x2 = self.dropout1(x2)
        x2_gate = torch.sigmoid(x2)
        x2 = x2 * x2_gate
        x2 = 0.9 * x01 + 0.1 * x2
        # x2 = x2.transpose(1, 2)
        # x2, _ = self.lstm(x2)

        x2 = x2.contiguous().view(x2.size(0), -1)
        x2 = F.relu(self.fc13(x2))
        x2 = self.dropout2(x2)
        x2 = F.relu(self.fc14(x2))
        x2 = self.dropout2(x2)
        x2 = F.relu(self.fc15(x2))
        x2 = self.dropout2(x2)
        x2 = self.fc2(x2)




        # 使用softmax进行分类
        x = F.softmax(x2, dim=1)


        # Kmer = data["Kmer"].to(torch.float32)
        # DPCP = data["DPCP"].to(torch.float32)
        # NCP = data["NCP"].to(torch.float32)
        # knf = data["knf"].to(torch.float32)
        # NCP = self.linear1(NCP)
        # knf = NCP + knf
        # DPCP = DPCP.transpose(1, 2)
        # NCP = NCP.transpose(1, 2)
        # # knf = knf.transpose(1, 2)
        #
        # x = knf.transpose(1, 2)
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # x = self.dropout1(x)
        #
        # # 将卷积层输出转为LSTM的输入 (batch_size, seq_length, nbfilter)
        # x = x.transpose(1, 2)
        # x, _ = self.lstm(x)
        #
        # # Flatten操作
        # x = x.contiguous().view(x.size(0), -1)
        #
        # x = F.relu(self.fc1(x))
        # x = self.dropout2(x)
        # x = self.fc2(x)
        #
        # # 使用softmax进行分类
        # x = F.softmax(x, dim=1)
        return x