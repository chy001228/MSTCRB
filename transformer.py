import math

import torch
from torch import nn



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        """
        Layer Normalization 初始化函数

        :param d_model: 输入张量的最后一个维度大小（通常为模型的隐藏单元数）
        :param eps: 用于数值稳定性的小值，默认为 1e-12
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))  # 可学习的缩放参数
        self.beta = nn.Parameter(torch.zeros(d_model))  # 可学习的偏置参数
        self.eps = eps  # 用于数值稳定性的小值

    def forward(self, x):
        """
        Layer Normalization 的前向传播函数

        :param x: 输入张量，shape [batch_size, seq_len, d_model]
        :return: 归一化后的张量，shape [batch_size, seq_len, d_model]
        """
        mean = x.mean(-1, keepdim=True)  # 沿着最后一个维度计算均值，保持维度
        var = x.var(-1, unbiased=False, keepdim=True)  # 沿着最后一个维度计算方差，保持维度

        # 计算 Layer Normalization
        out = (x - mean) / torch.sqrt(var + self.eps)  # 归一化
        out = self.gamma * out + self.beta  # 缩放和平移

        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
               编码器层初始化函数

               :param d_model: 模型的维度大小
               :param ffn_hidden: 位置前馈神经网络中隐藏层的维度大小
               :param n_head: 多头注意力机制中注意头的数量
               :param drop_prob: Dropout 概率
        """
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask=None):
        """
                编码器层的前向传播函数

                :param x: 输入张量，shape [batch_size, seq_len, d_model]
                :param src_mask: 源序列掩码张量，用于屏蔽无效位置
                :return: 编码器层的输出张量，shape [batch_size, seq_len, d_model]
        """
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x

class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        """
                编码器初始化函数
                :param d_model: 模型的维度大小
                :param ffn_hidden: 位置前馈神经网络中隐藏层的维度大小
                :param n_head: 多头注意力机制中注意头的数量
                :param n_layers: 编码器层的堆叠层数
                :param drop_prob: Dropout 概率
        """
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask=None):
        """
               编码器的前向传播函数

               :param x: 输入序列张量，shape [batch_size, seq_len]
               :param src_mask: 源序列掩码张量，用于屏蔽无效位置，shape [batch_size, 1, seq_len]
               :return: 编码器的输出张量，shape [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        """
               解码器层初始化函数

               :param d_model: 模型的维度大小
               :param ffn_hidden: 位置前馈神经网络中隐藏层的维度大小
               :param n_head: 多头注意力机制中注意头的数量
               :param drop_prob: Dropout 概率
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask=None, src_mask=None):
        """
               解码器层的前向传播函数

               :param dec: 解码器输入张量，shape [batch_size, trg_seq_len, d_model]
               :param enc: 编码器输出张量，shape [batch_size, src_seq_len, d_model]
               :param trg_mask: 目标序列掩码张量，用于屏蔽无效位置，shape [batch_size, 1, trg_seq_len]
               :param src_mask: 源序列掩码张量，用于屏蔽无效位置，shape [batch_size, 1, src_seq_len]
               :return: 解码器层的输出张量，shape [batch_size, trg_seq_len, d_model]
        """
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        """
                解码器初始化函数

                :param dec_voc_size: 输出词汇表的大小（词汇表中不同词的数量）
                :param d_model: 模型的维度大小
                :param ffn_hidden: 位置前馈神经网络中隐藏层的维度大小
                :param n_head: 多头注意力机制中注意头的数量
                :param n_layers: 解码器层的堆叠层数
                :param drop_prob: Dropout 概率
        """
        super().__init__()

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        """
                解码器的前向传播函数

                :param trg: 目标序列张量，shape [batch_size, trg_seq_len]
                :param src: 编码器输出张量，shape [batch_size, src_seq_len, d_model]
                :param trg_mask: 目标序列掩码张量，用于屏蔽无效位置，shape [batch_size, 1, trg_seq_len]
                :param src_mask: 源序列掩码张量，用于屏蔽无效位置，shape [batch_size, 1, src_seq_len]
                :return: 解码器的输出张量，shape [batch_size, trg_seq_len, dec_voc_size]
        """

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)
        return output
