import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device('cuda:0')
VALID_ACTIVATION = [a for a in dir(nn.modules.activation)
                    if not a.startswith('__')
                    and a not in ['torch', 'warnings', 'F', 'Parameter', 'Module']]
VALID_BATCHNORM_DIM = {1, 2, 3}


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation, dropout):
        super(MLP, self).__init__()

        if type(activation) == str:
            intermediate_activation = getattr(nn, activation)
            final_activation = getattr(nn, activation)
        else:
            intermediate_activation = getattr(nn, activation[0])
            if activation[1] is None:
                final_activation = None
            else:
                final_activation = getattr(nn, activation[1])

        layer_size = [input_size] + hidden_size + [output_size]
        layers = []
        for i in range(len(layer_size) - 1):
            layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if dropout != 0:
                layers.append(nn.Dropout(dropout))
            if i != len(layer_size) - 2:
                layers.append(intermediate_activation())
            elif final_activation is not None:
                layers.append(final_activation())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)

        return output


class StackRNN(nn.Module):
    def __init__(self, cell, input_size, hidden_size, num_layers, batch_first, dropout,
                 bidirectional):
        super(StackRNN, self).__init__()

        self.cell = cell

        rnn = getattr(nn, cell)
        num_directions = 2 if bidirectional else 1

        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            self.rnns.append(rnn(
                input_size=(input_size if i == 0 else hidden_size * num_directions),
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            ))

    def forward(self, x):
        outputs = [x]
        for i in range(len(self.rnns)):
            rnn_input = outputs[-1]
            output = self.rnns[i](rnn_input)
            outputs.append(output if self.cell == 'RNN' else output[0])

        return torch.cat(outputs, dim=2)


class Activation(nn.Module):
    def __init__(self, activation, *args, **kwargs):
        super(Activation, self).__init__()

        if activation in VALID_ACTIVATION:
            self.activation = \
                getattr(nn.modules.activation, activation)(*args, **kwargs)
        else:
            raise ValueError(
                f'Activation: {activation} is not a valid activation function')

    def forward(self, x):
        return self.activation(x)


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super(HighwayNetwork, self).__init__()

        self.H = nn.Linear(size, size)
        self.T = nn.Linear(size, size)
        self.activation = Activation('ReLU')

    def forward(self, x):
        T = self.activation(self.T(x))
        H = torch.sigmoid(self.H(x))
        C = 1 - H

        return H * T + x * C


class DepthwiseSeperableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeperableConv1d, self).__init__()

        self.depthwise_conv1d = nn.Conv1d(
            in_channels, in_channels, kernel_size, groups=in_channels,
            padding=kernel_size // 2)
        self.pointwise_conv1d = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise_conv1d(x)
        x = self.pointwise_conv1d(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, output_size, n_heads, max_pos_distance, dropout,
                 pos_embedding_K, pos_embedding_V):
        super(MultiHeadSelfAttention, self).__init__()

        if output_size % n_heads != 0:
            raise ValueError(
                f'MultiHeadSelfAttention: output_size({output_size}) isn\'t'
                f'a multiplier of n_heads({n_heads})')

        self.output_size = output_size
        self.n_heads = n_heads
        self.d_head = output_size // n_heads
        self.max_pos_distance = max_pos_distance

        self.pos_embedding_K = pos_embedding_K
        self.pos_embedding_V = pos_embedding_V

        self.input_linear = nn.Linear(input_size, output_size * 3)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.output_linear = nn.Linear(output_size, output_size)

    def forward(self, x, mask):
        batch_size, input_len, *_ = x.shape

        Q, K, V = self.input_linear(x).chunk(3, dim=-1)
        Q, K, V = [
            x.reshape(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            for x in (Q, K, V)
        ]

        pos_index = torch.arange(input_len).reshape(1, -1).repeat(input_len, 1)
        pos_index = pos_index - pos_index.t()
        pos_index = pos_index.clamp(-self.max_pos_distance, self.max_pos_distance)
        pos_index += self.max_pos_distance
        pos_index = pos_index.to(dtype=torch.int64, device=DEVICE)

        # TODO?: add dropout to position embedding
        # calculate attention score (relative position representation #1)
        S1 = Q @ K.transpose(-1, -2)
        Q = Q.reshape(-1, input_len, self.d_head).transpose(0, 1)
        pos_emb_K = self.pos_embedding_K(pos_index)
        S2 = (Q @ pos_emb_K.transpose(-1, -2)).transpose(0, 1)
        S2 = S2.reshape(batch_size, self.n_heads, input_len, input_len)
        S = (S1 + S2) / np.sqrt(self.d_head)

        # set score of V padding tokens to 0
        S.masked_fill_(mask.reshape(batch_size, 1, 1, -1) == 0, float('-inf'))
        A = F.softmax(S, dim=-1)
        if self.dropout:
            A = self.dropout(A)

        # apply attention to get output (relative position representation #2)
        O1 = A @ V
        A = A.reshape(-1, input_len, input_len).transpose(0, 1)
        pos_emb_V = self.pos_embedding_V(pos_index)
        O2 = (A @ pos_emb_V).transpose(0, 1)
        O2 = O2.reshape(batch_size, self.n_heads, input_len, self.d_head)
        output = O1 + O2
        output = output.transpose(1, 2).reshape(batch_size, -1, self.output_size)
        output = self.output_linear(output)

        return output


class PointwiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PointwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation = Activation('ReLU')
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.linear2(x)

        return x


class BatchNormResidual(nn.Module):
    def __init__(self, sublayer, n_features, dim=1, transpose=False, activation=None,
                 dropout=0):
        super(BatchNormResidual, self).__init__()

        self.sublayer = sublayer
        if dim in VALID_BATCHNORM_DIM:
            batch_norm = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]
            self.batch_norm = batch_norm[dim - 1](n_features)
        else:
            raise ValueError(
                f'BatchNormResidual: dim must be one of {{1, 2, 3}}, but got {dim}')
        self.transpose = transpose
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, x, *args, **kwargs):
        y = self.sublayer(x, *args, **kwargs)

        if self.transpose:
            y = y.transpose(1, 2).contiguous()
        y = self.batch_norm(y)
        if self.transpose:
            y = y.transpose(1, 2)
        y += x

        if self.activation:
            y = self.activation(y)
        if self.dropout:
            y = self.dropout(y)

        return y


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_convs, kernel_size, n_heads,
                 max_pos_distance, dropout, pos_embedding_K, pos_embedding_V):
        super(EncoderBlock, self).__init__()

        self.convs = nn.ModuleList([
            BatchNormResidual(
                DepthwiseSeperableConv1d(d_model, d_model, kernel_size), d_model,
                activation=Activation('ReLU'), dropout=dropout)
            for _ in range(n_convs)
        ])
        self.attention = BatchNormResidual(
            MultiHeadSelfAttention(
                d_model, d_model, n_heads, max_pos_distance, dropout, pos_embedding_K,
                pos_embedding_V),
            d_model, transpose=True, dropout=dropout)
        self.feedforward = BatchNormResidual(
            PointwiseFeedForward(d_model, d_model * 4, dropout), d_model,
            transpose=True, activation=Activation('ReLU'), dropout=dropout)

    def forward(self, x, x_pad_mask):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)

        x = self.attention(x, x_pad_mask)
        x = self.feedforward(x)

        return x


class Encoder(nn.Module):
    def __init__(self, n_blocks, input_size, d_model, n_convs, kernel_size,
                 n_heads, max_pos_distance, dropout):
        super(Encoder, self).__init__()

        self.conv = DepthwiseSeperableConv1d(input_size, d_model, kernel_size)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = Activation('ReLU')
        self.dropout = nn.Dropout(p=dropout) if dropout else None

        pos_embedding_K = nn.Embedding(2 * max_pos_distance + 1, d_model // n_heads)
        pos_embedding_V = nn.Embedding(2 * max_pos_distance + 1, d_model // n_heads)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                d_model, n_convs, kernel_size, n_heads, max_pos_distance, dropout,
                pos_embedding_K, pos_embedding_V
            )
            for i in range(n_blocks)
        ])

    def forward(self, x, x_pad_mask):
        x = x.transpose(1, 2)
        x = self.batch_norm(self.conv(x))
        x = self.activation(x)
        if self.dropout:
            x = self.dropout(x)
        x = x.transpose(1, 2)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, x_pad_mask)

        return x
