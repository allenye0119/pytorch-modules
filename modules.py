import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation, dropout):
        super(MLP, self).__init__()

        if type(activation) == str:
            intermediate_activation = getattr(nn, activation)
            final_activation = getattr(nn, activation)
        else:
            intermediate_activation = getattr(nn, activation[0])
            if activation[1] == None:
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
            elif final_activation != None:
                layers.append(final_activation())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        output = self.mlp(x)

        return output

class StackRNN(nn.Module):
    def __init__(self, cell, input_size, hidden_size, num_layers, batch_first, dropout, bidirectional):
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
