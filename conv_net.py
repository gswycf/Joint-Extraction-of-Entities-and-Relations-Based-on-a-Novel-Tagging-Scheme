import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, residual=True):
        super(ConvBlock, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.activate = nn.ReLU()
        self.residual = residual
        self.down_sample = nn.Conv1d(in_channels, out_channels, 1) if residual and in_channels != out_channels else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight.data, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0)
        if self.down_sample is not None:
            nn.init.kaiming_uniform_(self.down_sample.weight.data, mode='fan_in', nonlinearity='relu')
            if self.down_sample.bias is not None:
                self.down_sample.bias.data.fill_(0)

    def forward(self, inputs):
        output = self.activate(self.conv(inputs))
        if self.residual:
            output += self.down_sample(inputs) if self.down_sample else inputs
        return output


class ConvNet(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.5, dilated=False, residual=True):
        super(ConvNet, self).__init__()
        num_levels = len(channels)-1
        layers = []
        for i in range(num_levels):
            in_channels = channels[i]
            out_channels = channels[i+1]
            dilation = kernel_size ** i if dilated else 1
            padding = (kernel_size - 1) // 2 * dilation
            layers += [ConvBlock(in_channels, out_channels, kernel_size,
                                 padding=padding, dilation=dilation, residual=residual), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, inputs):
        return self.net(inputs)
'''
Sequential(
  (0): ConvBlock(
    (conv): Conv1d(200, 200, kernel_size=(3,), stride=(1,), padding=(1,))
    (activate): ReLU()
  )
)
con=ConvNet([200,200])
print(con.net)
'''
