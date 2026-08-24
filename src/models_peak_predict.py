import torch
import torch.nn as nn

class DenseDetector(nn.Module):
    def __init__(self, n_channels_in=1, base_layers=None):
        super().__init__()

        # n features
        self.n_channels_in = n_channels_in

        if base_layers is None:
            raise ValueError("base_layers must be provided (e.g. config.BASE_LAYERS)")

        layers = []
        in_ch = n_channels_in

        print('FCN layer spec')
        for spec in base_layers:
            out_ch = spec["out_channels"]
            k = spec["kernel_size"]
            s = spec["stride"]
            padding = k // 2  # "same"-style padding for odd kernels
            print(spec)
            layers.append(nn.Conv1d(in_ch, out_ch, k, stride=s, padding=padding))
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 4, kernel_size=1)  # presence, offset, amplitude, gamma

    def forward(self, x):
        # input
        #PyTorch conv1d expect inputs in the shape (batch_size, channels, length) -> x.unsqueeze
        if self.n_channels_in == 1 and x.dim() == 2:
            h = self.conv(x.unsqueeze(1)) 
        else:
            # if n_channels_in > 1
            h = self.conv(x) # no squeeze if more than 1 channel

        # The permute swaps the dimensions to (batch_size, channels=4, length) -> (batch_size, length, 4),
        # easier to extract the 4 predictions for each step in the sequence. i.e out[ ... , 0]
        # out[:,:,0] = out[...,0]
        out = self.head(h).permute(0,2,1)
        return (torch.sigmoid(out[...,0]),  # presence
                torch.tanh(out[...,1]),     # offset  tanh runs -1 to 1
                torch.sigmoid(out[...,2]),  # amplitude
                torch.sigmoid(out[...,3]))  # gamma
