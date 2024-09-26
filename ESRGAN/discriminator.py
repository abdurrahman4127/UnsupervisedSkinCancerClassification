 # ESRGAN Discriminator Network

class ESRGANDiscriminator(nn.Module):
    def __init__(self, num_conv_layers=14):
        super(ESRGANDiscriminator, self).__init__()
        self.num_conv_layers = num_conv_layers

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            *[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) for _ in range(self.num_conv_layers - 1)],
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        ])

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = x
        for conv_layer in self.conv_layers:
            out = conv_layer(out)
            out = self.leaky_relu(out)
        return out
