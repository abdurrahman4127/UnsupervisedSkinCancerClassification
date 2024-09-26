 # ESRGAN Generator Network

class ESRGANGenerator(nn.Module):
    def __init__(self, num_rrdb=28, residual_scaling=0.15, init_variance=0.03):
        super(ESRGANGenerator, self).__init__()
        self.num_rrdb = num_rrdb
        self.residual_scaling = residual_scaling
        self.init_variance = init_variance

        self.conv_in = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.rrdb_blocks = nn.ModuleList([RRDB(64, self.residual_scaling, self.init_variance) for _ in range(self.num_rrdb)])
        self.upscale1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upscale2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv_in(x)
        residual = out

        for rrdb in self.rrdb_blocks:
            out = rrdb(out)

        out = self.upscale1(out)
        out = self.upscale2(out)
        out = self.conv_out(out)

        return out + residual
