# RRDB (Residual-in-Residual Dense Block) module

class RRDB(nn.Module):
    def __init__(self, channels, residual_scaling, init_variance):
        super(RRDB, self).__init__()
        self.res_dense_blocks = nn.ModuleList([ResidualDenseBlock(channels, init_variance) for _ in range(3)])
        self.residual_scaling = residual_scaling

    def forward(self, x):
        out = x
        for res_dense_block in self.res_dense_blocks:
            out = res_dense_block(out)
        return x + self.residual_scaling * out

# Residual Dense Block module

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, init_variance):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels * 4, channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels * 5, channels, kernel_size=3, stride=1, padding=1)

        self.init_weights(init_variance)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat([x, conv1], dim=1))
        conv3 = self.conv3(torch.cat([x, conv1, conv2], dim=1))
        conv4 = self.conv4(torch.cat([x, conv1, conv2, conv3], dim=1))
        out = self.conv5(torch.cat([x, conv1, conv2, conv3, conv4], dim=1))
        return out + x

    def init_weights(self, init_variance):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=init_variance)
                nn.init.constant_(m.bias, 0) 
