import torch
import torch.nn as nn


class ResUNetEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, base_ch=64):
        super(ResUNetEncoder, self).__init__()
        self.inc = self._Conv(in_ch, base_ch, repeat_time=2)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = self._Down(base_ch, base_ch * 2, repeat_time=2)
        self.down2 = self._Down(base_ch * 2, base_ch * 4, repeat_time=3)
        self.down3 = self._Down(base_ch * 4, base_ch * 8, repeat_time=3)
        self.down4 = self._Down(base_ch * 8, base_ch * 16, repeat_time=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ###将前面的参数固定
        # for p in self.parameters():
        #     p.requires_grad = False
        self.classifier_my = nn.Sequential(nn.Linear(1024, 512),
                                           nn.BatchNorm1d(512),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(512, 256),
                                           nn.BatchNorm1d(256),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(256, out_ch))

        # self.up1 = self._Up(base_ch * 16, base_ch * 8, repeat_time=3)
        # self.up2 = self._Up(base_ch * 8, base_ch * 4, repeat_time=3)
        # self.up3 = self._Up(base_ch * 4, base_ch * 2, repeat_time=3)
        # self.up4 = self._Up(base_ch * 2, base_ch, repeat_time=2)
        # self.outc = nn.Conv2d(base_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.avgpool(x5)
        x = torch.flatten(x, 1)
        x = self.classifier_my(x)

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # x = self.outc(x)
        return x

    class _Down(nn.Module):
        def __init__(self, in_ch, out_ch, repeat_time):
            super(ResUNetEncoder._Down, self).__init__()
            self.mp = nn.MaxPool2d(kernel_size=2)
            self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
            self.dc = ResUNetEncoder._Conv(out_ch, out_ch, repeat_time)

        def forward(self, x):
            x1 = self.mp(x)
            x1 = self.conv1x1(x1)
            x2 = self.dc(x1)
            return x1 + x2

    class _Up(nn.Module):
        def __init__(self, in_ch, out_ch, repeat_time):
            super(ResUNetEncoder._Up, self).__init__()
            self.up = ResUNetEncoder._UpConv(in_ch, out_ch)
            self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=True)
            self.conv = ResUNetEncoder._Conv(out_ch, out_ch, repeat_time)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            x3 = torch.cat([x2, x1], dim=1)
            x3 = self.conv1x1(x3)
            x4 = self.conv(x3)
            return x3 + x4

    class _Conv(nn.Module):
        def __init__(self, in_ch, out_ch, repeat_time):
            super(ResUNetEncoder._Conv, self).__init__()
            self.inc = self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.PReLU()
            )
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.PReLU()
            )
            self.repeat_time = repeat_time

        def forward(self, x):
            x = self.inc(x)
            for _ in range(self.repeat_time - 1):
                x = self.conv(x)
            return x

    class _UpConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(ResUNetEncoder._UpConv, self).__init__()
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                ResUNetEncoder._Conv(out_ch, out_ch, repeat_time=1)
            )

        def forward(self, x):
            x = self.up(x)
            return x


if __name__ == '__main__':
    from torchsummary import summary

    net = ResUNetEncoder(in_ch=1, out_ch=2).cuda()
    summary(net, (1, 112, 112))
