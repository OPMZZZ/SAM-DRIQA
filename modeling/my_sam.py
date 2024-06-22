import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models as models


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc =torch.nn.Identity()
        self.adpt = nn.AdaptiveAvgPool1d(256)

        self.fc0 = nn.Linear(2048, 4*11)
        self.fc1 = nn.Linear(2048, 4*11)
        self.fc2 = nn.Linear(2048, 8*11)
        self.sigmoid = nn.Sigmoid()

    def forward(self, in_x0, in_x1):
        in_x0 = F.interpolate(in_x0, size=(640, 640), mode='bilinear', align_corners=False)
        in_x1 = F.interpolate(in_x1, size=(640, 640), mode='bilinear', align_corners=False)
        x0 = self.resnet50(in_x0)
        x0 = self.fc0(x0)
        x1 = self.resnet50(in_x1)
        x1 = self.fc1(x1)
        x = self.resnet50(in_x0+in_x1)
        x = self.fc2(x)
        x = torch.cat((x0, x1, x), dim=1)
        x = self.sigmoid(x)  # 111

        return x.view(x.shape[0], 16, 11)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(InceptionBlock, self).__init__()

        self.branch1 = nn.Conv2d(in_channels, out_channels1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2[0], kernel_size=1),
            nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3[0], kernel_size=1),
            nn.Conv2d(out_channels3[0], out_channels3[1], kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels4, kernel_size=1)
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        return torch.cat(outputs, dim=1)
