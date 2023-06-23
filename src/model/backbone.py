import torch
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck


class PyramidResNet(ResNet):
    def __init__(self):
        super().__init__(Bottleneck, [3, 4, 6, 3])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x2, x3, x4


if __name__ == '__main__':
    img = torch.randint(100, size=(1, 3, 64, 64), dtype=torch.float32)

    backbone = PyramidResNet()
    print(backbone(img))
