import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

class Base(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        layers = [*self.add_blocks, *self.box, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def bbox_view(self, src, box, conf):
        ret = []
        for s, l, c in zip(src, box, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.num_classes, -1)))

        boxs, confs = list(zip(*ret))
        boxs, confs = torch.cat(boxs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return boxs, confs


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(pretrained=True)
        self.out_channels = [1024, 512, 512, 256, 256, 256]
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class SSD(Base):
    def __init__(self, backbone=ResNet(), num_classes=4):
        super().__init__()

        self.feature_extractor = backbone
        self.num_classes = num_classes
        self._build_additional_features(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.box = []
        self.conf = []

        for num_df, out_ch in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.box.append(nn.Conv2d(out_ch, num_df * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(out_ch, num_df * self.num_classes, kernel_size=3, padding=1))

        self.box = nn.ModuleList(self.box)
        self.conf = nn.ModuleList(self.conf)
        self.init_weights()

    def _build_additional_features(self, input_size):
        self.add_blocks = []
        for i, (input_size, output_size, channels) in enumerate(
                zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.add_blocks.append(layer)

        self.add_blocks = nn.ModuleList(self.add_blocks)

    def forward(self, x):
        x = self.feature_extractor(x)
        detection_feed = [x]
        for l in self.add_blocks:
            x = l(x)
            detection_feed.append(x)
        boxs, confs = self.bbox_view(detection_feed, self.box, self.conf)
        return boxs, confs
