from torch import nn

from mcd.networks.resnet import modules


class ResBase(nn.Module):
    def __init__(self, num_classes, layer='50', input_ch=3):
        super(ResBase, self).__init__()
        self.num_classes = num_classes

        print(f'resnet layer: {layer}')
        if layer == '18':
            resnet = modules.resnet18(pretrained=True, input_ch=input_ch)
        elif layer == '50':
            resnet = modules.resnet50(pretrained=True, input_ch=input_ch)
        elif layer == '101':
            resnet = modules.resnet101(pretrained=True, input_ch=input_ch)
        elif layer == '152':
            resnet = modules.resnet152(pretrained=True, input_ch=input_ch)
        else:
            raise ValueError(f"NOT Support layer: {layer}")

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        img_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        out_dic = {
            "img_size": img_size,
            "conv_x": conv_x,
            "pool_x": pool_x,
            "fm2": fm2,
            "fm3": fm3,
            "fm4": fm4
        }

        return out_dic


class ResClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResClassifier, self).__init__()

        self.num_classes = num_classes
        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)
        self.out5 = self._classifier(32)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes / 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes / 2),
            nn.ReLU(inplace=True),
            # nn.Dropout(.1),
            nn.Conv2d(inplanes / 2, self.num_classes, 1),
        )

    def forward(self, gen_out_dic):
        gen_out_dic = edict(gen_out_dic)
        fsfm1 = self.fs1(gen_out_dic.fm3, self.upsample1(gen_out_dic.fm4, gen_out_dic.fm3.size()[2:]))
        fsfm2 = self.fs2(gen_out_dic.fm2, self.upsample2(fsfm1, gen_out_dic.fm2.size()[2:]))
        fsfm3 = self.fs4(gen_out_dic.pool_x, self.upsample3(fsfm2, gen_out_dic.pool_x.size()[2:]))
        fsfm4 = self.fs5(gen_out_dic.conv_x, self.upsample4(fsfm3, gen_out_dic.conv_x.size()[2:]))
        fsfm5 = self.upsample5(fsfm4, gen_out_dic.img_size)
        out = self.out5(fsfm5)
        return out
