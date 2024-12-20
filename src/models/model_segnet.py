import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SegNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SegNet, self).__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(ConvBlock(3, 64), ConvBlock(64, 64))
        self.encoder2 = nn.Sequential(ConvBlock(64, 128), ConvBlock(128, 128))
        self.encoder3 = nn.Sequential(ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256))
        self.encoder4 = nn.Sequential(ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.encoder5 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))

        # Decoder
        self.decoder5 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 512))
        self.decoder4 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512), ConvBlock(512, 256))
        self.decoder3 = nn.Sequential(ConvBlock(256, 256), ConvBlock(256, 256), ConvBlock(256, 128))
        self.decoder2 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 64))
        self.decoder1 = nn.Sequential(ConvBlock(64, 64), ConvBlock(64, num_classes))
        
        # Pooling and Unpooling
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        # Final activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoder
        x, ind1 = self.pool(self.encoder1(x))
        x, ind2 = self.pool(self.encoder2(x))
        x, ind3 = self.pool(self.encoder3(x))
        x, ind4 = self.pool(self.encoder4(x))
        x, ind5 = self.pool(self.encoder5(x))

        # Decoder
        x = self.unpool(x, ind5)
        x = self.decoder5(x)
        x = self.unpool(x, ind4)
        x = self.decoder4(x)
        x = self.unpool(x, ind3)
        x = self.decoder3(x)
        x = self.unpool(x, ind2)
        x = self.decoder2(x)
        x = self.unpool(x, ind1)
        x = self.decoder1(x)

        return self.softmax(x)

if __name__ == "__main__":
    # Instantiate and test
    model = SegNet(num_classes=2)
    input_tensor = torch.randn(1, 3, 224, 224)  # Example input
    output = model(input_tensor)
    print(output.shape)  # Should output (1, num_classes, 128, 128)
