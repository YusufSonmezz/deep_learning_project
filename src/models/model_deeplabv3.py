import torch
import torch.nn as nn
import torchvision.models as models

class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3, self).__init__()

        # Backbone: Pretrained ResNet as the feature extractor
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]  # Remove fully connected layers and average pooling
        )

        # Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.backbone(x)  # Extract features
        x = self.aspp(x)      # ASPP module
        x = self.classifier(x)  # Classifier
        x = self.upsample(x)    # Upsample to original resolution
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_block2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.atrous_block3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_avg_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]  # Save the spatial dimensions

        conv1x1 = self.conv1x1(x)
        atrous1 = self.atrous_block1(x)
        atrous2 = self.atrous_block2(x)
        atrous3 = self.atrous_block3(x)

        global_avg = self.global_avg_pool(x)
        global_avg = self.global_avg_conv(global_avg)
        global_avg = nn.functional.interpolate(global_avg, size=size, mode='bilinear', align_corners=False)

        x = torch.cat([conv1x1, atrous1, atrous2, atrous3, global_avg], dim=1)
        x = self.final_conv(x)
        return x


# Example usage
if __name__ == "__main__":
    num_classes = 2  # For example, VOC dataset
    model = DeepLabV3(num_classes=num_classes)

    model.eval()

    # Input tensor
    input_tensor = torch.randn(1, 3, 256, 256)  # Batch size 1, 3 channels, 256x256 image
    output = model(input_tensor)

    print("Output shape:", output.shape)  # Expected: (1, num_classes, 256, 256)
