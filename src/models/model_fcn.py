import torch
import torch.nn as nn
import torchvision.models as models

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        
        # Pretrained ResNet as the backbone
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.base_layers = list(resnet.children())
        
        # Encoder
        self.enc1 = nn.Sequential(*self.base_layers[:3])  # First few layers
        self.enc2 = nn.Sequential(*self.base_layers[3:5]) # Downsample 1
        self.enc3 = self.base_layers[5]                  # Downsample 2
        self.enc4 = self.base_layers[6]                  # Downsample 3
        self.enc5 = self.base_layers[7]                  # Downsample 4

        # Fully convolutional classifier
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

        # Upsampling layers
        self.upsample4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upsample1 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2)

    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        # Classifier
        out = self.classifier(enc5)

        # Upsampling
        out = self.upsample4(out)  # Upsample 1
        out = self.upsample3(out)  # Upsample 2
        out = self.upsample2(out)  # Upsample 3
        out = self.upsample1(out)  # Upsample 4

        return out

# Example usage
if __name__ == "__main__":
    num_classes = 2  # For example, VOC dataset
    model = FCN(num_classes=num_classes)
    
    # Input tensor
    input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 channels, 256x256 image
    output = model(input_tensor)
    
    print("Output shape:", output.shape)  # Expected: (1, num_classes, 256, 256)
