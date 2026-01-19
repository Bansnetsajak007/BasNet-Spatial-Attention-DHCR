import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels,attn_channels=64):
        super().__init__()

        self.query_conv = nn.Conv2d(in_channels, attn_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, attn_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1) # Added value_conv which was missing and caused `value_conv` not defined error

        #learnable scaling parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):

        """
        x: [B × 256 × 8 × 8]  input feature map from stage 3
        return: out: [B × 256 × 8 × 8] attention value
        """
        B,C,H,W = x.shape  # getting batchsize, channel , height and weight from x --> input from stage 3

        N = H*W

        # Q,K,V projection
        Q = self.query_conv(x)  # [B × 64 × 8 × 8]
        K = self.key_conv(x)    # [B × 64 × 8 × 8]
        V = self.value_conv(x)  # [B × 256 × 8 × 8]

        # Flatten spatial dimensions
        Q = Q.view(B, -1, N).permute(0, 2, 1)  # [B × 64 × 64] Transpose of Q
        K = K.view(B, -1, N)                   # [B × 64 × 64]
        V = V.view(B, -1, N).permute(0, 2, 1)  # [B × 64 × 256]

        # Attention matrix
        attention = torch.bmm(Q, K)            # [B × 64 × 64]
        attention = F.softmax(attention, dim=-1)

        # Apply attention to V
        out = torch.bmm(attention, V)          # [B × 64 × 256]
        # Reshape back
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(B, C, H, W)              # [B × 256 × 8 × 8]

        #  Residual fusion
        out = self.gamma * out + x

        return out



class Basnet(nn.Module):
    def __init__(self, num_classes=46):
        super().__init__()
        self.attention = SpatialSelfAttention(in_channels=256) # Instantiate SpatialSelfAttention

        #  Stem (Bridge between input image and residual block )  just increases the feature map
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        #  Residual Stages
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),   #64x32x32
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128,2),
            ResidualBlock(128, 128),   #128x16x16
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256,2),
            ResidualBlock(256, 256),  #256x8x8
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(256, 512,2),

        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        #  Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.attention(x) #at 8x8 feature map
        x = self.stage4(x)  # [B,512,4,4]
        x = self.gap(x)    # Global Average Pooling to [B,512,1,1]
        x = x.view(x.size(0), -1)  # Flatten the tensor [B,512]
        x = self.classifier(x)
        return x

