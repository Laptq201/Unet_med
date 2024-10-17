import torch
import torch.nn as nn
import torch.nn.functional as F


class convolutionBlock(nn.Module):
    # set up with kernel size = 3, padding = true, stride = 1
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class recurrentBlock(nn.Module):
    """
    2x conv3D -> relu
    input conv3D: 4D (depth, height, width, channels) -> BraTS here
                  5D (batch, depth, height, width, channels)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class AttentionBlock(nn.Module):
    """
    Attention Block
    """

    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()

        self.theta = nn.Conv3d(
            encoder_channels, encoder_channels//2, kernel_size=1)
        self.phi = nn.Conv3d(
            decoder_channels, encoder_channels//2, kernel_size=1)
        self.psi = nn.Conv3d(
            decoder_channels, encoder_channels//2, kernel_size=1)
        self.out_conv = nn.Conv3d(
            encoder_channels, encoder_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, encoder_x, decoder_x):
        theta = self.theta(encoder_x)
        phi = self.phi(decoder_x)
        psi = self.psi(encoder_x)
        attention = self.relu(theta + phi + psi)
        attention = self.out_conv(attention)
        attention = self.sigmoid(attention)
        return encoder_x * attention.expand_as(encoder_x) + decoder_x


class AR2B_UNet(nn.Module):
    """
    Implement 3D Unet model with attention block and recurrent block

    1. Encoder:
    - Input đi vào các convBlocks (convBlock = 2x [conv3D -> BatchNorm3d -> relu])
    - Sau mỗi convBlock, downsample bằng MaxPool3d

    2. Recurrent Block:
    - Đầu vào của recurrent block là output của encoder, bao gồm 2 conv layers
        (NOTE: 2 conv layers này không downsample, chỉ làm feature extraction
                để giữ temporal information)

    3. Decoder:
    - Output của recurrent block = upsample bằng ConvTranspose3d
    - Concatenate với output của encoder (cùng level)
        +) Đưa qua attention block để nhấn mạnh feature encode quan trọng từ encodere
        +) Đầu ra của attention block sẽ được đưa qua convBlocks.

    4. Final Convolution:
    - Lớp 1x1 Conv3D cuối cùng sẽ tạo ra một pixel-wise predictions.
        NOTE: "pixel-wise" predictions ám chỉ rằng mạng lưới sẽ tạo ra đầu ra cho 
            từng voxel riêng lẻ, thay vì cho toàn bộ khối dữ liệu cùng một lúc.

    Args:
    - in_channels: số lượng channels của input
    - num_classes: số lượng classes cần phân loại

    Output:
    - A PyTorch model   
    """

    def __init__(self, in_channels, num_classes):
        super(AR2B_UNet, self).__init__()

        # Encoder
        self.encoder1 = convolutionBlock(in_channels, 64)
        self.encoder2 = convolutionBlock(64, 64*2)
        self.encoder3 = convolutionBlock(128, 256)
        self.encoder4 = convolutionBlock(256, 512)
        self.encoder5 = convolutionBlock(512, 1024)
        # Recurrent Block
        self.recurrent = recurrentBlock(1024, 256)

        # Decoder: Attention Block + Convolution Block + ConvTranspose3d

        # Attention Block
        self.attention1 = AttentionBlock(512, 512)
        self.attention2 = AttentionBlock(256, 256)
        self.attention3 = AttentionBlock(128, 128)
        self.attention4 = AttentionBlock(64, 64)

        # Convolution Block
        self.decoder1 = convolutionBlock(512*2, 512)
        self.decoder2 = convolutionBlock(256*2, 256)
        self.decoder3 = convolutionBlock(128*2, 128)
        self.decoder4 = convolutionBlock(64*2, 64)

        # ConvTranspose3d
        self.upconv1 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Final Convolution
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder flow
        encoder1 = self.encoder1(x)
        x = F.max_pool3d(encoder1, 2)
        encoder2 = self.encoder2(x)
        x = F.max_pool3d(encoder2, 2)
        encoder3 = self.encoder3(x)
        x = F.max_pool3d(encoder3, 2)
        encoder4 = self.encoder4(x)
        x = F.max_pool3d(encoder4, 2)
        x = self.encoder5(x)
        # Recurrent Block
        x = self.recurrent(x)

        # Decoder flow
        x = self.upconv1(x)
        x = torch.cat([x, self.attention1(encoder4, x)], dim=1)
        x = self.decoder1(x)

        x = self.upconv2(x)
        x = torch.cat([x, self.attention2(encoder3, x)], dim=1)
        x = self.decoder2(x)

        x = self.upconv3(x)
        x = torch.cat([x, self.attention3(encoder2, x)], dim=1)
        x = self.decoder3(x)

        x = self.upconv4(x)
        x = torch.cat([x, self.attention4(encoder1, x)], dim=1)
        x = self.decoder4(x)

        # Final Convolution
        x = self.final_conv(x)
        return x
