import torch
import torch.nn as nn

class MaskedConv2d(nn.Conv2d):
    """
    Implements a masked convolution.
    A masked convolution ensures that the model cannot see 'future' pixels
    when predicting a given pixel. This is essential for autoregressive models.
    """
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}, "Mask type must be 'A' or 'B'"
        self.register_buffer('mask', self.weight.data.clone())

        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        # Mask the second half of the filters' height
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class ResidualBlock(nn.Module):
    """
    A simple residual block to help with training deeper networks.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels // 2, 1), # Bottleneck
            nn.ReLU(),
            MaskedConv2d('B', in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1) # Expand
        )

    def forward(self, x):
        return x + self.net(x)

class PixelCNN(nn.Module):
    """
    The main PixelCNN model.
    The output is a tensor of size (B, C * 256, H, W) because for each pixel and
    each channel (R, G, B), we predict one of 256 possible values.
    """
    def __init__(self, in_channels=3, n_filters=128, n_blocks=7, output_bins=256):
        super(PixelCNN, self).__init__()
        self.output_bins = output_bins
        self.in_channels = in_channels

        self.net = nn.Sequential(
            # Mask A for the first layer to prevent self-connection
            MaskedConv2d('A', in_channels, n_filters, kernel_size=7, padding=3),
            nn.ReLU(),
            # Sequence of residual blocks with Mask B
            *[ResidualBlock(n_filters) for _ in range(n_blocks)],
            nn.ReLU(),
            nn.Conv2d(n_filters, 1024, 1),
            nn.ReLU(),
            # Output convolution
            nn.Conv2d(1024, in_channels * output_bins, 1)
        )

    def forward(self, x):
        out = self.net(x)
        # Reshape the output to be (B, C, bins, H, W) and then (B, bins, C, H, W)
        # This is for compatibility with CrossEntropyLoss
        B, C_bins, H, W = out.shape
        out = out.view(B, self.in_channels, self.output_bins, H, W)
        out = out.permute(0, 2, 1, 3, 4)
        return out

    def sample(self, occluded_img):
        """
        Generates the missing pixels in an image autoregressively.
        This is slow and done pixel-by-pixel.
        """
        self.eval()
        with torch.no_grad():
            img_copy = occluded_img.clone()
            # Identify the occluded region (assuming it's white: 1.0)
            is_occluded = (occluded_img.sum(dim=1) >= 2.9).float() # Sum R,G,B > 2.9 (close to 3)

            _, _, height, width = occluded_img.shape
            for i in range(height):
                for j in range(width):
                    if is_occluded[0, i, j] > 0: # Check if this pixel is occluded
                        # Get model prediction for the entire image so far
                        out = self.forward(img_copy) # (B, bins, C, H, W)
                        
                        # Get the probabilities for the current pixel
                        probs = torch.softmax(out[:, :, :, i, j], dim=1)
                        
                        # Sample from the distribution for each channel
                        for c in range(self.in_channels):
                            pixel_val = torch.multinomial(probs[0, :, c], 1).float() / (self.output_bins - 1)
                            img_copy[0, c, i, j] = pixel_val
        return img_copy