import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class WaveletEdge(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.edge_conv = nn.Parameter(torch.randn(out_channels, in_channels, 3, 3))
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.edge_conv.data.copy_(sobel_x)

    def forward(self, x):
        edges = []
        B, C, H, W = x.shape
        for d in self.scales:
            pad = d
            edge_map = F.conv2d(x, self.edge_conv, padding=pad, dilation=d)
            edge_map = torch.sigmoid(torch.abs(edge_map))
            edges.append(edge_map)
        return edges  # List of [B, 1, H, W], same spatial size

class AtrousWavelet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.edge_extractor = DifferentiableWaveletEdge(in_channels=in_channels, scales=[1, 2, 4])

    def forward(self, x):
        edges = self.edge_extractor(x)
        return edges[0], edges[1], edges[2]  # (e1, e2, e3)

class Encoder(nn.Module):

    def __init__(self, in_channels=1, latent_dim=32, dim=64): # Adjusted dims based on paper's latent space (64x64x32)
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.dim = dim

        # ATW module for edge extraction
        self.atw = AtrousWavelet(in_channels=in_channels)

        self.edge_adapter1 = nn.Conv2d(1, dim, kernel_size=1) # Adapts edge map from ATW scale 1
        self.edge_adapter2 = nn.Conv2d(1, dim * 2, kernel_size=1) # Adapts edge map from ATW scale 2
        self.edge_adapter3 = nn.Conv2d(1, dim * 4, kernel_size=1) # Adapts edge map from ATW scale 3


        self.down_block1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=4, stride=2, padding=1), # 512->256
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.down_block2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1), # 256->128
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )
        self.down_block3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1), # 128->64
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )
        self.down_block4 = nn.Sequential( # Final downsampling to latent space
            nn.Conv2d(dim * 4, latent_dim, kernel_size=4, stride=2, padding=1), # 64->32
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )

        self.down_block4 = nn.Sequential(
            nn.Conv2d(dim * 4, latent_dim, kernel_size=3, stride=1, padding=1), # 64->64 (no spatial down)
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        edge_map1, edge_map2, edge_map3 = self.atw(x) # e.g., (B, 1, H, W), (B, 1, H//2, W//2), (B, 1, H//4, W//4)


        x = self.down_block1(x) # Shape: (B, dim, H//2, W//2)

        adapted_edge1 = self.edge_adapter1(F.interpolate(edge_map1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False))
        x = x + adapted_edge1 # Element-wise addition or concatenation could be used

        # Block 2: Process x, inject edge_map2
        x = self.down_block2(x) # Shape: (B, dim*2, H//4, W//4)
        adapted_edge2 = self.edge_adapter2(F.interpolate(edge_map2, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False))
        x = x + adapted_edge2


        x = self.down_block3(x) # Shape: (B, dim*4, H//8, W//8)
        # Upsample edge_map3 to match x's spatial size (H//8, W//8)
        adapted_edge3 = self.edge_adapter3(F.interpolate(edge_map3, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False))
        x = x + adapted_edge3

        # Block 4: Final processing to latent space
        z = self.down_block4(x)
        return z


# --- Standard Decoder (Image Reconstruction) ---
class ImageDecoder(nn.Module):
    """
    Decoder for reconstructing the image from the latent code.
    """
    def __init__(self, latent_dim=32, out_channels=1, dim=64): # Adjusted dims
        super(ImageDecoder, self).__init__()
        # Upsample 64x64x32 -> 512x512x1
        # Reverse the encoder's downsampling path
        self.up_block1 = nn.Sequential( # Input: (B, 32, 64, 64)
            nn.ConvTranspose2d(latent_dim, dim * 4, kernel_size=3, stride=1, padding=1), # 64->64 (no up)
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(inplace=True)
        )
        self.up_block2 = nn.Sequential( # Input: (B, dim*4, 64, 64)
            nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1), # 64->128
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True)
        )
        self.up_block3 = nn.Sequential( # Input: (B, dim*2, 128, 128)
            nn.ConvTranspose2d(dim * 2, dim, kernel_size=4, stride=2, padding=1), # 128->256
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.up_block4 = nn.Sequential( # Input: (B, dim, 256, 256)
            nn.ConvTranspose2d(dim, out_channels, kernel_size=4, stride=2, padding=1), # 256->512
            nn.Sigmoid() # Or Tanh depending on data normalization
        )

    def forward(self, z):
        x = self.up_block1(z) # (B, dim*4, 64, 64)
        x = self.up_block2(x) # (B, dim*2, 128, 128)
        x = self.up_block3(x) # (B, dim, 256, 256)
        x = self.up_block4(x) # (B, out_channels, 512, 512)
        return x

# --- Perceptual Loss Helper (VGG-based) ---
class VGGLoss(nn.Module):
    """
    Calculates perceptual loss using features from a pre-trained VGG network.
    """
    def __init__(self, layers=[3, 8, 15, 22]): # Layers corresponding to VGG19 relu1_2, relu2_2, relu3_4, relu4_4
        super(VGGLoss, self).__init__()
        # Load pre-trained VGG19
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(layers[0]): # 0 to 3 (relu1_2)
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(layers[0], layers[1]): # 4 to 7 (relu2_2)
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(layers[1], layers[2]): # 8 to 14 (relu3_4)
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(layers[2], layers[3]): # 15 to 21 (relu4_4)
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, pred, target):
        # If input is single channel (1, H, W), repeat to 3 channels (3, H, W)
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # Normalize if required (uncomment if needed based on VGG input expectation)
        # pred = (pred - self.mean) / self.std
        # target = (target - self.mean) / self.std

        h1 = self.slice1(pred)
        h1_target = self.slice1(target)
        loss1 = F.mse_loss(h1, h1_target)

        h2 = self.slice2(h1)
        h2_target = self.slice2(h1_target)
        loss2 = F.mse_loss(h2, h2_target)

        h3 = self.slice3(h2)
        h3_target = self.slice3(h2_target)
        loss3 = F.mse_loss(h3, h3_target)

        h4 = self.slice4(h3)
        h4_target = self.slice4(h3_target)
        loss4 = F.mse_loss(h4, h4_target)

        # Weighted sum of losses from different layers (weights can be tuned)
        total_loss = loss1 + loss2 + loss3 + loss4
        return total_loss



class PerceptualAutoencoder(nn.Module):
    """
    The perceptual compression module combining encoder with edge prior and image decoder.
    Pre-trained separately using LMSE + LPerceptual loss.
    """
    def __init__(self, in_channels=1, latent_dim=32, dim=64): # Adjusted default dims
        super(PerceptualAutoencoder, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim, dim)
        self.image_decoder = ImageDecoder(latent_dim, in_channels, dim)
        self.vgg_loss = VGGLoss()

    def forward(self, x):
        z = self.encoder(x)
        y_recon = self.image_decoder(z)
        return y_recon, z # Return both reconstruction and latent code

    def calculate_compression_loss(self, y, y_recon):
        """
        Calculates the composite loss LCompress = LMSE + lambda * LPerceptual
        """
        mse_loss = F.mse_loss(y_recon, y)
        vgg_loss_val = self.vgg_loss(y_recon, y)
        # Lambda is a hyperparameter to balance MSE and Perceptual loss
        lambda_vgg = 0.1 # Example value, tune based on paper or experiments
        total_loss = mse_loss + lambda_vgg * vgg_loss_val
        return total_loss, mse_loss, vgg_loss_val


# --- Example Usage for Pre-training ---
if __name__ == '__main__':
    # Example parameters matching the paper's implementation details (Table I)
    latent_dim = 32  # c in paper's h x w x c (64x64x32)
    spatial_compressed = 64 # h, w in paper's latent space
    input_size = 512 # H, W of input image

    model = PerceptualAutoencoder(in_channels=1, latent_dim=latent_dim, dim=64) # Adjust dim if needed
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Example input batch
    x = torch.randn(2, 1, input_size, input_size).to(device) # Batch size 2

    # Forward pass
    y_recon, z = model(x)

