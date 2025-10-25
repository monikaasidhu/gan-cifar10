import torch
import torch.nn as nn
from torchvision.utils import save_image

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_channels=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
generator.load_state_dict(torch.load('generator_final.pth', map_location=device))
generator.eval()

with torch.no_grad():
    noise = torch.randn(64, 100, 1, 1, device=device)
    fake_images = generator(noise).cpu()
    save_image(fake_images, 'generated_images.png', normalize=True, nrow=8)
    print("Generated 64 images saved to generated_images.png")
