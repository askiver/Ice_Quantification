import torch
import torch.nn.functional as f
from torch import nn
import torch.nn.functional as F
from config import CONFIG


class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = CONFIG["TRAINING"]["DEVICE"]
        self.act_function = nn.GELU()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 14 x 14
            nn.BatchNorm2d(16),
            self.act_function,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 7 x 7
            nn.BatchNorm2d(32),
            self.act_function,
            nn.Conv2d(32, 64, kernel_size=7),  # Output: 64 x 1 x 1 (compressed representation)
            nn.BatchNorm2d(64),
            self.act_function,
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=7),  # Output: 32 x 7 x 7
            nn.BatchNorm2d(32),
            self.act_function,
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16 x 14 x 14
            nn.BatchNorm2d(16),
            self.act_function,
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 1 x 28 x 28
            nn.Sigmoid(),  # Output values between 0 and 1
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.encoder(x)
        return self.decoder(x)

    def loss(self, output: torch.tensor, target: torch.tensor) -> torch.tensor:
        return f.mse_loss(output, target)


class VariationalAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = CONFIG["TRAINING"]["DEVICE"]
        self.act_function = nn.GELU()
        self.latent_dim = 128

        # Encoder: Convolutional layers to map the input image to latent distribution
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (B, 32, 14, 14)
            self.act_function,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (B, 64, 7, 7)
            self.act_function,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),  # (B, 128, 3, 3)
            self.act_function,
            nn.Flatten(),  # (B, 128*3*3)
        )

        # Latent space: Two fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(128 * 3 * 3, self.latent_dim)
        self.fc_log_var = nn.Linear(128 * 3 * 3, self.latent_dim)

        # Decoder: Transposed convolutional layers to reconstruct the image
        self.fc_decode = nn.Linear(self.latent_dim, 128 * 3 * 3)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0, output_padding=0),  # (B, 64, 7, 7)
            self.act_function,
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 32, 14, 14)
            self.act_function,
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 1, 28, 28)
            nn.Sigmoid(),  # Use Sigmoid for pixel values in range [0, 1]
        )

    def encode(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_log_var(encoded)
        return mu, log_var

    def reparameterize(self, mu: torch.tensor, log_var: torch.tensor) -> torch.tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.tensor) -> torch.tensor:
        z = self.fc_decode(z)
        z = z.view(-1, 128, 3, 3)  # Reshape back into a spatial tensor
        return self.decoder(z)

    def forward(self, x: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var

    def loss(self, reconstructed_x: torch.tensor, mu: torch.tensor, log_var: torch.tensor, x: torch.tensor) \
            -> torch.tensor:
        # Reconstruction loss: Pixel-wise binary cross-entropy
        recon_loss = f.binary_cross_entropy(reconstructed_x, x, reduction="sum")

        # KL Divergence loss: KLD = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return recon_loss + kl_divergence



class SnowRanker(nn.Module):

    def __init__(self, num_channels, kernel_size):
        super().__init__()

        image_height = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["HEIGHT"]
        image_width = CONFIG["TRAINING"]["IMAGE_DIMENSIONS"]["WIDTH"]

        self.layers = nn.ModuleList()

        # Add cnn layers
        for _ in range(3):
            self.layers.append(nn.Conv2d(num_channels,num_channels*3, kernel_size, 1, kernel_size // 2))
            self.layers.append(nn.MaxPool2d(2, 2))
            self.layers.append(nn.BatchNorm2d(num_channels*3))
            self.layers.append(nn.GELU())
            num_channels *= 3
            image_height, image_width = image_height//2, image_width//2

        self.layers.append(nn.Flatten())
        # Add fc layers
        num_neurons = image_height * image_width * num_channels
        self.layers.append(nn.Linear(num_neurons, 200))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(200, 20))
        self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(20, 1))
        #self.layers.append(nn.ReLU())


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


    def loss(self, lower_img_output, higher_img_output):

        reward_diff = higher_img_output - lower_img_output
        loss = -torch.mean(F.logsigmoid(reward_diff))

        return loss

