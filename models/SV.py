import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, latent_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        query = self.query(z)
        key = self.key(z)
        value = self.value(z)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attention_weights = F.softmax(
            torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)),
            dim=-1)
        z_attention = torch.matmul(attention_weights, value).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        z_attention = self.fc_out(z_attention)
        return z_attention, attention_weights

class CustomVAE_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_heads=8):
        super(CustomVAE_Decoder, self).__init__()
        self.multihead_attention = MultiHeadSelfAttention(latent_dim, num_heads)
        self.fc1 = nn.Linear(latent_dim , 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, output_dim)

    def forward(self, z):
        # Apply multi-head attention
        z_attention, attention_weights = self.multihead_attention(z)
        # Fully connected layers
        z = F.relu(self.fc1(z_attention))
        z = F.relu(self.fc2(z))
        x_recon = torch.sigmoid(self.fc3(z))
        return x_recon, attention_weights

class CustomVAE_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(CustomVAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class CustomVAE_CapsuleIntegration(nn.Module):
    def __init__(self, input_dim,latent_dim):
        super(CustomVAE_CapsuleIntegration, self).__init__()
        self.vae_encoder = CustomVAE_Encoder(input_dim, latent_dim)
        self.vae_decoder = CustomVAE_Decoder(latent_dim, input_dim,num_heads=8)
        noise_std = 0.1
        self.noise_std = noise_std

    def forward(self, x):
        x_noisy = x + torch.randn_like(x) * self.noise_std
        mu, logvar = self.vae_encoder(x_noisy)
        z = self.sample_latent(mu, logvar)
        x_recon, attention_weights = self.vae_decoder(z)
        return x_recon, attention_weights,mu,logvar

    def sample_latent(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z


class CustomVAE_CapsuleLoss(nn.Module):
    def __init__(self, recon_weight=1.0, kl_weight=1.0):
        super(CustomVAE_CapsuleLoss, self).__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight

    def forward(self,  x_recon, mu, logvar, x):
        x = x.squeeze(1)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        total_loss = self.recon_weight * recon_loss + self.kl_weight * kl_loss
        return total_loss

