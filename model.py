import torch
import numpy as np


class SineLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, is_first=False, is_last=False, omega=30):
        super().__init__()
        self.omega = omega
        self.linear = torch.nn.Linear(in_features, out_features)
        self.linear.bias.data.fill_(0.0)
        self.is_last = is_last
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega, np.sqrt(6 / in_features) / omega)

    def forward(self, x):
        if self.is_last:
            return self.linear(x)
        return torch.sin(self.omega * self.linear(x))

class L2M(torch.nn.Module):  # Map latent modulation phi to theta
    def __init__(self, latent_dim, hidden_width, hidden_depth):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.modulations = hidden_width * hidden_depth
        self.linear = torch.nn.Linear(latent_dim, self.modulations)
        self.linear.bias.data.fill_(0.0)

    def forward(self, latent):
        modulations = self.linear(latent)  # [B, W*D]
        outputs = {i: modulations[:, self.hidden_width * i:self.hidden_width * i + self.hidden_width] for i in range(self.hidden_depth)}
        return outputs


class LatentModulatedSiren(torch.nn.Module):
    def __init__(self, in_dim=2, hidden_width=256, hidden_depth=5, out_dim=1, latent_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.out_dim = out_dim
        self.latent_dim = [latent_dim]

        self.l2m = L2M(latent_dim, hidden_width, hidden_depth)
        self.siren_layers = [SineLayer(in_dim, hidden_width, is_first=True)] + [SineLayer(hidden_width, hidden_width) for i in range(hidden_depth - 1)] + [SineLayer(hidden_width, out_dim, is_last=True)]

    def forward(self, coords, modulations):
        x = self.siren_layers[0](coords)  # [B, H*W, C]
        for layer in range(self.hidden_depth):
            x = self.siren_layers[layer + 1](x + modulations[layer].unsqueeze(1))  # modulation: [B, 1, C]
        return x.reshape(-1, self.out_dim)

class ConvL2M(torch.nn.Module):
    def __init__(self, latent_size, latent_channel, hidden_width, hidden_depth):
        super().__init__()
        self.latent_dim = [latent_channel, latent_size, latent_size]
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.modulations = hidden_width * hidden_depth
        self.conv = torch.nn.Conv2d(latent_channel, self.modultations, 3, padding=1)
        self.conv.bias.data.fill_(0.0)

    def forward(self, latent):
        modulations = self.conv(latent)  # [B, C, H, W]
        modulations = torch.nn.functional.interpolate(modulations, size=(32, 32), mode='nearest')
        outputs = {i: modulations[:, self.hidden_width * i:self.hidden_width * i + self.hidden_width] for i in range(self.hidden_depth)}
        return outputs


class SpatialFuncta(torch.nn.Module):
    def __init__(self, in_dim=2, hidden_width=256, hidden_depth=5, out_dim=1, latent_size=8, latent_channel=16):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.out_dim = out_dim
        self.latent_size = latent_size
        self.latent_channel = latent_channel
        self.latent_dim = [latent_channel, latent_size, latent_size]

        self.l2m = ConvL2M(latent_size, latent_channel, hidden_width, hidden_depth)
        self.siren_layers = [SineLayer(in_dim, hidden_width, is_first=True, omega=10)] + [SineLayer(hidden_width, hidden_width, omega=10) for i in range(hidden_depth - 1)] + [SineLayer(hidden_width, out_dim, omega=10, is_last=True)]

    def forward(self, coords, modulations):
        x = self.siren_layers[0](coords)
        x = x.view(-1, 32, 32, self.hidden_width)  # [B, H, W, C]
        for layer in range(self.hidden_depth):
            x = self.siren_layers[layer + 1](x + modulations[layer].permute(0, 2, 3, 1))  # [B, H, W, C]
        return x.reshape(-1, self.out_dim)
