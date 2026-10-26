"""Minimal 32×32 DCGAN generator compatible with inversefed.porting."""
import torch.nn as nn


class Generator(nn.Module):
    """Experiment support utility."""

    def __init__(self, ngpu=1, latent_dim=100, base_channels=64):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, latent):
        return self.main(latent)

    
    
    _stage_ranges = ((0, 3), (3, 6), (6, 9), (9, 11))

    def forward_stage(self, value, stage):
        """Handle GIFD generator stages and gradient matching."""
        begin, end = self._stage_ranges[stage]
        for layer_index in range(begin, end):
            value = self.main[layer_index](value)
        return value

    def forward_from(self, value, start_stage=0):
        """Experiment support utility."""
        for stage in range(start_stage, len(self._stage_ranges)):
            value = self.forward_stage(value, stage)
        return value

    @property
    def num_feature_stages(self):
        """Handle GIFD generator stages and gradient matching."""
        return len(self._stage_ranges)
