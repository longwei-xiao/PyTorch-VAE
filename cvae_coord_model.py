import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import math


# ---------------- Conditional VAE ----------------
class ConditionalVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 loc_channels: int = 2,
                 mask_channels: int = 1,
                 latent_dim: int = 16,
                 hidden_dims: List[int] = [32, 64, 128, 256],
                 img_size: int = 40,
                 coord_decode: bool = True):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels
        self.loc_channels = loc_channels
        self.mask_channels = mask_channels
        self.coord_decode = coord_decode

        # ---------------- Encoder ----------------
        enc_in_channels = in_channels + loc_channels + mask_channels
        self.hidden_dims_enc = hidden_dims
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(enc_in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            enc_in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        # Compute encoder output sizes dynamically
        encoder_sizes = self.compute_encoder_sizes(img_size, hidden_dims)
        self.final_spatial = encoder_sizes[-1]
        flat_size = hidden_dims[-1] * self.final_spatial**2

        # Latent space
        self.fc_mu = nn.Linear(flat_size, latent_dim)
        self.fc_var = nn.Linear(flat_size, latent_dim)

        # ---------------- CNN Decoder (optional) ----------------
        if not coord_decode:
            self.decoder_input = nn.Linear(
                latent_dim + (loc_channels + mask_channels) * img_size * img_size,
                flat_size
            )
            self.hidden_dims_dec = hidden_dims[::-1]
            dec_modules = []
            for i in range(len(self.hidden_dims_dec)-1):
                dec_modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(
                            self.hidden_dims_dec[i],
                            self.hidden_dims_dec[i+1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            output_padding=0
                        ),
                        nn.BatchNorm2d(self.hidden_dims_dec[i+1]),
                        nn.LeakyReLU()
                    )
                )
            self.decoder = nn.Sequential(*dec_modules)
            self.final_mu = nn.Conv2d(self.hidden_dims_dec[-1], out_channels=in_channels, kernel_size=3, padding=1)
            self.final_logvar = nn.Conv2d(self.hidden_dims_dec[-1], out_channels=in_channels, kernel_size=3, padding=1)
        else:
            # ---------------- Coordinate-based decoder ----------------
            self.coord_mlp = nn.Sequential(
                nn.Linear(latent_dim + 2, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 2)  # predict mu and logvar
            )

    # ---------------- Static methods ----------------
    @staticmethod
    def compute_encoder_sizes(img_size, hidden_dims, kernel_size=3, stride=2, padding=1):
        sizes = []
        size = img_size
        for _ in hidden_dims:
            size = (size + 2*padding - kernel_size) // stride + 1
            sizes.append(size)
        return sizes

    # ---------------- Forward methods ----------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z: torch.Tensor, loc_mask: torch.Tensor = None, meas_mask: torch.Tensor = None,
               coords: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.coord_decode:
            if coords.dim() == 2:  # [N,2] -> [1,N,2]
                coords = coords.unsqueeze(0)
            if coords.shape[0] != z.shape[0]:
                coords = coords.repeat(z.shape[0], 1, 1)  # [B,N,2]
            B, N, _ = coords.shape
            z_exp = z.unsqueeze(1).repeat(1, N, 1)  # [B, N, latent_dim]
            mlp_input = torch.cat([z_exp, coords], dim=-1)  # [B, N, latent+2]
            out = self.coord_mlp(mlp_input)  # [B, N, 2]
            mu_pred = out[..., 0:1]
            logvar_pred = out[..., 1:2]
            return mu_pred, logvar_pred
        else:
            # CNN decode (full grid)
            cond = torch.cat([loc_mask, meas_mask], dim=1)
            cond_flat = cond.view(cond.size(0), -1)
            dec_input = torch.cat([z, cond_flat], dim=1)
            result = self.decoder_input(dec_input)
            result = result.view(-1, self.hidden_dims_dec[0], self.final_spatial, self.final_spatial)
            result = self.decoder(result)

            # Zero-pad to img_size
            _, _, h, w = result.shape
            pad_h = self.img_size - h
            pad_w = self.img_size - w
            if pad_h > 0 or pad_w > 0:
                result = F.pad(result, (0, pad_w, 0, pad_h))

            mu_pred = self.final_mu(result)
            logvar_pred = torch.clamp(self.final_logvar(result), -10, 10)
            return mu_pred, logvar_pred

    def forward(self, x: torch.Tensor, loc_mask: torch.Tensor = None, meas_mask: torch.Tensor = None,
                coords: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tile location mask if needed
        if loc_mask is not None and loc_mask.dim() == 3:
            loc_mask = loc_mask.unsqueeze(0).repeat(x.size(0), 1, 1, 1)

        enc_in = x
        if loc_mask is not None and meas_mask is not None:
            enc_in = torch.cat([x, loc_mask, meas_mask], dim=1)

        mu, log_var = self.encode(enc_in)
        z = self.reparameterize(mu, log_var)
        mu_pred, logvar_pred = self.decode(z, loc_mask, meas_mask, coords)
        return mu_pred, logvar_pred, mu, log_var

    def loss_function(self,
        mu_pred,         # [B, N, 1] predicted mean at 650 coords
        logvar_pred,     # [B, N, 1] predicted logvar at 650 coords
        x_values,        # [B, N, 1] ground truth at 650 coords
        meas_mask,       # [B, N, 1] 1 for 30 measured pts, 0 otherwise
        mu, log_var,     # latent distribution params
        kld_weight=1e-3
    ):

        # -------------------------------------------------------
        # 1. Stabilize predicted log-variance
        # -------------------------------------------------------
        logvar_pred = torch.clamp(logvar_pred, -10, 10)
        recon_var = torch.exp(logvar_pred)

        # Constant log(2π)
        const = math.log(2.0 * math.pi)

        # -------------------------------------------------------
        # 2. Gaussian NLL per point
        #    0.5 * [ (x - μ)^2 / σ²  +  log σ²  + log(2π) ]
        # -------------------------------------------------------
        nll_element = 0.5 * (
            (x_values - mu_pred)**2 / recon_var +
            logvar_pred +
            const
        )

        # -------------------------------------------------------
        # 3. Apply measurement mask → only 30 points contribute
        # -------------------------------------------------------
        nll_element = nll_element * meas_mask   # zeros out non-measured points

        # -------------------------------------------------------
        # 4. Normalize per sample by number of measured points
        # -------------------------------------------------------
        valid_counts = meas_mask.sum(dim=1).clamp(min=1e-8)  # [B,1]

        nll_per_sample = torch.sum(nll_element, dim=1) / valid_counts  # [B,1] / [B,1]

        nll_loss = torch.mean(nll_per_sample)

        # -------------------------------------------------------
        # 5. KL divergence term
        # -------------------------------------------------------
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

        # -------------------------------------------------------
        # 6. Total loss
        # -------------------------------------------------------
        total_loss = nll_loss + kld_weight * kld_loss

        return {
            "loss": total_loss,
            "NLL": nll_loss,
            "KLD": kld_loss
        }
