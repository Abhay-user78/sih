"""Phase 3: conditional denoising diffusion for super-resolution
(downscaling) of weather anomaly crops.

SR3-style conditional DDPM:
  * model input = [noisy sharp target, upscaled-coarse condition] (2 ch)
  * trained to predict the noise with a peak-aware, quantile-aware loss so
    the extreme values (storm core) are not smoothed away (MSE alone blurs
    peaks - explicitly avoided per the brief).
  * stochastic multi-pass sampling produces an ensemble used to derive a
    severity probability map.

Uses HuggingFace diffusers (DDPMScheduler + UNet2DModel).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel

from .config import CONFIG, model_dir


def make_models(size: int = 48, in_channels: int = 2,
               channels: tuple = (64, 128, 192, 256)) -> object:
    """Conditional SR3-style DDPM: [noisy target, coarse condition] -> denoise.

    Conditioning is achieved by channel concatenation (in_channels=2);
    `encoder_hidden_states` cross-attention is not needed for this task.
    """
    unet = UNet2DModel(
        sample_size=size,
        in_channels=in_channels,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=tuple(channels),
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = DDPMScheduler(num_train_timesteps=600,
                              beta_schedule="scaled_linear")
    return type("DiffusionModels", (), {
        "unet": unet, "scheduler": scheduler})()


def sample_crop(unet, sched, cond: np.ndarray, n_members: int = 24,
                n_steps: int = 100, device: str | None = None,
                seed: int = 0) -> np.ndarray:
    """Stochastic ensemble sampling for a single normalised coarse crop.

    cond: (1, 1, H, W) [-1,1] coarse conditioning.
    n_steps: number of denoising passes (fewer than training steps is the
    standard DDPM fast-sampling regime).
    Returns: (n_members, 1, H, W) normalised downscaled fields.
    """
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"
    unet.to(device).eval()
    c = torch.tensor(np.asarray(cond, dtype=np.float32), device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    sched.set_timesteps(min(n_steps, sched.config.num_train_timesteps),
                        device=device)
    H = W = c.shape[-1]
    members = []
    with torch.no_grad():
        for m in range(n_members):
            x = torch.randn((1, 1, H, W), generator=rng, device=device)
            for t in sched.timesteps:
                model_in = torch.cat([x, c], dim=1)
                pred = unet(model_in, t).sample
                x = sched.step(pred, t, x).prev_sample
            members.append(x[0].cpu().numpy())
    return np.stack(members)


def _peak_weight(x0: torch.Tensor, gamma: float = 3.0) -> torch.Tensor:
    """Per-pixel loss weight emphasising high-magnitude (extreme) pixels."""
    return 1.0 + gamma * torch.abs(x0).clamp(0, 1)


def train_downscale(dd, epochs: int = 40, batch_size: int = 16,
                    lr: float = 2e-4, gamma_peak: float = 3.0,
                    device: str | None = None,
                    out_path=None,
                    print_every: int = 8) -> dict:
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"
    models = make_models(size=dd.target.shape[-1])
    unet = models.unet.to(device)
    sched = models.scheduler
    opt = torch.optim.AdamW(unet.parameters(), lr=lr)

    x0_all = torch.tensor(dd.target, device=device)
    c_cond_all = torch.tensor(dd.coarse, device=device)
    sup_all = (torch.tensor(dd.support, device=device)
               if dd.support is not None else None)
    from . import physics as phys
    n = x0_all.shape[0]

    rows = []
    unet.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        tot = 0.0; nb = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x0 = x0_all[idx]
            cond = c_cond_all[idx]
            bsz = x0.shape[0]
            t = torch.randint(0, sched.config.num_train_timesteps,
                              (bsz,), device=device).long()
            noise = torch.randn_like(x0)
            noisy = sched.add_noise(x0, noise, t)
            model_in = torch.cat([noisy, cond], dim=1)
            pred = unet(model_in, t).sample
            w = _peak_weight(x0, gamma=gamma_peak)
            mse = F.mse_loss(pred, noise, reduction="none")
            # quantile-aware term: extra weight on extreme-magnitude pixels
            bsz = x0.shape[0]
            qw = torch.quantile(torch.abs(x0).reshape(bsz, -1), 0.9,
                                dim=1, keepdim=True)      # (bsz,1)
            qw = qw[..., None, None]                       # (bsz,1,1,1)
            boost = (1.0 + 2.0 * (torch.abs(x0) > qw).float())
            loss = (mse * w * boost).mean()
            if sup_all is not None:
                # physics: reconstruct clean field, penalise magnitude
                # wherever moist-convergence support is absent
                a_cum = sched.alphas_cumprod[t].reshape(-1, 1, 1, 1)
                x0_hat = (noisy - (1 - a_cum).sqrt() * pred) / a_cum.sqrt()
                loss = loss + phys.diffusion_physics_penalty(
                    x0_hat, sup_all[idx], weight=CONFIG.physics_diff_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        row = (ep, tot / max(nb, 1))
        rows.append(row)
        if ep % print_every == 0 or ep == epochs:
            print(f"  ep {ep:3d}  loss {row[1]:.4f}")

    torch.save({"unet": unet.state_dict(), "epochs": epochs,
                "losses": rows, "size": dd.target.shape[-1],
                "num_train_timesteps": sched.config.num_train_timesteps,
                "ts": int(sched.config.num_train_timesteps),
                "sched_cfg": sched.config},
               out_path or (model_dir("diffusion") / "downscale_unet.pt"))
    return {"losses": rows}


def load_downscale(device: str | None = None,
                   path=None):
    import torch
    ck = torch.load(path or (model_dir("diffusion") /
                             "downscale_unet.pt"),
                    map_location="cpu", weights_only=False)
    unet = UNet2DModel.from_config({
        "sample_size": ck["size"], "in_channels": 2, "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": (64, 128, 192, 256),
    })
    unet.load_state_dict(ck["unet"])
    sched = DDPMScheduler.from_config(ck["sched_cfg"])
    return unet.to(device or ("cuda" if torch.cuda.is_available() else "cpu")), sched


def sample_ensemble(dd, unet, sched, index: int, n_members: int = 24,
                    device: str | None = None, seed: int = 0) -> dict:
    """Stochastic multi-pass sampling for crop `index`.

    Returns members, cond, target (normalised), plus denoised intermediates.
    """
    rng = torch.Generator(device=device or "cpu").manual_seed(seed)
    device = (device or CONFIG.device)
    device = device if torch.cuda.is_available() else "cpu"
    rng = torch.Generator(device=device).manual_seed(seed)
    unet.to(device).eval()
    cond = torch.tensor(dd.coarse[index:index + 1], device=device)
    target = dd.target[index]

    sched.set_timesteps(sched.config.num_train_timesteps, device=device)
    members = []
    with torch.no_grad():
        for m in range(n_members):
            x = torch.randn((1, 1, dd.target.shape[-1], dd.target.shape[-1]),
                            generator=rng, device=device)
            for t in sched.timesteps:
                model_in = torch.cat([x, cond], dim=1)
                noise_pred = unet(model_in, t).sample
                x = sched.step(noise_pred, t, x).prev_sample
            members.append(x[0].cpu().numpy())
    members = np.stack(members)
    return {"members": members, "cond": dd.coarse[index],
            "target": target, "index": index, "n_members": n_members}


def severity_from_ensemble(members: np.ndarray, coarse: np.ndarray,
                           ref_quantile: float = 0.99,
                           min_delta_frac: float = 0.02) -> dict:
    """Derive severity probability from an ensemble of downscaled fields.

    severity_prob[i,j] = fraction of members that (a) exceed ref_threshold
    (by default the coarse field's `ref_quantile` quantile) and (b) do so
    by a physically meaningful margin (`min_delta_frac * field span`), i.e.
    members that reproduce a sharp extreme peak rather than noise.

    Returns the probability map, the ensemble mean/std and per-member maxima.
    """
    obs = members.mean(axis=0)[0]
    thr = float(np.quantile(coarse, ref_quantile))
    span = float(coarse.max() - coarse.min())
    margin = max(min_delta_frac * span, 1e-4)
    exceed = members[:, 0] >= (thr + margin)
    sev = exceed.mean(axis=0)
    return {
        "severity_prob": sev,
        "ensemble_mean": obs,
        "ensemble_std": members[:, 0].std(axis=0),
        "per_member_max": members[:, 0].max(axis=(1, 2)),
        "coarse_max": float(coarse.max()),
        "thr": thr,
    }