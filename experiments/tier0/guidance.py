"""Inference-time guidance for MDM (Tier 0): CFG, interval CFG, APG, best-of-N.

All of it wraps a trained MDM; nothing here changes weights.

    GuidedModel     replaces MDM's ClassifierFreeSampleModel. Plain CFG reproduces MDM's
                    formula exactly. Adds: guidance only on a step interval (interval CFG,
                    Kynkaenniemi et al. 2024), adaptive projected guidance (APG, Sadat et al.
                    2024), and caching of the CLIP text embedding (outputs identical, faster).
    BestOfNDiffusion stands in for the diffusion object given to MDM's eval loader: draws N
                    candidates per prompt in one batch and keeps the best one per verifier.
"""
import numpy as np
import torch
import torch.nn as nn


class GuidedModel(nn.Module):
    """x0-prediction guidance. Step index t: 0 = clean ... T-1 = pure noise (MDM convention).

    scale:      guidance weight w (MDM: out_u + w * (out_c - out_u)); 1 = no guidance.
    interval:   (t_lo, t_hi) inclusive; outside it only the conditional pass runs (1 NFE/step).
    apg:        None, or dict(eta, momentum, norm_threshold). eta=1, momentum=0, norm=0 == CFG.
    """

    def __init__(self, model, scale=2.5, interval=None, apg=None, num_steps=50):
        super().__init__()
        self.model = model
        self.scale = float(scale)
        self.interval = interval
        self.apg = apg
        self.num_steps = num_steps
        self.nfe = 0                 # network passes per sample, accumulated (reset by caller)
        self._momentum = None
        # attributes MDM's samplers / loaders read from the model
        for k in ['rot2xyz', 'translation', 'njoints', 'nfeats', 'data_rep', 'cond_mode', 'encode_text']:
            setattr(self, k, getattr(model, k))

    def guided_at(self, t):
        if self.scale == 1.0:
            return False
        if self.interval is None:
            return True
        lo, hi = self.interval
        return lo <= t <= hi

    def forward(self, x, timesteps, y=None):
        if 'text_embed' not in y:      # encode the captions once per sampling call, not every step
            with torch.no_grad():
                y['text_embed'] = self.model.encode_text(y['text'])
        t = int(timesteps[0].item())
        if t == self.num_steps - 1:   # first step of a new trajectory
            self._momentum = None
        out_c = self.model(x, timesteps, y)
        self.nfe += 1
        if not self.guided_at(t):
            return out_c
        y_u = dict(y)
        y_u['uncond'] = True
        out_u = self.model(x, timesteps, y_u)
        self.nfe += 1
        if self.apg is None:
            return out_u + self.scale * (out_c - out_u)
        return self._apg(out_c, out_u, y.get('mask'))

    def _apg(self, cond, uncond, mask):
        """APG on x0 predictions; inner products / norms only over valid (unpadded) frames."""
        eta = self.apg.get('eta', 0.0)
        beta = self.apg.get('momentum', 0.0)
        r = self.apg.get('norm_threshold', 0.0)
        diff = cond - uncond
        if beta != 0.0:
            self._momentum = diff if self._momentum is None else diff + beta * self._momentum
            diff = self._momentum
        m = torch.ones_like(cond) if mask is None else mask.to(cond.dtype).expand_as(cond)
        dims = (1, 2, 3)
        if r > 0:  # threshold on the per-element RMS of the update (scale-free for any length)
            rms = ((diff ** 2 * m).sum(dims, keepdim=True) / m.sum(dims, keepdim=True)).sqrt()
            diff = diff * torch.clamp(r / (rms + 1e-8), max=1.0)
        c = cond * m
        coef = (diff * c).sum(dims, keepdim=True) / ((c * c).sum(dims, keepdim=True) + 1e-8)
        parallel = coef * cond
        update = (diff - parallel) + eta * parallel
        return cond + (self.scale - 1.0) * update


def _repeat_y(y, n):
    """Repeat every per-sample entry of MDM's y dict n times (candidate-major: [c0 b0..bB, c1 ...])."""
    out = {}
    for k, v in y.items():
        if k == 'text_embed':
            out[k] = v.repeat(1, n, 1) if v.dim() == 3 else v.repeat(n, *[1] * (v.dim() - 1))
        elif torch.is_tensor(v) and v.dim() > 0:
            out[k] = v.repeat(n, *[1] * (v.dim() - 1))
        elif isinstance(v, list):
            out[k] = v * n
        else:
            out[k] = v
    return out


class BestOfNDiffusion:
    """Drop-in for `diffusion` in MDM's CompMDMGeneratedDataset: only p_sample_loop is used."""

    def __init__(self, diffusion, n, verifier):
        self.diffusion = diffusion
        self.n = n
        self.verifier = verifier      # callable(samples[N*B,263,1,T], y_rep, n, b) -> chosen idx [B]
        self.chosen_scores = []

    def p_sample_loop(self, model, shape, model_kwargs=None, **kw):
        b = shape[0]
        y = model_kwargs['y']
        if 'text_embed' not in y:
            with torch.no_grad():
                y['text_embed'] = model.encode_text(y['text'])
        y_rep = _repeat_y(y, self.n)
        samples = self.diffusion.p_sample_loop(model, (b * self.n,) + tuple(shape[1:]),
                                               model_kwargs={'y': y_rep}, **kw)
        idx, best = self.verifier(samples, y_rep, self.n, b)
        self.chosen_scores.append(best)
        cand = samples.view(self.n, b, *samples.shape[1:])
        return cand[idx, torch.arange(b, device=samples.device)]


def ranks(x, higher_better):
    """Per-prompt ranks over the candidate axis 0 of x[N, B]; 0 = best."""
    order = np.argsort(-x if higher_better else x, axis=0, kind='stable')
    r = np.empty_like(order)
    np.put_along_axis(r, order, np.arange(x.shape[0])[:, None].repeat(x.shape[1], 1), axis=0)
    return r.astype(np.float64)
