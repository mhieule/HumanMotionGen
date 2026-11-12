"""Scores for generated motions: TMR text-motion similarity and physical plausibility.

Used two ways: as the best-of-N verifier, and as extra evaluation metrics. Neither uses the
T2M evaluator (Guo et al.) that computes FID / R-precision, so best-of-N cannot optimise the
benchmark's own model directly. TMR and T2M were trained on the same HumanML3D train split,
so a TMR-selected sample can still look better to T2M than it is; check samples by eye.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TMR_DIR = ROOT / 'baselines/TMR'
sys.path.insert(0, str(ROOT / 'scripts'))
from motion_stats import stats  # noqa: E402  (scripts/motion_stats.py: float/sink/slide/jerk)

PHYS_KEYS = ['float_pct', 'sink_cm', 'slide_cms', 'jerk']


class ACTORStyleEncoder(torch.nn.Module):
    """Re-implementation of TMR's encoder (src/model/actor.py) so that only TMR's weights are
    loaded (tensor-only), never its code. Same parameter names, so its state_dict loads as is."""

    def __init__(self, nfeats, latent_dim=256, ff_size=1024, num_layers=6, num_heads=4, max_len=5000):
        super().__init__()
        self.projection = torch.nn.Linear(nfeats, latent_dim)
        self.tokens = torch.nn.Parameter(torch.zeros(2, latent_dim))  # mu, logvar tokens (VAE)
        layer = torch.nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads, dim_feedforward=ff_size,
                                                 dropout=0.1, activation='gelu', batch_first=True)
        self.seqTransEncoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        pos = torch.arange(max_len, dtype=torch.float)[:, None]
        div = torch.exp(torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim))
        pe = torch.zeros(max_len, latent_dim)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x, mask):
        """x: [B, T, nfeats], mask: [B, T] bool (True = valid). Returns latent mean [B, latent]."""
        x = self.projection(x)
        bs = len(x)
        xseq = torch.cat((self.tokens[None].expand(bs, -1, -1), x), 1)
        aug_mask = torch.cat((torch.ones(bs, 2, dtype=torch.bool, device=x.device), mask), 1)
        xseq = xseq + self.pe[None, : xseq.shape[1]]
        return self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)[:, 0]


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True)  # tensors only, no pickled code


class TMRScorer:
    """Cosine similarity between TMR's text and motion latents (latent means, as in TMR)."""

    def __init__(self, device, run='models/tmr_humanml3d_guoh3dfeats'):
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        w = TMR_DIR / run / 'last_weights'
        self.motion_enc = ACTORStyleEncoder(nfeats=263)
        self.text_enc = ACTORStyleEncoder(nfeats=768)
        self.motion_enc.load_state_dict(_load(w / 'motion_encoder.pt'))
        self.text_enc.load_state_dict(_load(w / 'text_encoder.pt'))
        self.motion_enc.to(device).eval()
        self.text_enc.to(device).eval()
        stats_dir = TMR_DIR / 'stats/humanml3d/guoh3dfeats'
        self.mean = _load(stats_dir / 'mean.pt').float().to(device)
        self.std = _load(stats_dir / 'std.pt').float().to(device)
        self.tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased').to(device).eval()

    @torch.no_grad()
    def encode_text(self, texts):
        enc = self.tok(list(texts), return_tensors='pt', padding=True).to(self.device)
        h = self.bert(**enc).last_hidden_state
        lat = self.text_enc(h, enc.attention_mask.bool())
        return torch.nn.functional.normalize(lat, dim=-1)

    @torch.no_grad()
    def encode_motion(self, raw, lengths):
        """raw: [B, T, 263] un-normalised HumanML3D features; lengths: [B]."""
        x = (raw.to(self.device).float() - self.mean) / (self.std + 1e-12)
        lengths = torch.as_tensor(lengths, device=self.device)
        mask = torch.arange(x.shape[1], device=self.device)[None] < lengths[:, None]
        lat = self.motion_enc(x, mask)
        return torch.nn.functional.normalize(lat, dim=-1)

    def similarity(self, raw, lengths, texts):
        return (self.encode_motion(raw, lengths) * self.encode_text(texts)).sum(-1)


def to_raw(samples, mean, std):
    """MDM sample [B, 263, 1, T] (normalised with HumanML3D Mean/Std) -> raw features [B, T, 263]."""
    x = samples.squeeze(2).permute(0, 2, 1)
    return x * std + mean


def physics(raw, lengths):
    """Per-sample physics stats on the valid frames. raw: [B, T, 263] torch."""
    from data_loaders.humanml.scripts.motion_process import recover_from_ric
    joints = recover_from_ric(raw.float(), 22).cpu().numpy()
    out = {k: [] for k in PHYS_KEYS}
    for j, n in zip(joints, np.asarray(lengths.cpu() if torch.is_tensor(lengths) else lengths)):
        s = stats(j[: int(n)])[0]
        for k in PHYS_KEYS:
            out[k].append(s[k])
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def make_verifier(kind, tmr, mean, std):
    """kind: 'tmr' (max text-motion similarity) or 'tmr_phys' (rank sum: TMR counts as much as
    foot sliding, floating and jerk together). Returns verifier(samples, y_rep, n, b)."""
    from guidance import ranks

    def verifier(samples, y_rep, n, b):
        raw = to_raw(samples, mean, std)
        lengths = y_rep['lengths']
        sim = tmr.similarity(raw, lengths, y_rep['text']).cpu().numpy().reshape(n, b)
        if kind == 'tmr':
            score = -ranks(sim, higher_better=True)
        elif kind == 'tmr_phys':
            ph = physics(raw, lengths)
            phys_rank = sum(ranks(np.nan_to_num(ph[k], nan=0.0).reshape(n, b), higher_better=False)
                            for k in ['slide_cms', 'float_pct', 'jerk']) / 3.0
            score = -(ranks(sim, higher_better=True) + phys_rank)
        else:
            raise ValueError(kind)
        idx = score.argmax(0)
        return torch.as_tensor(idx, device=samples.device), float(sim[idx, np.arange(b)].mean())

    return verifier
