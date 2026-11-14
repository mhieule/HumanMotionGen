"""Does this machine's PyTorch compute attention correctly on the GPU? Run it in every model's env.

    <env>/bin/python scripts/check_attention_kernel.py          # e.g. baselines/mdm/.venv/bin/python

Background (2026-11-14): on the lab PC (RTX 5070 Ti, PyTorch 2.7.1 + CUDA 12.8) the
"memory-efficient" kernel behind torch.nn.functional.scaled_dot_product_attention returns wrong
values and wrong gradients for float32 inputs, silently. nn.MultiheadAttention / nn.TransformerEncoder
use that kernel by default, so MDM training diverged after ~600 steps and sampling was slightly off.
Write-up: vault/80 Setup/Attention kernel bug on the lab PC.md

What it checks, all against a float32 CPU reference:
  1. each attention kernel (flash, memory-efficient, cuDNN, math) in float32, float16 and bfloat16,
     for head sizes 32-256 (head size = model width / number of heads), forward and gradient;
  2. what PyTorch picks by default for each case;
  3. one MDM-sized nn.TransformerEncoderLayer, default kernels vs. memory-efficient off.
A kernel counts as wrong when its error is more than 10x that of the math kernel in the same dtype
(and above 1e-3). Exit code 1 if any kernel PyTorch would pick by default is wrong.
"""
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DEV = 'cuda'
BACKENDS = {'flash': SDPBackend.FLASH_ATTENTION, 'mem-efficient': SDPBackend.EFFICIENT_ATTENTION,
            'cudnn': SDPBackend.CUDNN_ATTENTION, 'math': SDPBackend.MATH}
DTYPES = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
print(f'torch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)} | '
      f'capability {torch.cuda.get_device_capability(0)}')


def inputs(scale, head_dim):
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(16, 4, 197, head_dim, generator=g) for _ in range(3))
    s = scale / head_dim ** 0.25
    return q * s, k * s, v


def run(q, k, v, dev, dtype, backend=None):
    """Attention output and gradient wrt q, as float32 on the CPU; None if the kernel refuses the input."""
    qq, kk, vv = (x.to(dev, dtype).requires_grad_(True) for x in (q, k, v))
    try:
        if backend is None:
            out = F.scaled_dot_product_attention(qq, kk, vv)
        else:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(qq, kk, vv)
        grad, = torch.autograd.grad(out.float().pow(2).sum(), qq)
    except RuntimeError:
        return None
    return out.detach().float().cpu(), grad.float().cpu()


def err(res, ref):
    if res is None:
        return None
    return ((res[0] - ref[0]).abs().max().item(),
            ((res[1] - ref[1]).abs().max() / ref[1].abs().max()).item())


HEAD_DIMS = [32, 64, 72, 80, 96, 112, 128, 136, 160, 256]
bad_default, rows = False, []
for dname, dtype in DTYPES.items():
    for hd in HEAD_DIMS:
        q, k, v = inputs(4, hd)
        ref = run(q, k, v, 'cpu', torch.float32)
        e_math = err(run(q, k, v, DEV, dtype, SDPBackend.MATH), ref)
        tol_out, tol_grad = max(10 * e_math[0], 1e-3), max(10 * e_math[1], 1e-3)
        for bname, backend in list(BACKENDS.items()) + [('DEFAULT', None)]:
            e = e_math if bname == 'math' else err(run(q, k, v, DEV, dtype, backend), ref)
            ok = None if e is None else (e[0] <= tol_out and e[1] <= tol_grad)
            rows.append((dname, hd, bname, e, ok))
            bad_default |= bname == 'DEFAULT' and ok is False

print('\nattention vs. CPU float32 reference: 16 x 4 heads x 197 tokens, sharp attention (logit scale 4)')
print('cell = max |output error| / max relative gradient error; "-" = kernel does not support the input')
names = list(BACKENDS) + ['DEFAULT']
print(f'{"dtype":>9} {"head dim":>8} | ' + ' | '.join(f'{n:>22}' for n in names))
for dname in DTYPES:
    for hd in HEAD_DIMS:
        cells = []
        for n in names:
            e, ok = next((e, ok) for d, h, b, e, ok in rows if d == dname and h == hd and b == n)
            cells.append(f'{"-":>22}' if e is None else f'{e[0]:8.1e}/{e[1]:8.1e}{"  ok" if ok else " BAD"}')
        print(f'{dname:>9} {hd:>8} | ' + ' | '.join(cells))


def layer_error(mem_efficient):
    torch.backends.cuda.enable_mem_efficient_sdp(mem_efficient)
    torch.manual_seed(2)
    layer = torch.nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=1024, dropout=0.0, activation='gelu')
    x = torch.randn(197, 64, 512) * 4
    ref_out = layer(x)
    out = layer.to(DEV)(x.to(DEV)).cpu()
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    return (out - ref_out).abs().max().item()


a, b = layer_error(True), layer_error(False)
bad_default |= a > 1e-3
print(f'\nnn.TransformerEncoderLayer (MDM-sized, float32), max |GPU - CPU|: default kernels {a:.2e} | mem-efficient off {b:.2e}')
wrong = sorted({f'{b} {d} head dim {h}' for d, h, b, _, ok in rows if ok is False and b != 'DEFAULT'},
               key=lambda x: (x.split()[0], x.split()[1], int(x.split()[-1])))
print('\nRESULT:', ('a kernel PyTorch uses BY DEFAULT is WRONG here; disable it (see wrong kernels below) before training or evaluating'
                    if bad_default else 'the kernels PyTorch picks by default agree with the CPU'))
print('wrong kernels:\n  ' + ('\n  '.join(wrong) if wrong else 'none'))
print('fix for mem-efficient: torch.backends.cuda.enable_mem_efficient_sdp(False); '
      'flash: enable_flash_sdp(False); cudnn: enable_cudnn_sdp(False)')
sys.exit(1 if bad_default else 0)
