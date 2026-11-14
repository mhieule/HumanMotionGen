"""Run any Python module with the faulty attention kernel disabled (see check_attention_kernel.py).

From baselines/mdm, instead of `python -m sample.generate ...`:
    .venv/bin/python ../../scripts/fixed_attention.py -m sample.generate --model_path ...
"""
import os
import runpy
import sys

import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
assert len(sys.argv) > 2 and sys.argv[1] == '-m', __doc__
module = sys.argv[2]
sys.argv = [module] + sys.argv[3:]
sys.path.insert(0, os.getcwd())
runpy.run_module(module, run_name='__main__', alter_sys=True)
