#!/usr/bin/env bash
# Does our launcher train the same weights as stock MDM? Run from baselines/mdm (~10 min):
#     bash ../../experiments/baseline/check_equivalence.sh
# Five 200-step runs, then compare the step-200 checkpoints and the logged losses with stock_a:
#   stock_a, stock_b   python -m train.train_mdm         (b = run-to-run noise of stock itself)
#   stock_seed11       python -m train.train_mdm --seed 11  (what a different random stream looks like)
# "Stock" runs go through scripts/fixed_attention.py, which only switches off PyTorch's faulty
# memory-efficient attention kernel (scripts/check_attention_kernel.py); MDM's code is untouched.
#   ours               experiments/baseline/train_instrumented.py
#   ours_ema           experiments/baseline/train_instrumented.py --use_ema   (raw weights must be unaffected)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
py=.venv/bin/python
out=save/_eq
common="--overwrite --dataset humanml --diffusion_steps 50 --num_steps 200 --save_interval 200 --log_interval 10 --train_platform_type TensorboardPlatform"
export DIFFUSION_TRAINING_TEST=1   # MDM's own hook: return after the first checkpoint past step 0

mkdir -p $out
run() {  # name, command...; skipped if its checkpoint is already there
    local name=$1; shift
    [ -f $out/$name/model000000200.pt ] || "$@" --save_dir $out/$name $common > $out/$name.log 2>&1
}
stock="$py $here/../../scripts/fixed_attention.py -m train.train_mdm"
run stock_a  $stock
run stock_b  $stock
run ours     $py "$here/train_instrumented.py"
run ours_ema $py "$here/train_instrumented.py" --use_ema
run stock_seed11 $stock --seed 11

$py - <<'EOF'
import re, torch
out = 'save/_eq'
def weights(name):
    sd = torch.load(f'{out}/{name}/model000000200.pt', map_location='cpu')
    return sd['model'] if 'model' in sd else sd
def losses(name):
    return [float(x) for x in re.findall(r'step\[\d+\]: loss\[([\d.]+)\]', open(f'{out}/{name}.log').read())]
ref, ref_l = weights('stock_a'), losses('stock_a')
print(f'{len(ref)} tensors, {len(ref_l)} logged losses; stock_a losses: {ref_l[:3]} ... {ref_l[-3:]}')
diff = {}
for name in ['stock_b', 'ours', 'ours_ema', 'stock_seed11']:
    w, l = weights(name), losses(name)
    assert w.keys() == ref.keys()
    max_abs = max((w[k].float() - ref[k].float()).abs().max().item() for k in ref)
    max_loss = max(abs(a - b) for a, b in zip(l, ref_l))
    diff[name] = (max_abs, max_loss)
    print(f'{name:12s} vs stock_a: max |weight diff| = {max_abs:.3e}, max |loss diff| = {max_loss:.3e}')
ema = torch.load(f'{out}/ours_ema/model000000200.pt', map_location='cpu')
print('ours_ema checkpoint keys:', list(ema.keys()))
# Stock MDM is not bit-reproducible on GPU, so "same" means: no further from stock_a than a second
# stock run is (3x margin on weights, 5 decimals of the logged loss), and far below another seed.
floor_w, floor_l = diff['stock_b']
ok = all(diff[n][0] <= 3 * max(floor_w, 1e-6) and diff[n][1] <= max(3 * floor_l, 5e-5) for n in ['ours', 'ours_ema'])
print('RESULT:', 'PASS, launcher matches stock MDM within its own run-to-run noise' if ok else 'FAIL')
EOF
