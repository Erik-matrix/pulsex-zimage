import os
import numpy as np

D = r"D:\prov\2026-09-22_zimage-gguf\golden_test"
pairs = [
    ("got_cap_after_cr.f32", "mid_cap_after_cr.f32"),
    ("got_x_after_nr.f32", "mid_x_after_nr.f32"),
    ("got_uni_pre.f32", "mid_uni_pre.f32"),
    ("layer_00.bin", "mid_uni_L0.f32"),
]
for got, gold in pairs:
    a = np.fromfile(os.path.join(D, got), dtype=np.float32)
    b = np.fromfile(os.path.join(D, gold), dtype=np.float32)
    if a.size != b.size:
        print(f"{got:24s} vs {gold:20s}  SIZE MISMATCH {a.size} vs {b.size}")
        continue
    se = np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    sy = np.sum(b.astype(np.float64) ** 2)
    print(f"{got:24s} vs {gold:20s}  relL2={np.sqrt(se / sy):.6f}  got_max={np.abs(a).max():.4f}  gold_max={np.abs(b).max():.4f}")
