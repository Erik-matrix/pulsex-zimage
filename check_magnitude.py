import numpy as np
import os

D = r"D:\prov\2026-09-22_zimage-gguf\block_test"
H = os.path.join(D, "htp_fixed")
for stage, mag in [("dbg_w1o", 143), ("dbg_w3o", 273), ("dbg_inter", 13660), ("dbg_w2o", 131000)]:
    cpu = np.fromfile(os.path.join(D, stage + ".bin"), dtype=np.float32)
    htp = np.fromfile(os.path.join(H, stage + ".bin"), dtype=np.float32)
    mask = np.isfinite(cpu) & np.isfinite(htp)
    se = np.sum(((htp[mask] - cpu[mask]).astype(np.float64)) ** 2)
    sy = np.sum(cpu[mask].astype(np.float64) ** 2)
    print(f"{stage:12s} typ.magnitude~{mag:>7d}  relL2={np.sqrt(se / sy):.5f}")
