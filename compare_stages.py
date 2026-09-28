import numpy as np
import os

D = r"D:\prov\2026-09-22_zimage-gguf\block_test"
H = os.path.join(D, "htp_run")
stages = ["dbg_q_rope", "dbg_scores", "dbg_attn", "dbg_x_after_attn", "dbg_w1o", "dbg_w3o", "dbg_inter", "dbg_w2o", "ggml_out"]
for s in stages:
    cpu = np.fromfile(os.path.join(D, s + ".bin"), dtype=np.float32)
    htp = np.fromfile(os.path.join(H, s + ".bin"), dtype=np.float32)
    if cpu.size != htp.size:
        print(s, "SIZE MISMATCH", cpu.size, htp.size)
        continue
    se = np.sum((htp - cpu).astype(np.float64) ** 2)
    sy = np.sum(cpu.astype(np.float64) ** 2)
    rel = np.sqrt(se / sy) if sy > 0 else float("nan")
    print(f"{s:20s} n={cpu.size:6d}  relL2(htp vs cpu)={rel:.6f}  cpu_max={np.abs(cpu).max():.4g}  htp_max={np.abs(htp).max():.4g}")
