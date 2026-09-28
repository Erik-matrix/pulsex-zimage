import numpy as np, sys, os
d = "D:\\prov\\2026-09-24_fresh-golden\\stream_in"
def load(p): return np.fromfile(os.path.join(d, p), dtype=np.float32)
for name, a, b in [("DiT-out", "ggml_out_htp.bin", "ggml_out_cpu.bin"),
                   ("final",   "got_final_out_htp.f32", "got_final_out_cpu.f32")]:
    try:
        x, y = load(a), load(b)
    except Exception as e:
        print(f"{name}: SKIP ({e})"); continue
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
    rel = float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30))
    print(f"{name:8s} n={n} cos={cos:.6f} relL2={rel:.6f} "
          f"|htp|={np.linalg.norm(x):.2f} |cpu|={np.linalg.norm(y):.2f} "
          f"nan_htp={int(np.isnan(x).sum())} nan_cpu={int(np.isnan(y).sum())}")
