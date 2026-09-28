import numpy as np, os
d = "D:/prov/2026-09-24_fresh-golden"
def L(p): return np.fromfile(os.path.join(d, p), dtype=np.float32)
def cmp(a, b, nm):
    x, y = L(a), L(b); n = min(len(x), len(y)); x, y = x[:n], y[:n]
    cos = float(np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-30))
    rel = float(np.linalg.norm(x-y)/(np.linalg.norm(y)+1e-30))
    print(f"{nm:14s} n={n:7d} cos={cos:.6f} relL2={rel:.6f} |a|={np.linalg.norm(x):.2f} |b|={np.linalg.norm(y):.2f} nan_a={int(np.isnan(x).sum())} nan_b={int(np.isnan(y).sum())}")
cmp("stream_in/ggml_out_htp.bin", "golden_out.f32", "DiT-out HTP")
cmp("stream_in/got_cap_after_cr.f32", "golden_cap_after_cr.f32", "cap_after_cr")
cmp("stream_in/got_adaln.f32", "adaln_ref.f32", "adaln")
