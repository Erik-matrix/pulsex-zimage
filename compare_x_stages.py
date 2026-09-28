import os
import numpy as np

D = r"D:\prov\2026-09-22_zimage-gguf\golden_test"

def relL2(a, b):
    se = np.sum((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    sy = np.sum(b.astype(np.float64) ** 2)
    return np.sqrt(se / sy)

for stage, gname in [("got_x_after_embed.f32", "ref_x_after_embed.f32"),
                      ("got_x_after_nr0.f32", "ref_x_after_nr0.f32"),
                      ("got_x_after_nr1.f32", "ref_x_after_nr1.f32")]:
    a = np.fromfile(os.path.join(D, stage), dtype=np.float32)
    b = np.fromfile(os.path.join(D, gname), dtype=np.float32)
    print(f"{stage:24s} vs {gname:24s}  ggml-vs-numpyref relL2={relL2(a,b):.6f}  ggml_max={np.abs(a).max():.4f} ref_max={np.abs(b).max():.4f}")
