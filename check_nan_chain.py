import os
import numpy as np

D = r"D:\prov\2026-09-22_zimage-gguf\golden_test"
files = ["got_x_after_embed.f32", "got_x_after_nr0.f32", "got_x_after_nr1.f32", "got_x_after_nr.f32",
         "got_cap_after_embed.f32", "got_cap_after_cr.f32"]
for f in files:
    p = os.path.join(D, f)
    if not os.path.exists(p):
        print(f, "MISSING")
        continue
    a = np.fromfile(p, dtype=np.float32)
    print(f, "n=", a.size, "nan=", int(np.isnan(a).sum()), "inf=", int(np.isinf(a).sum()))
