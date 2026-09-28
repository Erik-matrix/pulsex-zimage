#!/usr/bin/env python3
"""Convert the real dense-reference golden dump (dit_cond/) into the flat
.bin file layout zimage-dit-stream.exe reads, so we can test against REAL
in-distribution activations instead of synthetic random noise. Also copies
the golden mid-layer checkpoints for direct comparison.
"""
import json
import shutil
from pathlib import Path

import numpy as np

SRC = Path(r"C:\PulseCore\models\zImage\dit_cond")
DST = Path(r"D:\prov\2026-09-22_zimage-gguf\golden_test")
DST.mkdir(parents=True, exist_ok=True)

meta = json.loads((SRC / "meta.json").read_text())
DIM, Nimg, Scap, Su = meta["DIM"], meta["Nimg"], meta["Scap"], meta["Su"]
Himg, Wimg, Ht, Wt, PATCH, INCH, CAPD = meta["Himg"], meta["Wimg"], meta["Ht"], meta["Wt"], meta["PATCH"], meta["INCH"], meta["CAPD"]
print(f"[prep] meta: Nimg={Nimg} Scap={Scap} Su={Su} Ht={Ht} Wt={Wt}")

latent = np.fromfile(SRC / "latent.f32", dtype=np.float32).reshape(INCH, Himg, Wimg)
img_raw = np.zeros((Nimg, PATCH * PATCH * INCH), dtype=np.float32)
for hi in range(Ht):
    for wi in range(Wt):
        for phi in range(PATCH):
            for pwi in range(PATCH):
                for c in range(INCH):
                    img_raw[hi * Wt + wi, (phi * PATCH + pwi) * INCH + c] = latent[c, hi * PATCH + phi, wi * PATCH + pwi]
img_raw.tofile(DST / "img_raw.bin")

cap_feats = np.fromfile(SRC / "cap_feats.f32", dtype=np.float32).reshape(Scap, CAPD)
cap_feats.tofile(DST / "cap_raw.bin")

adaln = np.fromfile(SRC / "adaln.f32", dtype=np.float32)
assert adaln.size == 256, adaln.size
adaln.tofile(DST / "adaln.bin")

img_ids = np.fromfile(SRC / "img_ids.f32", dtype=np.float32).reshape(Nimg, 3)
cap_ids = np.fromfile(SRC / "cap_ids.f32", dtype=np.float32).reshape(Scap, 3)
img_ids[:, 0].astype(np.int32).tofile(DST / "img_ids0.bin")
img_ids[:, 1].astype(np.int32).tofile(DST / "img_ids1.bin")
img_ids[:, 2].astype(np.int32).tofile(DST / "img_ids2.bin")
cap_ids[:, 0].astype(np.int32).tofile(DST / "cap_ids0.bin")
cap_ids[:, 1].astype(np.int32).tofile(DST / "cap_ids1.bin")
cap_ids[:, 2].astype(np.int32).tofile(DST / "cap_ids2.bin")

with open(DST / "meta.txt", "w") as f:
    f.write(f"{Ht}\n{Wt}\n{Nimg}\n{Scap}\n{Su}\n")

for name in ["mid_cap_after_cr.f32", "mid_x_after_nr.f32", "mid_uni_pre.f32", "mid_uni_L0.f32", "out.f32"]:
    shutil.copy(SRC / name, DST / name)

print(f"[prep] wrote inputs + golden checkpoints to {DST}")
print(f"[prep] out.f32 stats from meta: mean={meta['out_mean']:.5f} std={meta['out_std']:.5f} norm={meta['out_norm']:.3f}")
