# -*- coding: utf-8 -*-
# Forbereder stream_in for residens-laget: meta.txt, img_ids*.bin och lat_init.f32.
# Samma seed (1234) som run_scheduler.py sa resultatet blir direkt jamforbart.
# Sjalva scheduler-loopen kors sedan INTERNT i harnessen (ZI_STEPS=N).
import sys, numpy as np
from pathlib import Path

DIR  = Path(r"D:\prov\2026-09-24_fresh-golden\stream_in")
LH   = int(sys.argv[1]) if len(sys.argv) > 1 else 64
INCH, PATCH = 16, 2
Ht = Wt = LH // PATCH
Nimg = Ht * Wt
Scap = np.fromfile(DIR / "cap_raw.bin", np.float32).size // 2560
Su   = Nimg + Scap

img_ids = np.zeros((Nimg, 3), np.int32)
for y in range(Ht):
    img_ids[y*Wt:(y+1)*Wt, 0] = Scap + 1
    img_ids[y*Wt:(y+1)*Wt, 1] = y
    img_ids[y*Wt:(y+1)*Wt, 2] = np.arange(Wt)
for k in range(3):
    img_ids[:, k].astype(np.int32).tofile(DIR / f"img_ids{k}.bin")
(DIR / "meta.txt").write_text(f"{Ht} {Wt} {Nimg} {Scap} {Su}\n")

rng = np.random.default_rng(1234)
x = rng.standard_normal((INCH, LH, LH)).astype(np.float32)
x.tofile(DIR / "lat_init.f32")
print(f"[prep] latent {LH}x{LH} Nimg={Nimg} Scap={Scap} Su={Su} -> lat_init.f32 (seed 1234)")
