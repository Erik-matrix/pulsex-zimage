# -*- coding: utf-8 -*-
# Bygger en komplett indata-katalog for en given bildstorlek, med SAMMA seed och SAMMA
# cap_raw som de andra, sa 256/512/1024 blir direkt jamforbara.
import sys, shutil
import numpy as np
from pathlib import Path

PX   = int(sys.argv[1])                      # 256 / 512 / 1024
DST  = Path(sys.argv[2])
SRC  = Path(r"D:\prov\2026-09-25_fa-bc-sweep\stream_in_q4cap")   # dar cap_raw + cap_ids finns
INCH, PATCH, VAE = 16, 2, 8
LH   = PX // VAE                             # latenthojd: 1024->128, 512->64, 256->32
Ht = Wt = LH // PATCH
Nimg = Ht * Wt
DST.mkdir(parents=True, exist_ok=True)

for f in ("cap_raw.bin", "cap_ids0.bin", "cap_ids1.bin", "cap_ids2.bin"):
    shutil.copy(SRC / f, DST / f)
Scap = np.fromfile(DST / "cap_raw.bin", np.float32).size // 2560
Su   = Nimg + Scap

img_ids = np.zeros((Nimg, 3), np.int32)
for y in range(Ht):
    img_ids[y*Wt:(y+1)*Wt, 0] = Scap + 1
    img_ids[y*Wt:(y+1)*Wt, 1] = y
    img_ids[y*Wt:(y+1)*Wt, 2] = np.arange(Wt)
for k in range(3):
    img_ids[:, k].astype(np.int32).tofile(DST / f"img_ids{k}.bin")

(DST / "meta.txt").write_text(f"{Ht} {Wt} {Nimg} {Scap} {Su}\n")

rng = np.random.default_rng(1234)            # samma seed som run_scheduler.py
rng.standard_normal((INCH, LH, LH)).astype(np.float32).tofile(DST / "lat_init.f32")
np.zeros(Nimg * (INCH * PATCH * PATCH), np.float32).tofile(DST / "img_raw.bin")  # rakans om av harnessen
np.array([1.0], np.float32).tofile(DST / "t.bin")
print(f"[prep] {PX}px  latent {LH}x{LH}  Ht={Ht} Nimg={Nimg} Scap={Scap} Su={Su}  -> {DST}")
