# -*- coding: utf-8 -*-
# Scheduler-loop (FlowMatchEuler) i valfri upplösning. Driver som kör DiT-harnessen per steg.
# Anv: run_scheduler.py <dev> <N> <LATENT_H>   (LATENT_H=128 -> 1024px bild)
import os, sys, subprocess, numpy as np
from pathlib import Path

EXE  = r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin\zimage-dit-stream.exe"
DIR  = Path(r"D:\prov\2026-09-24_fresh-golden\stream_in")
OUT  = Path(r"D:\prov\2026-09-24_fresh-golden")
DEV  = sys.argv[1] if len(sys.argv) > 1 else "GPUOpenCL"
N    = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LH   = int(sys.argv[3]) if len(sys.argv) > 3 else 128        # latent H=W
GGUF = sys.argv[4] if len(sys.argv) > 4 else r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo-q8_0.gguf"
INCH, PATCH = 16, 2
Ht = Wt = LH // PATCH
Nimg = Ht * Wt
Scap = np.fromfile(DIR / "cap_raw.bin", np.float32).size // 2560
Su = Nimg + Scap

def patchify(lat):                                          # [INCH,LH,LH] -> [Nimg, 64]
    l = lat.reshape(INCH, Ht, PATCH, Wt, PATCH)
    return np.ascontiguousarray(l.transpose(1, 3, 2, 4, 0).reshape(Nimg, PATCH*PATCH*INCH)).astype(np.float32)

# img_ids + meta för denna upplösning (fixt över steg)
img_ids = np.zeros((Nimg, 3), np.int32)
for y in range(Ht):
    img_ids[y*Wt:(y+1)*Wt, 0] = Scap + 1
    img_ids[y*Wt:(y+1)*Wt, 1] = y
    img_ids[y*Wt:(y+1)*Wt, 2] = np.arange(Wt)
for k in range(3): img_ids[:, k].astype(np.int32).tofile(DIR / f"img_ids{k}.bin")
(DIR / "meta.txt").write_text(f"{Ht} {Wt} {Nimg} {Scap} {Su}\n")

NTRAIN = 1000                                  # num_train_timesteps
# HF/Qualcomm FlowMatchEuler: ts=linspace(N_train,1,N)/N_train -> t-endpoint = 1/N_train (EJ 1/N).
# Fel endpoint (1/N) lamnade sista steget som ett grovt 0,3->0-hopp = residualbrus/korn.
raw = np.linspace(1.0, 1.0/NTRAIN, N, dtype=np.float64)
SHIFT = 3.0                                    # scheduler_config: static shift, use_dynamic_shifting=false
shifted = SHIFT * raw / (1.0 + (SHIFT - 1.0) * raw)
sig = np.concatenate([shifted, [0.0]]).astype(np.float32)
ts  = sig[:-1] * 1000.0
print(f"[sched] sigmas(shift={SHIFT}) = {np.round(sig,3).tolist()}")
rng = np.random.default_rng(1234)
x = rng.standard_normal((INCH, LH, LH)).astype(np.float32)
print(f"[sched] dev={DEV} N={N} latent={LH}x{LH} img={LH*8}px Nimg={Nimg} Su={Su}")

import time
for i in range(N):
    t = float(ts[i]); t_norm = (1000.0 - t)/1000.0
    patchify(x).tofile(DIR / "img_raw.bin")
    np.array([t_norm], np.float32).tofile(DIR / "t.bin")
    t0=time.time()
    os.environ['ZI_STEPS'] = '0'   # legacy: en forward -> stage-filer
    subprocess.run([EXE, GGUF, str(DIR), DEV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    dit = np.fromfile(DIR / "got_final_out.f32", np.float32).reshape(INCH, LH, LH)
    x = x + float(sig[i+1] - sig[i]) * (-dit)
    print(f"[step {i}] t_norm={t_norm:.3f} |x|={np.linalg.norm(x):.1f} std={x.std():.3f} nan={np.isnan(x).any()} {time.time()-t0:.1f}s")

out = OUT / f"latent_sharp_{LH}.f32"
x.astype(np.float32).tofile(out)
print(f"[ok] slutlatent -> {out}")
