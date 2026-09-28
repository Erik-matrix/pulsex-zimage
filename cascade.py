# -*- coding: utf-8 -*-
# Kaskad: kor de tidiga (billiga) stegen pa lag upplosning, skala upp latenten, och kor
# de sista (dyra) stegen pa full upplosning. Sigma-schemat ar HELA n_steps i bada korningar,
# sa steg k pa hog upplosning fortsatter exakt dar steg k-1 pa lag slutade.
import os, shutil, subprocess, sys, time
import numpy as np, torch
from pathlib import Path

BIN  = r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin\zimage-dit-stream.exe"
GGUF = r"C:\PulseCore\models\zImage\z-image-turbo-q4_0.gguf"
BASE = Path(r"D:\prov\2026-09-26_size-compare")
ENV  = {**os.environ,
        "ADSP_LIBRARY_PATH": r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin",
        "ZI_FLASH": "1", "ZI_DUMP": "0"}

N_STEPS = 4
RENOISE = os.environ.get("CASC_RENOISE", "1") != "0"
_t = np.linspace(1.0, 1.0/1000.0, N_STEPS); _sh = 3.0
SIGMA = list(_sh*_t/(1.0+(_sh-1.0)*_t)) + [0.0]
plan = sys.argv[1] if len(sys.argv) > 1 else "512x3+1024x1"     # "<px>x<steg>+<px>x<steg>..."
etapper = []
for bit in plan.split("+"):
    px, k = bit.split("x"); etapper.append((int(px), int(k)))
assert sum(k for _, k in etapper) == N_STEPS, "stegen maste summera till %d" % N_STEPS

def latent_up(src_f32, px_from, px_to, sigma=None, seed=1234):
    # Bilinjar uppskalning slatar ut latenten och tar darmed bort brusdelen: std foll
    # 1,074 -> 0,757 och bilden blev suddig med vavmonster. Tva rattningar:
    #   1) aterstall variansen efter interpolationen
    #   2) ater-brusa till ratt sigma-niva (x_t = (1-s)*x0 + s*brus), sa modellen far
    #      nagot som ser ut som den brusniva schemat sager att vi ar pa
    lh_f, lh_t = px_from // 8, px_to // 8
    x = torch.from_numpy(np.fromfile(src_f32, np.float32).reshape(1, 16, lh_f, lh_f))
    y = torch.nn.functional.interpolate(x, size=(lh_t, lh_t), mode="bilinear", align_corners=False)
    y = y * (x.std() / y.std())                      # (1) bevara variansen
    if sigma is not None and sigma > 0:              # (2) ater-brusa
        g = torch.Generator().manual_seed(seed)
        n = torch.randn(y.shape, generator=g)
        y = (1.0 - sigma) * y + sigma * n
    return y.numpy().astype(np.float32).reshape(-1)

t0 = time.perf_counter(); steg = 0; forra = None
for i, (px, k) in enumerate(etapper):
    d = BASE / f"casc_{px}_{i}"
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(BASE / f"in_{px}", d)
    if forra is not None:                       # skala upp foregaende etapps latent
        latent_up(forra, etapper[i-1][0], px, sigma=SIGMA[steg] if RENOISE else None).tofile(d / "lat_init.f32")
    env = {**ENV, "ZI_STEPS": str(N_STEPS), "ZI_STEP_FROM": str(steg), "ZI_STEP_TO": str(steg + k)}
    t1 = time.perf_counter()
    r = subprocess.run([BIN, GGUF, str(d), "HTP0"], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-800:]); print(r.stderr[-800:]); sys.exit(1)
    dt = time.perf_counter() - t1
    bud = [l for l in r.stderr.splitlines() if "[budget]" in l]
    comp = bud[0].split("compute")[1].split("|")[0].strip() if bud else "?"
    print(f"  etapp {i}: {px}px steg {steg}..{steg+k-1}  {dt:5.1f} s vagg, compute {comp} s")
    steg += k; forra = d / "lat_out.f32"

out = BASE / f"lat_kaskad_{plan.replace('+','_')}.f32"
shutil.copy(forra, out)
print(f"[kaskad] {plan}: {time.perf_counter()-t0:.1f} s totalt -> {out.name}")
