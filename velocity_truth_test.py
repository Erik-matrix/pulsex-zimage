# -*- coding: utf-8 -*-
# ICKE-CIRKULART DiT-TEST. Vi konstruerar marksanningen sjalva:
#   x0 = riktig fotolatent (pa datamanifolden), eps ~ N(0,1), sigma kant
#   x_sigma = (1-sigma)*x0 + sigma*eps      (flow-matching-vagen)
#   sann hastighet  v_true = eps - x0       (dx/dsigma langs vagen)
# Kor var DiT pa x_sigma och jamfor mot v_true. Ingen numpy-referens inblandad.
import os, sys, subprocess, numpy as np
from pathlib import Path

EXE  = r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin\zimage-dit-stream.exe"
GGUF = sys.argv[1] if len(sys.argv) > 1 else r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo-q8_0.gguf"
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
DIR  = Path(r"D:\prov\2026-09-24_fresh-golden\stream_in")
OUT  = Path(r"D:\prov\2026-09-24_fresh-golden")
INCH, PATCH, LH = 16, 2, 64
Ht = Wt = LH // PATCH; Nimg = Ht * Wt
Scap = np.fromfile(DIR / "cap_raw.bin", np.float32).size // 2560
Su = Nimg + Scap

x0 = np.fromfile(OUT / "lat_photo.f32", np.float32).reshape(INCH, LH, LH)
rng = np.random.default_rng(7)
eps = rng.standard_normal((INCH, LH, LH)).astype(np.float32)
x_sig = ((1.0 - SIGMA) * x0 + SIGMA * eps).astype(np.float32)
v_true = (eps - x0).astype(np.float32)

def patchify(lat):
    l = lat.reshape(INCH, Ht, PATCH, Wt, PATCH)
    return np.ascontiguousarray(l.transpose(1, 3, 2, 4, 0).reshape(Nimg, PATCH * PATCH * INCH)).astype(np.float32)

img_ids = np.zeros((Nimg, 3), np.int32)
for y in range(Ht):
    img_ids[y*Wt:(y+1)*Wt, 0] = Scap + 1
    img_ids[y*Wt:(y+1)*Wt, 1] = y
    img_ids[y*Wt:(y+1)*Wt, 2] = np.arange(Wt)
for k in range(3): img_ids[:, k].astype(np.int32).tofile(DIR / f"img_ids{k}.bin")
(DIR / "meta.txt").write_text(f"{Ht} {Wt} {Nimg} {Scap} {Su}\n")

patchify(x_sig).tofile(DIR / "img_raw.bin")
t_norm = 1.0 - SIGMA                      # samma konvention som run_scheduler (t_norm=(1000-sigma*1000)/1000)
np.array([t_norm], np.float32).tofile(DIR / "t.bin")
print(f"[test] sigma={SIGMA} t_norm={t_norm:.3f} Su={Su}  |x0|std={x0.std():.3f} |x_sig|std={x_sig.std():.3f}")

os.environ['ZI_STEPS'] = '0'   # legacy: en forward -> stage-filer
subprocess.run([EXE, GGUF, str(DIR), "HTP0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
dit = np.fromfile(DIR / "got_final_out.f32", np.float32).reshape(INCH, LH, LH)

def cos(a, b): return float(np.dot(a.ravel(), b.ravel()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
print(f"[resultat] cos(+dit, v_true) = {cos(dit, v_true):+.4f}")
print(f"[resultat] cos(-dit, v_true) = {cos(-dit, v_true):+.4f}")
print(f"           |dit|std={dit.std():.3f}  |v_true|std={v_true.std():.3f}  kvot={dit.std()/v_true.std():.3f}")
print("  (en frisk DiT ska ge tydligt positiv cos i EN av riktningarna, och kvot ~1)")
