# -*- coding: utf-8 -*-
# Färskt golden MED RIKTIG cap_feats + dumpar zimage-dit-stream:s indata till stream_in/.
# Golden (numpy, GGUF-vikter) = golden_out.f32 (uni efter 30 lager, = haressens ggml_out.bin).
import numpy as np, gguf
from pathlib import Path
import ref_full as rf

OUT = Path(r"D:\prov\2026-09-24_fresh-golden")
SD  = OUT / "stream_in"; SD.mkdir(parents=True, exist_ok=True)
DIM, NR, NL, PATCH_IN, CAPD = rf.DIM, rf.NR, rf.NL, rf.PATCH_IN, rf.CAPD

GGUF = r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo.gguf"  # F16 (numpy kan ej läsa Q8_0 rått)
reader = gguf.GGUFReader(GGUF); by = {t.name: t for t in reader.tensors}
def get(n):
    t = by[n]; return t.data.reshape(t.shape[::-1]).astype(np.float32)

T_VAL = 0.5   # timestep
def t_embedder(t):
    half = 128
    fr = np.exp(-np.log(10000.0) * np.arange(half) / half)
    a = float(t) * fr
    tf = np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)          # [256] sinusoidal
    h = tf @ get("t_embedder.mlp.0.weight").T + get("t_embedder.mlp.0.bias") # [1024]
    h = (h / (1.0 + np.exp(-h))).astype(np.float32)                          # SiLU
    return (h @ get("t_embedder.mlp.2.weight").T + get("t_embedder.mlp.2.bias")).astype(np.float32)  # [256]

# ---- RIKTIG cap_raw (stock penult) ----
cap_raw = np.fromfile(OUT / "real_cap_feats.f32", dtype=np.float32).reshape(-1, CAPD)
Scap = cap_raw.shape[0]
cap = rf.rmsnorm(cap_raw, get("cap_embedder.0.weight"))
cap = cap @ get("cap_embedder.1.weight").T + get("cap_embedder.1.bias")
cap_ids = np.zeros((Scap, 3), dtype=np.float32); cap_ids[:, 0] = np.arange(1, Scap + 1)
cfc, cfs = rf.rope_fc(cap_ids)
for r in range(NR):
    cap = rf.block(cap, cfc, cfs, None, rf.load_block_weights(by, f"context_refiner.{r}", mod=False), mod=False)

# ---- img-väg (syntetisk seedad, SAMMA som harness kör efter dump) ----
rng = np.random.default_rng(7)
Ht = Wt = 4; Nimg = Ht * Wt
img_raw = rng.normal(0, 1.0, size=(Nimg, PATCH_IN)).astype(np.float32)
x = img_raw @ get("x_embedder.weight").T + get("x_embedder.bias")
img_ids = np.zeros((Nimg, 3), dtype=np.float32)
for y in range(Ht):
    for xx in range(Wt): img_ids[y*Wt+xx] = [Scap+1, y, xx]
ifc, ifs = rf.rope_fc(img_ids)
adaln = t_embedder(T_VAL)   # RIKTIG adaln från timestep via t_embedder
print(f"[t_emb] t={T_VAL} -> adaln[256] std={adaln.std():.4f} maxabs={np.abs(adaln).max():.3f}")
for r in range(NR):
    x = rf.block(x, ifc, ifs, adaln, rf.load_block_weights(by, f"noise_refiner.{r}", mod=True), mod=True)

uni = np.concatenate([x, cap], axis=0)
uni_ids = np.concatenate([img_ids, cap_ids], axis=0)
ufc, ufs = rf.rope_fc(uni_ids)
for l in range(NL):
    uni = rf.block(uni, ufc, ufs, adaln, rf.load_block_weights(by, f"layers.{l}", mod=True), mod=True)
Su = uni.shape[0]

# ---- dumpa harness-indata (cap_raw = RÅ penult; harness gör embedder inline) ----
cap_raw.astype(np.float32).tofile(SD / "cap_raw.bin")
img_raw.astype(np.float32).tofile(SD / "img_raw.bin")
adaln.astype(np.float32).tofile(SD / "adaln.bin")
for k, nm in ((0,"cap_ids0"),(1,"cap_ids1"),(2,"cap_ids2")):
    cap_ids[:, k].astype(np.int32).tofile(SD / f"{nm}.bin")
for k, nm in ((0,"img_ids0"),(1,"img_ids1"),(2,"img_ids2")):
    img_ids[:, k].astype(np.int32).tofile(SD / f"{nm}.bin")
np.array([T_VAL], dtype=np.float32).tofile(SD / "t.bin")     # timestep för harness t_embedder
adaln.astype(np.float32).tofile(OUT / "adaln_ref.f32")       # numpy t_embedder-facit
open(SD / "meta.txt", "w").write(f"{Ht} {Wt} {Nimg} {Scap} {Su}\n")
uni.astype(np.float32).tofile(OUT / "golden_out.f32")
print(f"[ok] stream_in dumpad (Scap={Scap} Su={Su}), golden_out.f32 |out|={np.linalg.norm(uni):.2f} nan={np.isnan(uni).any()}")
