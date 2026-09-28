# -*- coding: utf-8 -*-
# HELA genereringen i numpy (F32, vår ref-DiT) — isolerar arch vs backend/quant.
# Samma setup som on-device: seed 1234, LH=32 (256px), shift 3.0, real cap_feats.
import numpy as np, gguf, sys
from pathlib import Path
import ref_full as rf

GGUF = r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo.gguf"   # F16 (numpy dequant->F32)
OUT  = Path(r"D:\prov\2026-09-24_fresh-golden")
LH   = int(sys.argv[1]) if len(sys.argv) > 1 else 32
N    = 8
INCH, PATCH, DIM, NR, NL, CAPD = 16, 2, rf.DIM, rf.NR, rf.NL, rf.CAPD
Ht = Wt = LH // PATCH; Nimg = Ht * Wt

reader = gguf.GGUFReader(GGUF); by = {t.name: t for t in reader.tensors}
def get(n):
    t = by[n]; return t.data.reshape(t.shape[::-1]).astype(np.float32)

def t_embedder(t):
    half = 128; fr = np.exp(-np.log(10000.0)*np.arange(half)/half); a = float(t)*fr
    tf = np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)
    h = tf @ get("t_embedder.mlp.0.weight").T + get("t_embedder.mlp.0.bias")
    h = (h/(1+np.exp(-h))).astype(np.float32)
    return (h @ get("t_embedder.mlp.2.weight").T + get("t_embedder.mlp.2.bias")).astype(np.float32)

def layernorm(x, eps=1e-6):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)

def final_layer(uni, adaln):
    sad = (adaln/(1+np.exp(-adaln))).astype(np.float32)        # silu
    scale = sad @ get("final_layer.adaLN_modulation.1.weight").T + get("final_layer.adaLN_modulation.1.bias")
    scale = 1.0 + scale
    xf = layernorm(uni) * scale[None, :]
    return xf @ get("final_layer.linear.weight").T + get("final_layer.linear.bias")   # [Su,64]

def patchify(lat):
    return np.ascontiguousarray(lat.reshape(INCH,Ht,PATCH,Wt,PATCH).transpose(1,3,2,4,0).reshape(Nimg, PATCH*PATCH*INCH)).astype(np.float32)
def unpatchify(tok):   # [Nimg,64] -> [INCH,LH,LH]
    return np.ascontiguousarray(tok.reshape(Ht,Wt,PATCH,PATCH,INCH).transpose(4,0,2,1,3).reshape(INCH,LH,LH)).astype(np.float32)

# ---- cap path (en gång) ----
cap_raw = np.fromfile(OUT/"real_cap_feats.f32",dtype=np.float32).reshape(-1,CAPD); Scap = cap_raw.shape[0]
cap = rf.rmsnorm(cap_raw, get("cap_embedder.0.weight")) @ get("cap_embedder.1.weight").T + get("cap_embedder.1.bias")
cap_ids = np.zeros((Scap,3),np.float32); cap_ids[:,0] = np.arange(1,Scap+1)
cfc,cfs = rf.rope_fc(cap_ids)
for r in range(NR): cap = rf.block(cap, cfc, cfs, None, rf.load_block_weights(by,f"context_refiner.{r}",False), False)
cap_w = [rf.load_block_weights(by,f"layers.{l}",True) for l in range(NL)]
nr_w  = [rf.load_block_weights(by,f"noise_refiner.{r}",True) for r in range(NR)]

img_ids = np.zeros((Nimg,3),np.float32)
for y in range(Ht):
    for x in range(Wt): img_ids[y*Wt+x] = [Scap+1, y, x]
Su = Nimg + Scap
uni_ids = np.concatenate([img_ids, cap_ids],0); ufc,ufs = rf.rope_fc(uni_ids)
ifc,ifs = rf.rope_fc(img_ids)

raw = np.linspace(1.0,1.0/N,N,dtype=np.float32); S=3.0
sig = np.concatenate([S*raw/(1+(S-1)*raw), [0.0]]).astype(np.float32); ts = sig[:-1]*1000
rng = np.random.default_rng(1234)
xlat = rng.standard_normal((INCH,LH,LH)).astype(np.float32)
print(f"[numpy] LH={LH} img={LH*8}px Su={Su} sigmas={np.round(sig,3).tolist()}")

import time
for i in range(N):
    t0=time.time(); tn = (1000.0-float(ts[i]))/1000.0; adaln = t_embedder(tn)
    x = patchify(xlat) @ get("x_embedder.weight").T + get("x_embedder.bias")
    for r in range(NR): x = rf.block(x, ifc, ifs, adaln, nr_w[r], True)
    uni = np.concatenate([x, cap],0)
    for l in range(NL): uni = rf.block(uni, ufc, ufs, adaln, cap_w[l], True)
    tok = final_layer(uni, adaln)[:Nimg]
    dit = unpatchify(tok)
    xlat = xlat + float(sig[i+1]-sig[i])*(-dit)
    print(f"[step {i}] tn={tn:.3f} |x|={np.linalg.norm(xlat):.1f} std={xlat.std():.3f} {time.time()-t0:.1f}s")

xlat.astype(np.float32).tofile(OUT/"latent_numpy.f32")
print(f"[ok] -> {OUT}\\latent_numpy.f32")
