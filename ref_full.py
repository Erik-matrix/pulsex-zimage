#!/usr/bin/env python3
"""Independent numpy reference for the FULL z-image-turbo DiT block chain:
cap_embedder -> context_refiner x2 (mod=False) in parallel with
x_embedder -> noise_refiner x2 (mod=True), concat, then layers x30 (mod=True).
Matches dit_full_engine.cpp's forward() wiring (rows ~2179-2330): image ids
axis0=Scap+1 constant, axis1/2=y/x grid; cap ids axis0=1..Scap, axis1/2=0.

Unlike ref_block.py's single-block test (raw N(0,1) noise into an interior
block, which produced an out-of-distribution SwiGLU outlier), inputs here are
realistic: cap/img activations come from feeding random RAW inputs through
the model's own embedder weights, so downstream magnitudes match what a real
generation step would actually see.
"""
import sys
from pathlib import Path

import numpy as np
import gguf

GGUF_PATH = Path(r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo.gguf")
OUT_DIR = Path(r"D:\prov\2026-09-22_zimage-gguf\full_test")

DIM, NH, HD, HID = 3840, 30, 128, 10240
HALF = 64
EPS, THETA = 1e-5, 256.0
AXES = [32, 48, 48]
NL, NR, Scap, CAPD, PATCH_IN = 30, 2, 32, 2560, 64


def rmsnorm(x, w):
    ss = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ss + EPS)
    return (x * inv.astype(np.float32)) * w


def rope_fc(ids):
    S = ids.shape[0]
    fcos = np.zeros((S, HALF), dtype=np.float32)
    fsin = np.zeros((S, HALF), dtype=np.float32)
    off = 0
    for ax, dd in enumerate(AXES):
        nf = dd // 2
        j = np.arange(nf)
        fr = 1.0 / (THETA ** ((2.0 * j) / dd))
        ang = ids[:, ax:ax + 1].astype(np.float64) * fr[None, :]
        fcos[:, off:off + nf] = np.cos(ang)
        fsin[:, off:off + nf] = np.sin(ang)
        off += nf
    return fcos, fsin


def apply_rope(t, fcos, fsin):
    S = t.shape[0]
    t = t.reshape(S, NH, HALF, 2).copy()
    x0, x1 = t[..., 0], t[..., 1]
    fc, fs = fcos[:, None, :], fsin[:, None, :]
    o0 = x0 * fc - x1 * fs
    o1 = x0 * fs + x1 * fc
    return np.stack([o0, o1], axis=-1).reshape(S, NH, HD)


def silu(x):
    return x / (1.0 + np.exp(-x))


def block(x, fcos, fsin, adaln, w, mod):
    S = x.shape[0]
    an1, an2 = w["attention_norm1.weight"], w["attention_norm2.weight"]
    fn1, fn2 = w["ffn_norm1.weight"], w["ffn_norm2.weight"]
    qn, kn = w["attention.q_norm.weight"], w["attention.k_norm.weight"]

    if mod:
        m = adaln @ w["adaLN_modulation.0.weight"].T + w["adaLN_modulation.0.bias"]
        smsa = 1 + m[0:DIM]
        gmsa = np.tanh(m[DIM:2 * DIM])
        smlp = 1 + m[2 * DIM:3 * DIM]
        gmlp = np.tanh(m[3 * DIM:4 * DIM])

    h = rmsnorm(x, an1)
    if mod:
        h = h * smsa
    qkv = h @ w["attention.qkv.weight"].T
    q, k, v = qkv[:, :DIM], qkv[:, DIM:2 * DIM], qkv[:, 2 * DIM:]
    q, k, v = (t.reshape(S, NH, HD) for t in (q, k, v))
    q, k = rmsnorm(q, qn), rmsnorm(k, kn)
    q, k = apply_rope(q, fcos, fsin), apply_rope(k, fcos, fsin)
    scl = 1.0 / np.sqrt(HD)
    scores = np.einsum("shd,thd->hst", q, k) * scl
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    attn = np.einsum("hst,thd->shd", p, v).reshape(S, DIM)
    aop = attn @ w["attention.out.weight"].T
    aon = rmsnorm(aop, an2)
    x = x + (gmsa * aon if mod else aon)

    hf = rmsnorm(x, fn1)
    if mod:
        hf = hf * smlp
    w1o = hf @ w["feed_forward.w1.weight"].T
    w3o = hf @ w["feed_forward.w3.weight"].T
    inter = silu(w1o) * w3o
    w2o = inter @ w["feed_forward.w2.weight"].T
    w2n = rmsnorm(w2o, fn2)
    x = x + (gmlp * w2n if mod else w2n)
    return x


def load_block_weights(by_name, pfx, mod):
    names = [
        "attention_norm1.weight", "attention_norm2.weight",
        "ffn_norm1.weight", "ffn_norm2.weight",
        "attention.q_norm.weight", "attention.k_norm.weight",
        "attention.qkv.weight", "attention.out.weight",
        "feed_forward.w1.weight", "feed_forward.w3.weight", "feed_forward.w2.weight",
    ]
    w = {}
    for n in names:
        t = by_name[f"{pfx}.{n}"]
        w[n] = t.data.reshape(t.shape[::-1]).astype(np.float32)
    if mod:
        w["adaLN_modulation.0.weight"] = by_name[f"{pfx}.adaLN_modulation.0.weight"].data.reshape(
            by_name[f"{pfx}.adaLN_modulation.0.weight"].shape[::-1]).astype(np.float32)
        w["adaLN_modulation.0.bias"] = by_name[f"{pfx}.adaLN_modulation.0.bias"].data.astype(np.float32)
    return w


def main():
    Ht = Wt = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    Nimg = Ht * Wt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ref] loading {GGUF_PATH}")
    reader = gguf.GGUFReader(str(GGUF_PATH))
    by_name = {t.name: t for t in reader.tensors}

    def get(name):
        t = by_name[name]
        return t.data.reshape(t.shape[::-1]).astype(np.float32)

    rng = np.random.default_rng(7)

    # ---- caption path: random raw text-encoder-shaped input -> cap_embedder -> context_refiner x2 ----
    cap_raw = rng.normal(0, 1.0, size=(Scap, CAPD)).astype(np.float32)
    cap0_w = get("cap_embedder.0.weight")  # RMSNorm weight, not a plain scale (dit_full_engine.cpp:2893)
    cap_normed = rmsnorm(cap_raw, cap0_w)
    cap = cap_normed @ get("cap_embedder.1.weight").T + get("cap_embedder.1.bias")
    cap_ids = np.zeros((Scap, 3), dtype=np.float32)
    cap_ids[:, 0] = np.arange(1, Scap + 1)
    cfc, cfs = rope_fc(cap_ids)
    for r in range(NR):
        w = load_block_weights(by_name, f"context_refiner.{r}", mod=False)
        cap = block(cap, cfc, cfs, None, w, mod=False)
    print(f"[ref] cap after context_refiner: mean={cap.mean():.4f} std={cap.std():.4f} max={np.abs(cap).max():.4f}")

    # ---- image path: random raw patch pixels -> x_embedder -> noise_refiner x2 ----
    img_raw = rng.normal(0, 1.0, size=(Nimg, PATCH_IN)).astype(np.float32)
    x = img_raw @ get("x_embedder.weight").T + get("x_embedder.bias")
    img_ids = np.zeros((Nimg, 3), dtype=np.float32)
    for y in range(Ht):
        for xx in range(Wt):
            img_ids[y * Wt + xx] = [Scap + 1, y, xx]
    ifc, ifs = rope_fc(img_ids)
    adaln = rng.normal(0, 1.0, size=(256,)).astype(np.float32)
    for r in range(NR):
        w = load_block_weights(by_name, f"noise_refiner.{r}", mod=True)
        x = block(x, ifc, ifs, adaln, w, mod=True)
    print(f"[ref] x after noise_refiner: mean={x.mean():.4f} std={x.std():.4f} max={np.abs(x).max():.4f}")

    # ---- concat + main layers ----
    uni = np.concatenate([x, cap], axis=0)  # [Su, DIM]
    Su = uni.shape[0]
    uni_ids = np.concatenate([img_ids, cap_ids], axis=0)
    ufc, ufs = rope_fc(uni_ids)
    for l in range(NL):
        w = load_block_weights(by_name, f"layers.{l}", mod=True)
        uni = block(uni, ufc, ufs, adaln, w, mod=True)
        if l % 10 == 0 or l == NL - 1:
            print(f"[ref] after layers.{l}: mean={uni.mean():.5f} std={uni.std():.5f} max={np.abs(uni).max():.4f}")

    # ---- save everything the C++ side needs ----
    cap_raw.tofile(OUT_DIR / "cap_raw.bin")
    img_raw.tofile(OUT_DIR / "img_raw.bin")
    adaln.tofile(OUT_DIR / "adaln.bin")
    cap_ids[:, 0].astype(np.int32).tofile(OUT_DIR / "cap_ids0.bin")
    cap_ids[:, 1].astype(np.int32).tofile(OUT_DIR / "cap_ids1.bin")
    cap_ids[:, 2].astype(np.int32).tofile(OUT_DIR / "cap_ids2.bin")
    img_ids[:, 0].astype(np.int32).tofile(OUT_DIR / "img_ids0.bin")
    img_ids[:, 1].astype(np.int32).tofile(OUT_DIR / "img_ids1.bin")
    img_ids[:, 2].astype(np.int32).tofile(OUT_DIR / "img_ids2.bin")
    uni.astype(np.float32).tofile(OUT_DIR / "ref_out.bin")
    with open(OUT_DIR / "meta.txt", "w") as f:
        f.write(f"{Ht}\n{Wt}\n{Nimg}\n{Scap}\n{Su}\n")
    print(f"[ref] wrote everything to {OUT_DIR}, Su={Su}")


if __name__ == "__main__":
    main()
