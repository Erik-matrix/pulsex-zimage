#!/usr/bin/env python3
"""Independent numpy check of cap_embedder+context_refiner and x_embedder+
noise_refiner against the REAL golden reference data (dit_cond/), using the
same block()/rope_fc() logic already validated in ref_full.py. If this
matches golden, the bug is ggml-graph-specific; if it doesn't, the bug is in
our shared understanding of the math (patchify, embedder, or rope details).
"""
import json
from pathlib import Path

import numpy as np
import gguf

GGUF_PATH = Path(r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo.gguf")
SRC = Path(r"C:\PulseCore\models\zImage\dit_cond")

DIM, NH, HD, HID = 3840, 30, 128, 10240
HALF = 64
EPS, THETA = 1e-5, 256.0
AXES = [32, 48, 48]
NR = 2


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
        smsa = 1 + m[0:DIM]; gmsa = np.tanh(m[DIM:2 * DIM])
        smlp = 1 + m[2 * DIM:3 * DIM]; gmlp = np.tanh(m[3 * DIM:4 * DIM])
    h = rmsnorm(x, an1)
    if mod: h = h * smsa
    qkv = h @ w["attention.qkv.weight"].T
    q, k, v = qkv[:, :DIM], qkv[:, DIM:2 * DIM], qkv[:, 2 * DIM:]
    q, k, v = (t.reshape(S, NH, HD) for t in (q, k, v))
    q, k = rmsnorm(q, qn), rmsnorm(k, kn)
    q, k = apply_rope(q, fcos, fsin), apply_rope(k, fcos, fsin)
    scl = 1.0 / np.sqrt(HD)
    scores = np.einsum("shd,thd->hst", q, k) * scl
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores); p /= p.sum(axis=-1, keepdims=True)
    attn = np.einsum("hst,thd->shd", p, v).reshape(S, DIM)
    aop = attn @ w["attention.out.weight"].T
    aon = rmsnorm(aop, an2)
    x = x + (gmsa * aon if mod else aon)
    hf = rmsnorm(x, fn1)
    if mod: hf = hf * smlp
    w1o = hf @ w["feed_forward.w1.weight"].T
    w3o = hf @ w["feed_forward.w3.weight"].T
    inter = silu(w1o) * w3o
    w2o = inter @ w["feed_forward.w2.weight"].T
    w2n = rmsnorm(w2o, fn2)
    x = x + (gmlp * w2n if mod else w2n)
    return x


def load_block_weights(by_name, pfx, mod):
    names = ["attention_norm1.weight", "attention_norm2.weight", "ffn_norm1.weight", "ffn_norm2.weight",
             "attention.q_norm.weight", "attention.k_norm.weight", "attention.qkv.weight", "attention.out.weight",
             "feed_forward.w1.weight", "feed_forward.w3.weight", "feed_forward.w2.weight"]
    w = {}
    for n in names:
        t = by_name[f"{pfx}.{n}"]
        w[n] = t.data.reshape(t.shape[::-1]).astype(np.float32)
    if mod:
        t = by_name[f"{pfx}.adaLN_modulation.0.weight"]
        w["adaLN_modulation.0.weight"] = t.data.reshape(t.shape[::-1]).astype(np.float32)
        w["adaLN_modulation.0.bias"] = by_name[f"{pfx}.adaLN_modulation.0.bias"].data.astype(np.float32)
    return w


def main():
    meta = json.loads((SRC / "meta.json").read_text())
    Nimg, Scap, Himg, Wimg, Ht, Wt, PATCH, INCH, CAPD = (
        meta["Nimg"], meta["Scap"], meta["Himg"], meta["Wimg"], meta["Ht"], meta["Wt"], meta["PATCH"], meta["INCH"], meta["CAPD"])

    reader = gguf.GGUFReader(str(GGUF_PATH))
    by_name = {t.name: t for t in reader.tensors}

    def get(name):
        t = by_name[name]
        return t.data.reshape(t.shape[::-1]).astype(np.float32)

    latent = np.fromfile(SRC / "latent.f32", dtype=np.float32).reshape(INCH, Himg, Wimg)
    img_raw = np.zeros((Nimg, PATCH * PATCH * INCH), dtype=np.float32)
    for hi in range(Ht):
        for wi in range(Wt):
            for phi in range(PATCH):
                for pwi in range(PATCH):
                    for c in range(INCH):
                        img_raw[hi * Wt + wi, (phi * PATCH + pwi) * INCH + c] = latent[c, hi * PATCH + phi, wi * PATCH + pwi]

    cap_feats = np.fromfile(SRC / "cap_feats.f32", dtype=np.float32).reshape(Scap, CAPD)
    adaln = np.fromfile(SRC / "adaln.f32", dtype=np.float32)
    img_ids = np.fromfile(SRC / "img_ids.f32", dtype=np.float32).reshape(Nimg, 3)
    cap_ids = np.fromfile(SRC / "cap_ids.f32", dtype=np.float32).reshape(Scap, 3)

    # cap path
    cap = rmsnorm(cap_feats, get("cap_embedder.0.weight"))
    cap = cap @ get("cap_embedder.1.weight").T + get("cap_embedder.1.bias")
    cap.tofile(r"D:\prov\2026-09-22_zimage-gguf\golden_test\ref_cap_after_embed.f32")
    cfc, cfs = rope_fc(cap_ids)
    for r in range(NR):
        w = load_block_weights(by_name, f"context_refiner.{r}", mod=False)
        cap = block(cap, cfc, cfs, None, w, mod=False)

    gold_cap = np.fromfile(SRC / "mid_cap_after_cr.f32", dtype=np.float32).reshape(Scap, DIM)
    se = np.sum((cap.astype(np.float64) - gold_cap.astype(np.float64)) ** 2)
    sy = np.sum(gold_cap.astype(np.float64) ** 2)
    print(f"[ref] cap_after_cr relL2 vs golden = {np.sqrt(se / sy):.6f}   got_max={np.abs(cap).max():.4f} gold_max={np.abs(gold_cap).max():.4f}")

    # image path
    x = img_raw @ get("x_embedder.weight").T + get("x_embedder.bias")
    x.tofile(r"D:\prov\2026-09-22_zimage-gguf\golden_test\ref_x_after_embed.f32")
    ifc, ifs = rope_fc(img_ids)
    for r in range(NR):
        w = load_block_weights(by_name, f"noise_refiner.{r}", mod=True)
        x = block(x, ifc, ifs, adaln, w, mod=True)
        x.tofile(rf"D:\prov\2026-09-22_zimage-gguf\golden_test\ref_x_after_nr{r}.f32")

    gold_x = np.fromfile(SRC / "mid_x_after_nr.f32", dtype=np.float32).reshape(Nimg, DIM)
    se = np.sum((x.astype(np.float64) - gold_x.astype(np.float64)) ** 2)
    sy = np.sum(gold_x.astype(np.float64) ** 2)
    print(f"[ref] x_after_nr relL2 vs golden   = {np.sqrt(se / sy):.6f}   got_max={np.abs(x).max():.4f} gold_max={np.abs(gold_x).max():.4f}")

    # full 30-layer chain with REAL data, check for NaN/Inf
    uni = np.concatenate([x, cap], axis=0)
    uni_ids = np.concatenate([img_ids, cap_ids], axis=0)
    ufc, ufs = rope_fc(uni_ids)
    for l in range(30):
        w = load_block_weights(by_name, f"layers.{l}", mod=True)
        uni = block(uni, ufc, ufs, adaln, w, mod=True)
        nnan = np.isnan(uni).sum()
        ninf = np.isinf(uni).sum()
        finite = uni[np.isfinite(uni)]
        mx = np.abs(finite).max() if finite.size else float("nan")
        print(f"[ref] layers.{l:<2d}  nan={nnan} inf={ninf} max_finite={mx:.4f}")
        if l == 0:
            gold_l0 = np.fromfile(SRC / "mid_uni_L0.f32", dtype=np.float32).reshape(-1, DIM)
            se = np.sum((uni.astype(np.float64) - gold_l0.astype(np.float64)) ** 2)
            sy = np.sum(gold_l0.astype(np.float64) ** 2)
            print(f"[ref]   layers.0 vs golden mid_uni_L0 relL2={np.sqrt(se / sy):.6f}")


if __name__ == "__main__":
    main()
