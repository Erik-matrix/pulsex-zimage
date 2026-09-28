#!/usr/bin/env python3
"""Independent numpy reference for one z-image-turbo DiT block (AdaLN-Zero
transformer block), re-implemented from the math in dit_full_engine.cpp's
block() function (rmsnorm/adaLN/qk-norm/3-axis RoPE/SDPA/SwiGLU). Reads
weights from the GGUF file we already converted and verified, so the C++
ggml graph and this script consume the identical numbers -- the only
thing under test is whether the ggml graph reproduces the same math.

Writes flat float32 .bin files (no header) for the C++ side to load, plus
the expected output, so the two implementations can be diffed numerically.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import gguf

GGUF_PATH = Path(r"D:\prov\2026-09-22_zimage-gguf\z-image-turbo.gguf")
OUT_DIR = Path(r"D:\prov\2026-09-22_zimage-gguf\block_test")

DIM, NH, HD, HID = 3840, 30, 128, 10240
HALF = 64
EPS, THETA = 1e-5, 256.0
AXES = [32, 48, 48]


def rmsnorm(x, w):
    # x: [S,D], w: [D]
    ss = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ss + EPS)
    return (x * inv.astype(np.float32)) * w


def rope_fc(ids):
    # ids: [S,3] -> fcos,fsin: [S,HALF]
    S = ids.shape[0]
    fcos = np.zeros((S, HALF), dtype=np.float32)
    fsin = np.zeros((S, HALF), dtype=np.float32)
    off = 0
    for ax, dd in enumerate(AXES):
        nf = dd // 2
        j = np.arange(nf)
        fr = 1.0 / (THETA ** ((2.0 * j) / dd))  # [nf]
        ang = ids[:, ax:ax + 1].astype(np.float64) * fr[None, :]  # [S,nf]
        fcos[:, off:off + nf] = np.cos(ang)
        fsin[:, off:off + nf] = np.sin(ang)
        off += nf
    return fcos, fsin


def apply_rope(t, fcos, fsin):
    # t: [S,NH,HD] in-place-style, interleaved pairs (2i,2i+1)
    S = t.shape[0]
    t = t.reshape(S, NH, HALF, 2).copy()
    x0 = t[..., 0]
    x1 = t[..., 1]
    fc = fcos[:, None, :]
    fs = fsin[:, None, :]
    o0 = x0 * fc - x1 * fs
    o1 = x0 * fs + x1 * fc
    out = np.stack([o0, o1], axis=-1)
    return out.reshape(S, NH, HD)


def silu(x):
    return x / (1.0 + np.exp(-x))


def block(x, fcos, fsin, adaln, w, mod):
    S = x.shape[0]
    an1, an2 = w["attention_norm1.weight"], w["attention_norm2.weight"]
    fn1, fn2 = w["ffn_norm1.weight"], w["ffn_norm2.weight"]
    qn, kn = w["attention.q_norm.weight"], w["attention.k_norm.weight"]

    if mod:
        m = adaln @ w["adaLN_modulation.0.weight"].T + w["adaLN_modulation.0.bias"]  # [4*DIM]
        smsa = 1 + m[0:DIM]
        gmsa = np.tanh(m[DIM:2 * DIM])
        smlp = 1 + m[2 * DIM:3 * DIM]
        gmlp = np.tanh(m[3 * DIM:4 * DIM])
    else:
        smsa = gmsa = smlp = gmlp = None

    # attention half
    h = rmsnorm(x, an1)
    if mod:
        h = h * smsa
    qkv = h @ w["attention.qkv.weight"].T  # [S,3*DIM]
    q, k, v = qkv[:, :DIM], qkv[:, DIM:2 * DIM], qkv[:, 2 * DIM:]
    q = q.reshape(S, NH, HD)
    k = k.reshape(S, NH, HD)
    v = v.reshape(S, NH, HD)
    q = rmsnorm(q, qn)
    k = rmsnorm(k, kn)
    q = apply_rope(q, fcos, fsin)
    k = apply_rope(k, fcos, fsin)
    scl = 1.0 / np.sqrt(HD)
    # [NH,S,S]
    scores = np.einsum("shd,thd->hst", q, k) * scl
    scores = scores - scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p = p / p.sum(axis=-1, keepdims=True)
    attn = np.einsum("hst,thd->shd", p, v).reshape(S, DIM)
    aop = attn @ w["attention.out.weight"].T
    aon = rmsnorm(aop, an2)
    x = x + (gmsa * aon if mod else aon)

    # ffn half
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


def load_block_weights(reader, pfx):
    names = [
        "attention_norm1.weight", "attention_norm2.weight",
        "ffn_norm1.weight", "ffn_norm2.weight",
        "attention.q_norm.weight", "attention.k_norm.weight",
        "attention.qkv.weight", "attention.out.weight",
        "feed_forward.w1.weight", "feed_forward.w3.weight", "feed_forward.w2.weight",
    ]
    by_name = {t.name: t for t in reader.tensors}
    w = {}
    for n in names:
        full = f"{pfx}.{n}"
        t = by_name[full]
        arr = t.data.reshape(t.shape[::-1]).astype(np.float32)
        w[n] = arr
    ada_w_name = f"{pfx}.adaLN_modulation.0.weight"
    if ada_w_name in by_name:
        t = by_name[ada_w_name]
        w["adaLN_modulation.0.weight"] = t.data.reshape(t.shape[::-1]).astype(np.float32)
        t = by_name[f"{pfx}.adaLN_modulation.0.bias"]
        w["adaLN_modulation.0.bias"] = t.data.astype(np.float32)
    return w


def main():
    pfx = sys.argv[1] if len(sys.argv) > 1 else "layers.0"
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mod = not pfx.startswith("context_refiner.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ref] loading {GGUF_PATH}")
    reader = gguf.GGUFReader(str(GGUF_PATH))
    w = load_block_weights(reader, pfx)
    print(f"[ref] block={pfx} mod={mod} S={S} tensors_loaded={len(w)}")

    rng = np.random.default_rng(1234)
    x = rng.normal(0, 1.0, size=(S, DIM)).astype(np.float32)
    adaln = rng.normal(0, 1.0, size=(256,)).astype(np.float32) if mod else np.zeros(256, np.float32)
    # simple 2D grid ids for image tokens on axes 1,2; axis 0 (temporal/cap) = 0
    ids = np.zeros((S, 3), dtype=np.float32)
    for s in range(S):
        ids[s, 0] = 0
        ids[s, 1] = s // 4
        ids[s, 2] = s % 4
    fcos, fsin = rope_fc(ids)

    out = block(x, fcos, fsin, adaln, w, mod)

    x.tofile(OUT_DIR / "x0.bin")
    adaln.tofile(OUT_DIR / "adaln.bin")
    fcos.tofile(OUT_DIR / "fcos.bin")
    fsin.tofile(OUT_DIR / "fsin.bin")
    ids[:, 0].astype(np.int32).tofile(OUT_DIR / "ids0.bin")
    ids[:, 1].astype(np.int32).tofile(OUT_DIR / "ids1.bin")
    ids[:, 2].astype(np.int32).tofile(OUT_DIR / "ids2.bin")
    out.tofile(OUT_DIR / "ref_out.bin")
    with open(OUT_DIR / "meta.txt", "w") as f:
        f.write(f"{pfx}\n{S}\n{1 if mod else 0}\n")
    print(f"[ref] wrote inputs + ref_out.bin to {OUT_DIR}")
    print(f"[ref] out stats: mean={out.mean():.6f} std={out.std():.6f} max_abs={np.abs(out).max():.6f}")


if __name__ == "__main__":
    main()
