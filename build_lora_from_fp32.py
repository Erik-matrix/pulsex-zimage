# build_lora_from_fp32.py - bake a Z-Image-Turbo LoRA into the NPU model the RIGHT way: add it to the ORIGINAL
# fp32 weights (Tongyi-MAI/Z-Image-Turbo, transformer/ folder) and quantize to Q4_0 ONCE.
# Why not simply add the LoRA to the Q4_0 model (merge_lora_q4.py): those weights already sit on the Q4_0 grid,
# and deltas far below half a step round straight back - attention and FFN kept about 3 % of the LoRA. From fp32
# the rounding is dithered, so the LoRA survives on average.
# Only the tensors the LoRA touches (180: per layer adaLN_modulation.0, attention.qkv, attention.out, feed_forward.w1-3)
# are rebuilt; everything else stays bit-identical to the base GGUF (written in place into a copy, same Q4_0 size).
# Build TWO models for a fair test: scale 0 (control: fp32 -> Q4_0, no LoRA) and scale 1 (with LoRA) -> they differ
# ONLY in the LoRA. Guard: each fp32 weight must match the base Q4_0 weight to Q4 noise (rel < 0.3), else the name
# mapping is wrong and it stops.
#   python build_lora_from_fp32.py <fp32_transformer_dir> <base.gguf> <out.gguf> [lora.safetensors] [scale=1.0]
import sys, json, shutil, time
from pathlib import Path
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT
from safetensors import safe_open

NL = 30


def targets():
    for n in range(NL):
        L = "layers.%d." % n
        yield L + "adaLN_modulation.0.weight", [L + "adaLN_modulation.0"]
        yield L + "attention.qkv.weight", [L + "attention.to_q", L + "attention.to_k", L + "attention.to_v"]   # rows [q|k|v]
        yield L + "attention.out.weight", [L + "attention.to_out.0"]
        for w in ("w1", "w2", "w3"):
            yield L + "feed_forward.%s.weight" % w, [L + "feed_forward." + w]


def main():
    src_dir, base, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    lora_p = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] not in ("-", "none") else None
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    wmap = json.loads((src_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    # 09-30: NO safe_open(framework="np") — it maps the whole 10 GB shard copy-on-write, and Windows charges COMMIT
    # for the entire view (two shards open = ~20 GB > the 19.6 GB limit -> "paging file too small", os error 1455).
    # Read just the tensor's bytes with seek + read instead.
    headers = {}
    def fp32(name):
        f = wmap[name + ".weight"]
        if f not in headers:
            with open(src_dir / f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little"); headers[f] = (json.loads(fh.read(n)), 8 + n)
        hdr, base_off = headers[f]
        m = hdr[name + ".weight"]; a, b = m["data_offsets"]
        with open(src_dir / f, "rb") as fh:
            fh.seek(base_off + a); raw = fh.read(b - a)
        if m["dtype"] == "F32":
            return np.frombuffer(raw, np.float32).reshape(m["shape"]).copy()
        if m["dtype"] == "BF16":
            return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
        sys.exit("unexpected dtype %s for %s" % (m["dtype"], name))
    lo = None
    if lora_p is not None and scale != 0.0:
        from safetensors.torch import load_file
        lo = {k: v.float().numpy() for k, v in load_file(str(lora_p)).items()}
    def delta(mod):
        k = "diffusion_model." + mod
        return lo[k + ".lora_B.weight"] @ lo[k + ".lora_A.weight"]
    t0 = time.time()
    shutil.copyfile(base, dst)
    print("copied %s -> %s (%.0f s); LoRA %s scale %.2f" % (base.name, dst.name, time.time() - t0,
          lora_p.name if lo is not None else "none", scale if lo is not None else 0.0), flush=True)
    r = gguf.GGUFReader(str(base))
    byname = {t.name: t for t in r.tensors}
    out = np.memmap(dst, dtype=np.uint8, mode="r+")
    rels, kept = [], []
    tl = list(targets())
    for i, (gname, mods) in enumerate(tl):
        t = byname[gname]
        assert t.tensor_type == QT.Q4_0, (gname, t.tensor_type)
        W = np.concatenate([fp32(m) for m in mods], 0) if len(mods) > 1 else fp32(mods[0])
        Wb = gguf.quants.dequantize(t.data, QT.Q4_0).astype(np.float32)
        assert W.shape == Wb.shape, (gname, W.shape, Wb.shape)
        rel = float(np.linalg.norm(W - Wb) / np.linalg.norm(W)); rels.append(rel)
        if rel > 0.3:
            sys.exit("MAPPING WRONG? %s: fp32 vs base Q4_0 rel %.3f (expected ~Q4 noise)" % (gname, rel))
        D = None
        if lo is not None:
            D = np.concatenate([delta(m) for m in mods], 0) * scale if len(mods) > 1 else delta(mods[0]) * scale
            W = W + D
        q = gguf.quants.quantize(W, QT.Q4_0)
        assert q.nbytes == t.data.nbytes
        out[t.data_offset:t.data_offset + q.nbytes] = q.reshape(-1).view(np.uint8)
        if D is not None:                                   # how much of the LoRA the Q4_0 weights carry now
            W0 = gguf.quants.dequantize(gguf.quants.quantize(W - D, QT.Q4_0), QT.Q4_0)
            Wq = gguf.quants.dequantize(q, QT.Q4_0)
            kept.append(float(((Wq - W0) * D).sum() / (D * D).sum()))
        if i % 30 == 0 or i == len(tl) - 1:
            print("[%3d/%d] %-38s fp32 vs base-Q4 rel %.3f%s" % (i + 1, len(tl), gname, rel,
                  ("  LoRA kept %.2f" % kept[-1]) if kept else ""), flush=True)
    out.flush(); del out
    print("done %.0f s | fp32 vs base-Q4 rel median %.3f max %.3f%s" % (time.time() - t0, np.median(rels), max(rels),
          (" | LoRA kept median %.2f (min %.2f)" % (np.median(kept), min(kept))) if kept else ""), flush=True)


if __name__ == "__main__":
    main()
