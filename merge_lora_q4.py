# merge_lora_q4.py - KEPT FOR REFERENCE, DO NOT USE: use build_lora_from_fp32.py. Adding a LoRA to weights that
# are already Q4_0 keeps only a few percent of it (small deltas round back to the same Q4_0 value).
# What it does: bake a Z-Image-Turbo LoRA straight into the Q4_0 DiT GGUF (the NPU engine has no runtime LoRA path):
#   W' = dequant_Q4_0(W) + scale * B @ A   ->   quantize_Q4_0(W')   written IN PLACE into a copy of the GGUF
# (Q4_0 size is unchanged, so every KV field, tensor order and offset stays identical).
# LoRA (ostris ai-toolkit, rank 16, no alpha tensors -> alpha = rank, scale 1) names -> our GGUF:
#   diffusion_model.layers.N.adaLN_modulation.0 -> layers.N.adaLN_modulation.0.weight
#   ... .attention.to_q / to_k / to_v            -> layers.N.attention.qkv.weight rows [q | k | v] (ref_block.py:90;
#                                                    RoPE is applied at run time, the weights are not permuted)
#   ... .attention.to_out.0                      -> layers.N.attention.out.weight
#   ... .feed_forward.w1/w2/w3                   -> layers.N.feed_forward.w1/w2/w3.weight
# Receipts per tensor: |delta| vs |W|, vs the Q4_0 step, and the extra rounding the requantisation adds.
#   python merge_lora_q4.py <lora.safetensors> <in.gguf> <out.gguf> [scale=1.0]
import sys, json, struct, shutil, time
from pathlib import Path
import numpy as np
import gguf
from gguf import GGMLQuantizationType as QT


def load_lora(path):
    import torch
    from safetensors.torch import load_file
    sd = load_file(str(path))
    return {k: v.float().numpy() for k, v in sd.items()}


def main():
    lora_p, src, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    lo = load_lora(lora_p)
    pre = "diffusion_model."
    # group: gguf tensor name -> list of (row_offset, B, A)
    plan = {}
    for k in lo:
        if not k.endswith(".lora_A.weight"):
            continue
        base = k[len(pre):-len(".lora_A.weight")]            # layers.N.attention.to_q
        A, B = lo[k], lo[k.replace("lora_A", "lora_B")]
        parts = base.split(".")
        L, mod = ".".join(parts[:2]), ".".join(parts[2:])
        if mod in ("attention.to_q", "attention.to_k", "attention.to_v"):
            row = {"attention.to_q": 0, "attention.to_k": 1, "attention.to_v": 2}[mod] * B.shape[0]
            plan.setdefault(L + ".attention.qkv.weight", []).append((row, B, A))
        elif mod == "attention.to_out.0":
            plan.setdefault(L + ".attention.out.weight", []).append((0, B, A))
        elif mod in ("adaLN_modulation.0", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"):
            plan.setdefault(L + "." + mod + ".weight", []).append((0, B, A))
        else:
            sys.exit("unmapped LoRA module: " + base)
    print("LoRA: %d modules -> %d GGUF tensors, scale %.2f" % (len(lo) // 2, len(plan), scale), flush=True)

    t0 = time.time()
    shutil.copyfile(src, dst)
    print("copied %s (%.0f s)" % (dst.name, time.time() - t0), flush=True)
    r = gguf.GGUFReader(str(src))
    byname = {t.name: t for t in r.tensors}
    missing = [n for n in plan if n not in byname]
    if missing:
        sys.exit("not in the GGUF: %s" % missing[:5])
    out = np.memmap(dst, dtype=np.uint8, mode="r+")
    rel_d, step_d, extra = [], [], []
    for i, (name, mods) in enumerate(sorted(plan.items())):
        t = byname[name]
        assert t.tensor_type == QT.Q4_0, (name, t.tensor_type)
        W = gguf.quants.dequantize(t.data, QT.Q4_0).astype(np.float32)        # [out, in]
        D = np.zeros_like(W)
        for row, B, A in mods:
            D[row:row + B.shape[0]] += scale * (B @ A)
        Wn = W + D
        q = gguf.quants.quantize(Wn, QT.Q4_0)
        assert q.nbytes == t.data.nbytes, (name, q.nbytes, t.data.nbytes)
        out[t.data_offset:t.data_offset + q.nbytes] = q.reshape(-1).view(np.uint8)
        Wq = gguf.quants.dequantize(q, QT.Q4_0)
        blk = np.abs(W.reshape(-1, 32)).max(1) / 8.0                          # Q4_0 step per block (approx)
        rel_d.append(np.linalg.norm(D) / np.linalg.norm(W))
        step_d.append(np.abs(D).mean() / blk.mean())
        extra.append(np.linalg.norm(Wq - Wn) / np.linalg.norm(D))
        if i % 30 == 0 or i == len(plan) - 1:
            print("[%3d/%d] %-40s |D|/|W| %.4f  mean|D|/Q4-step %.3f  requant-error/|D| %.2f" % (
                i + 1, len(plan), name, rel_d[-1], step_d[-1], extra[-1]), flush=True)
    out.flush(); del out
    print("done %.0f s | median |D|/|W| %.4f, mean|D| = %.2f Q4 steps, requant error = %.2f x |D| (median)" % (
        time.time() - t0, np.median(rel_d), np.median(step_d), np.median(extra)), flush=True)


if __name__ == "__main__":
    main()
