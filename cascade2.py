# -*- coding: utf-8 -*-
"""Staged generation: generate low, then REFINE high. Not a partition of the schedule.

WHY THE FIRST ATTEMPT FAILED (cascade.py, kept for the record)
    It split N_STEPS across the stages: "512x3+1024x1" ran steps 0..2 at 512 and step 3
    at 1024. But the sigma schedule is extremely front-loaded,

        [1.000, 0.857, 0.601, 0.003, 0]

    so the high-resolution stage inherited sigma 0.003 -> 0. There was no noise left to
    denoise: that stage could only polish, never add detail. The cascade therefore cost
    extra time and produced nothing.

THE FIX: OVERLAP INSTEAD OF PARTITION
    Run the low stage to completion, upscale the latent, re-noise back to a meaningful
    level, and re-run the TAIL of the schedule at high resolution. That is the classic
    highres-fix / img2img refinement, and the high stage now does real work: it sees a
    plausible image at sigma ~0.6 and denoises it at a resolution where the model can
    actually resolve detail.

    Cost, with our measured linear-in-tokens scaling (1044 -> 4116 tokens, 26.3 -> 109.4 s):
        native 1024, 4 steps          4.0 units
        512 full (4) + 1024 tail (2)  1.0 + 2.0 = 3.0 units   = 25 % cheaper
    ...and the 512 stage fixes what 256 could not: our latent std is resolution dependent
    (256: 0.679, 512: 0.809, 1024: 1.074, target ~1.05), i.e. the model UNDER-DENOISES at
    low resolution. Starting from 256 inherits that; starting from 512 does not.

    python cascade2.py "<prompt>" [--from 512] [--to 1024] [--tail 2] [--strength auto]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import zimage as Z

BIN = Path(r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin\zimage-dit-stream.exe")
ENV = {**os.environ,
       "ADSP_LIBRARY_PATH": r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin",
       "ZI_FLASH": "1", "ZI_DUMP": "0"}

# Must match zimage_stream.cpp exactly: the re-noise strength below is read from THIS
# schedule, while the engine refines with ITS schedule. If they differ, the latent is
# re-noised to one sigma and denoised as if it were at another. Pass it explicitly so
# both sides read the same value, and default to the same thing the engine defaults to.
ZI_SIGMA = int(os.environ.get("ZI_SIGMA", "1"))
ENV["ZI_SIGMA"] = str(ZI_SIGMA)

N_STEPS = 4
SHIFT = 3.0


def sigmas(n=N_STEPS, shift=SHIFT):
    if ZI_SIGMA == 1:
        # [1 .9 .75 .5]: the last step does real work (see zimage_stream.cpp)
        raw = 1.0 - np.arange(n) / n
    else:
        raw = np.linspace(1.0, 1.0 / 1000.0, n)
    return list(shift * raw / (1.0 + (shift - 1.0) * raw)) + [0.0]


def upscale_latent(lat, px_from, px_to, sigma, seed):
    """Bilinear upscale, restore variance, then re-noise to `sigma`.

    Interpolation is a low-pass filter: it averages neighbours and therefore removes
    exactly the high-frequency part, which in a latent is largely the noise. Restoring the
    std first and re-noising second gives the model something that actually looks like the
    noise level the schedule claims we are at - otherwise it sees an implausibly clean
    input and barely moves.
    """
    import torch
    h_from, h_to = px_from // 8, px_to // 8
    x = torch.from_numpy(lat.reshape(1, Z.INCH, h_from, h_from))
    y = torch.nn.functional.interpolate(x, size=(h_to, h_to), mode="bilinear", align_corners=False)
    y = y * (x.std() / y.std().clamp_min(1e-8))          # restore variance
    if sigma > 0:
        g = torch.Generator().manual_seed(seed + 977)
        n = torch.randn(y.shape, generator=g, dtype=y.dtype)
        y = (1.0 - sigma) * y + sigma * n                # re-noise to the schedule's level
    return y.numpy().astype(np.float32).reshape(-1)


def run_stage(tmp, px, cap, seed, step_from, step_to, lat_init=None, label=""):
    d = Path(tmp) / ("st_%d_%d" % (px, step_from))
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    su = Z.build_inputs(cap, px, seed, d)
    if lat_init is not None:
        lat_init.tofile(d / "lat_init.f32")
    env = {**ENV, "ZI_STEPS": str(N_STEPS),
           "ZI_STEP_FROM": str(step_from), "ZI_STEP_TO": str(step_to)}
    t = time.perf_counter()
    r = subprocess.run([str(BIN), str(Z.DIT), str(d), "HTP0"], env=env,
                       capture_output=True, text=True)
    dt = time.perf_counter() - t
    if r.returncode != 0:
        print(r.stdout[-600:]); print(r.stderr[-600:])
        raise SystemExit("stage %s failed" % label)
    lat = np.fromfile(d / "lat_out.f32", dtype=np.float32)
    print("  %-22s %4d px  steps %d..%d  %6.1f s   std %.3f"
          % (label, px, step_from, step_to - 1, dt, lat.std()))
    return lat, su


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default="cascade.jpg")
    ap.add_argument("--base", type=int, default=512, help="resolution of the full stage")
    ap.add_argument("--final", type=int, default=1024, help="resolution of the refine stage")
    ap.add_argument("--tail", type=int, default=2,
                    help="how many schedule steps the refine stage re-runs (1..3)")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    sig = sigmas()
    step_from = N_STEPS - a.tail
    strength = sig[step_from]
    print("schedule %s" % ["%.3f" % s for s in sig])
    print("refine from step %d, i.e. re-noise to sigma %.3f" % (step_from, strength))

    tmp = Path(os.environ.get("TEMP", ".")) / "zimage_cascade"
    tmp.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cap = Z.cap_feats(a.prompt, tmp, False)
    lat, _ = run_stage(tmp, a.base, cap, a.seed, 0, N_STEPS, label="base (full)")
    up = upscale_latent(lat, a.base, a.final, strength, a.seed)
    lat, _ = run_stage(tmp, a.final, cap, a.seed, step_from, N_STEPS, up, label="refine")

    im = Z.decode(lat, tmp)
    from PIL import Image
    Image.fromarray(im).save(a.out, quality=95)
    print("[cascade] %s  %.1f s total -> %s" % (
        "%d+%d" % (a.base, a.final), time.perf_counter() - t0, a.out))


if __name__ == "__main__":
    main()
