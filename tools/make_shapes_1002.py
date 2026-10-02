# 2026-10-02: the shapes the Z-Image window is going to offer, through zimage-make.exe - does each give a picture
# worth offering, and how long does it take (the window's first estimate)? Pictures to D:\prov (looked at by eye).
# NPU jobs: run only when no PulseX program is in use.      python make_shapes_1002.py [--only 512x512]
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BIN = Path(r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin")
OUT = Path(r"D:\prov\2026-10-02_zimage_gui")
ap = argparse.ArgumentParser()
ap.add_argument("--only", default="")
ap.add_argument("--prompt", default="a red fox sitting in fresh snow at the edge of a spruce forest, soft morning light")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

busy = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
for name in ("pulsex", "pulse_", "llama-server", "zimage", "ltx-", "flux-", "test-backend-ops"):
    if name in busy:
        sys.exit("a PulseX program is running (%s...) - not starting NPU jobs" % name)

OUT.mkdir(parents=True, exist_ok=True)
RUNS = [("512x512", 0), ("512x512", 1024), ("512x512", 2048), ("768x512", 0), ("512x768", 0), ("512x768", 1024), ("912x512", 0), ("912x512", 1024), ("1024x1024", 0),
        ("1024x1024", 2048)]
for size, up in RUNS:
    if a.only and a.only != size:
        continue
    out = OUT / ("fox_%s%s_s%d.jpg" % (size, "_up%d" % up if up else "", a.seed))
    cmd = [str(BIN / "zimage-make.exe"), a.prompt, "--config", str(HERE / "zimage.json"), "--size", size, "--seed", str(a.seed), "--out", str(out)]
    if up:
        cmd += ["--up", str(up)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(BIN))
    wall = time.time() - t0
    line = [ln for ln in r.stdout.split("\n") if ln.startswith("text ")]
    print("%-10s up %-4d rc %d  %5.1f s   %s%s" % (size, up, r.returncode, wall, line[0].strip() if line else "", "" if r.returncode == 0 else "\n   " + r.stderr[-500:]),
          flush=True)
