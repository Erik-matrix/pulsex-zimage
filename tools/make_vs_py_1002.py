# 2026-10-02: zimage-make.exe (C++, no Python) against zimage.py - the same prompt and the SAME start noise through
# both chains. zimage.py takes its noise from numpy's generator, which zimage-make does not rebuild, so the noise
# zimage.py would use for the seed is written to a file and handed to zimage-make with --noise-file.
#   1. the text sent, with and without --enrich (zimage-make --enrich-only against enrich.py + the prefix rule)
#   2. the picture at 512 without enlarging: zimage.py's own functions (cap_feats, build_inputs, run_dit,
#      taef1_decode) against zimage-make - compared as pixels (PNG, so that JPEG is not part of the question)
#   3. the same pair a second time through zimage.py = the known-good control: how much do two runs of the SAME
#      chain differ on this NPU? A difference between the chains means nothing without that number.
# NPU jobs: run only when no PulseX program is in use.      python make_vs_py_1002.py [--size 512] [--seed 4242]
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
ap = argparse.ArgumentParser()
ap.add_argument("--size", default="512")
ap.add_argument("--seed", type=int, default=4242)
ap.add_argument("--prompt", default="a red fox sitting in fresh snow at the edge of a spruce forest, soft morning light")
a = ap.parse_args()
sys.argv = sys.argv[:1]

busy = subprocess.run(["tasklist"], capture_output=True, text=True).stdout.lower()
for name in ("pulsex", "pulse_", "llama-server", "zimage-", "ltx-", "flux-", "test-backend-ops"):
    if name in busy:
        sys.exit("a PulseX program is running (%s...) - not starting NPU jobs" % name)

import zimage as z  # noqa: E402
from enrich import enrich as py_enrich  # noqa: E402
from PIL import Image  # noqa: E402

MAKE = Path(z.BIN) / "zimage-make.exe"
bad = []


def check(ok, what):
    print("   %s  %s" % ("ok  " if ok else "FAIL", what), flush=True)
    if not ok:
        bad.append(what)


def make(*args):
    r = subprocess.run([str(MAKE), "--config", str(z.CONFIG_FILE)] + [str(x) for x in args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(z.BIN))
    return r.returncode, r.stdout, r.stderr


# 1. the text sent
print("1. the text")
pre = str(z.DEFAULTS.get("prompt_prefix", "") or "")
for p in (a.prompt, "en älg i en snöig granskog under norrskenet", "renarna vandrar över höstfjället", "en ren skjorta på en galge",
          "L3N0V0, portrait of an old fisherman", "portrait of an old man with a felt hat and a grey beard, warm evening light"):
    for mode in ("off", "full"):
        want = py_enrich(p)[0] if mode == "full" else p
        rc, so, se = make(p, "--enrich", mode, "--enrich-only")
        check(rc == 0 and so.rstrip("\r\n") == want, "%-5s %s" % (mode, p[:60]) + ("" if so.rstrip("\r\n") == want else "\n         py : %s\n         c++: %s" % (want, so.strip())))

# 2. the picture
print("2. the picture (%s, seed %d)" % (a.size, a.seed))
px = z.parse_size(a.size)
w, h = z.size_wh(px)
steps = int(z.DEFAULTS["steps"])
tmp = Path(tempfile.mkdtemp(prefix="zimage_make_vs_py_"))


def py_picture(tag):
    wd = tmp / tag
    wd.mkdir()
    t0 = time.time()
    cap = z.cap_feats(a.prompt, wd, False)
    z.build_inputs(cap, px, a.seed, wd)
    lat = z.run_dit(wd, steps, False)
    im = z.taef1_decode(lat, wd)
    return cap, lat, im, time.time() - t0, wd


cap1, lat1, im1, s1, wd1 = py_picture("py1")
print("   zimage.py: %.1f s, %d prompt tokens" % (s1, cap1.shape[0]))
noise = wd1 / "lat_init.f32"
out = tmp / "make.png"
t0 = time.time()
rc, so, se = make(a.prompt, "--size", a.size, "--seed", a.seed, "--noise-file", noise, "--out", out, "--keep", "--progress")
s_make = time.time() - t0
print("   " + so.strip().replace("\n", "\n   "))
check(rc == 0 and out.exists(), "zimage-make made a picture (%.1f s)%s" % (s_make, "" if rc == 0 else ": " + se[-600:]))
steps_seen = [ln for ln in so.split("\n") if ln.startswith("PROGRESS step")]
check(len(steps_seen) == steps + 1 and steps_seen[-1].strip() == "PROGRESS step %d %d" % (steps, steps), "every step was reported: %s" % ", ".join(s.strip()[9:] for s in steps_seen))
cap2, lat2, im2, s2, _ = py_picture("py2")


def same(x, y):
    return "IDENTICAL" if np.array_equal(x, y) else "cos %.9f" % (np.dot(x.ravel().astype(np.float64), y.ravel().astype(np.float64))
                                                                  / (np.linalg.norm(x.astype(np.float64)) * np.linalg.norm(y.astype(np.float64))))


def diff(x, y):
    d = np.abs(x.astype(np.int32) - y.astype(np.int32))
    return int(d.max()), float(d.mean())


if rc == 0 and out.exists():
    kept = [ln.split(": ", 1)[1].strip() for ln in so.split("\n") if ln.startswith("work folder kept")]
    mk = np.asarray(Image.open(out).convert("RGB"))
    check(mk.shape == im1.shape, "the same size: %s" % (mk.shape,))
    if kept:
        wdm = Path(kept[0])
        capm = np.fromfile(wdm / "cap_raw.bin", dtype=np.float32).reshape(-1, z.CAPD)
        latm = np.fromfile(wdm / "lat_out.f32", dtype=np.float32)
        check(capm.shape == cap1.shape, "the same number of prompt tokens: %d" % capm.shape[0])
        if capm.shape == cap1.shape:
            print("   text features: c++ against py %s (max diff %.3g); py against py %s (max diff %.3g)"
                  % (same(capm, cap1), np.abs(capm - cap1).max(), same(cap2, cap1), np.abs(cap2 - cap1).max()))
        for f in ("cap_ids0.bin", "cap_ids1.bin", "cap_ids2.bin", "img_ids0.bin", "img_ids1.bin", "img_ids2.bin", "lat_init.f32", "img_raw.bin", "t.bin"):
            check((wdm / f).read_bytes() == (wd1 / f).read_bytes(), "the engine's input %s is the same bytes" % f)
        # meta.txt: Python's write_text ends the line CRLF, zimage-make LF - the engine reads five numbers
        check((wdm / "meta.txt").read_text().split() == (wd1 / "meta.txt").read_text().split(), "the engine's input meta.txt holds the same five numbers")
        print("   latent: c++ against py %s (max diff %.3g of max %.3g); py against py %s (max diff %.3g)"
              % (same(latm, lat1), np.abs(latm - lat1).max(), np.abs(lat1).max(), same(lat2, lat1), np.abs(lat2 - lat1).max()))
        # the decoder alone: zimage-make's latent through zimage.py's decoder call
        dd = tmp / "dec"
        dd.mkdir()
        imd = z.taef1_decode(latm, dd)
        print("   decoder: zimage-make's latent decoded by zimage.py against zimage-make's picture: max %d, mean %.4f" % diff(imd, mk))
    mx, mean = diff(mk, im1)
    mx0, mean0 = diff(im2, im1)
    print("   pixels: c++ against py  max %d, mean %.4f   |   py against py (control)  max %d, mean %.4f" % (mx, mean, mx0, mean0))
    check(mx <= mx0 and mean <= mean0, "the two chains differ no more than two runs of one chain")
    Image.fromarray(np.concatenate([im1, mk, im2], axis=1)).save(tmp / "py_make_py.png")
    print("   side by side (zimage.py | zimage-make | zimage.py again): %s" % (tmp / "py_make_py.png"))
print("RESULT:", "the same picture" if not bad else "FAILED: " + "; ".join(bad))
sys.exit(1 if bad else 0)
