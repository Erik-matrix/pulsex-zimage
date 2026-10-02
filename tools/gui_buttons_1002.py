# 2026-10-02: the Z-Image window's two buttons that work on the picture that is shown - "Add the text" (no NPU) and
# "Enlarge 2x" (the upscaler alone) - and the window in Swedish. Run after gui_check_1002.py --app zimage has put a
# picture in the sandbox; the newest one there is the picture shown. Each press must add ONE file of the expected
# size, and leave the picture it started from untouched.      python gui_buttons_1002.py
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

BIN = Path(r"C:\PulseCore\PulseX\ggml-hexagon\build-wos\bin")
OUT = Path(r"D:\prov\2026-10-02_zimage_gui\gui")
PICS, DATA = OUT / "sandbox" / "pictures", OUT / "sandbox" / "data"
ENV = dict(os.environ, PULSEX_ZIMAGE_OUT=str(PICS), PULSEX_ZIMAGE_DATA=str(DATA), ZIMAGE_CONFIG=r"C:\PulseCore\PulseX\zimage_dit\zimage.json")
bad = []


def check(ok, what):
    print("   %s  %s" % ("ok  " if ok else "FAIL", what), flush=True)
    if not ok:
        bad.append(what)


def pics():
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in PICS.iterdir() if p.suffix.lower() in (".jpg", ".png")}


def press(name, *args):
    before = pics()
    newest = max(before, key=lambda p: p.stat().st_mtime)
    r = subprocess.run([str(BIN / "pulsex-zimage.exe"), "--shot", str(OUT / (name + "_shot.png"))] + list(args), cwd=str(BIN), env=ENV, capture_output=True, text=True)
    after = pics()
    new = [p for p in after if p not in before]
    check(r.returncode == 0, "%s: the window ended with 0%s" % (name, "" if r.returncode == 0 else " (got %d: %s)" % (r.returncode, r.stderr[-300:])))
    check(len(new) == 1, "%s: one new file (%s)" % (name, ", ".join(p.name for p in new)))
    check(all(after[p] == h for p, h in before.items()), "%s: the pictures that were there are untouched" % name)
    return newest, (new[0] if new else None)


# the picture shown is the newest one: make that a picture small enough to enlarge (long side 1536 at most)
small = [p for p in pics() if max(Image.open(p).size) <= 1024]
if not small:
    sys.exit("no picture of 1024 or less in the sandbox - run gui_check_1002.py --app zimage --run ... first")
os.utime(max(small, key=lambda p: p.stat().st_mtime))
src, out = press("addtext", "--demo-addtext", "--demo-title", "Kalixälven", "--demo-sub", "Midnattssol")
if out:
    a, b = Image.open(src), Image.open(out)
    check(a.size == b.size, "addtext: the same size as the picture it was written on (%dx%d)" % b.size)
    check(list(a.convert("RGB").getdata()) != list(b.convert("RGB").getdata()), "addtext: the pixels changed (there is text)")
src, out = press("enlarge", "--demo-enlarge")
if out:
    a, b = Image.open(src), Image.open(out)
    check(b.size == (a.size[0] * 2, a.size[1] * 2), "enlarge: %dx%d became %dx%d" % (a.size + b.size))
r = subprocess.run([str(BIN / "pulsex-zimage.exe"), "--shot", str(OUT / "swedish_shot.png"), "--lang", "sv"], cwd=str(BIN), env=ENV)
check(r.returncode == 0 and (OUT / "swedish_shot.png").exists(), "the window in Swedish was drawn")
print("RESULT:", "both buttons work" if not bad else "FAILED: " + "; ".join(bad))
sys.exit(1 if bad else 0)
