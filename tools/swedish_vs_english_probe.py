# 09-28 ("Funkar svenska?"): samma seed, svensk mot engelsk prompt, UTAN berikning,
# sa att bara spraket skiljer. Kraver en varm encoder pa porten.  python tools/swedish_vs_english_probe.py <ut>
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ZI = Path(__file__).resolve().parents[1] / "zimage.py"
PAIRS = [
    ("en älg i en snöig granskog under norrskenet", "a moose in a snowy spruce forest under the northern lights"),
    ("ett gammalt kafé med ljusslingor och en pappersstjärna i fönstret",
     "an old cafe with string lights and a paper star in the window"),
    ("en röd timmerstuga vid en sjö en vinterkväll", "a red timber cabin by a lake on a winter evening"),
]
tiles = []
for i, (sv, en) in enumerate(PAIRS):
    for lang, p in (("sv", sv), ("en", en)):
        f = out / ("p%d_%s.png" % (i, lang))
        r = subprocess.run([sys.executable, str(ZI), p, "--size", "512", "--seed", "4242", "--no-upscale",
                            "--no-enrich", "-o", str(f)], capture_output=True, text=True, encoding="utf-8")
        print(i, lang, "rc", r.returncode, (r.stdout.strip().splitlines() or ["?"])[-1][-60:])
        tiles.append((f, "%s: %s" % (lang, p)))
sheet = Image.new("RGB", (1024, 512 * len(PAIRS)))
for k, (f, label) in enumerate(tiles):
    im = Image.open(f).convert("RGB")
    x, y = (k % 2) * 512, (k // 2) * 512
    sheet.paste(im, (x, y))
    ImageDraw.Draw(sheet).text((x + 6, y + 6), label[:70], fill=(255, 255, 0))
sheet.resize((768, 384 * len(PAIRS)), Image.LANCZOS).save(out / "sv_en_sheet.jpg", quality=90)
print("sheet", out / "sv_en_sheet.jpg")
