# 09-28 ("exempel-bilder till GitHub - forsen, en kvinna, en katt, en man med hatt,
# norrlandsmotiv"): renderar varje motiv i nativ 1024 med tva seeds, sa den basta kan valjas.
# Kraver en varm encoder pa porten.  python tools/make_examples.py <utkatalog>
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ZI = Path(__file__).resolve().parents[1] / "zimage.py"
EXAMPLES = [
    ("rapids", "forsarna i kalixälven under midnattssolen", True),
    ("woman", "portrait of a young woman with freckles by a window, soft window light", True),
    ("cat", "a tabby cat on a windowsill, winter light, snow outside", True),
    ("man_hat", "portrait of an old man with a felt hat and a grey beard, warm evening light", True),
    ("moose", "en älg i en snöig granskog under norrskenet", True),
]
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["11", "4242"])]
done = []
for name, prompt, enrich in EXAMPLES:
    for seed in SEEDS:
        f = out / ("%s_s%d.png" % (name, seed))
        if not f.exists():
            cmd = [sys.executable, str(ZI), prompt, "--size", "1024", "--seed", str(seed), "-o", str(f)]
            cmd.append("--enrich" if enrich else "--no-enrich")
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            print(name, seed, "rc", r.returncode, flush=True)
        done.append((f, "%s s%d" % (name, seed)))
cols = len(SEEDS)
sheet = Image.new("RGB", (512 * cols, 512 * len(EXAMPLES)))
for k, (f, label) in enumerate(done):
    im = Image.open(f).convert("RGB").resize((512, 512), Image.LANCZOS)
    x, y = (k % cols) * 512, (k // cols) * 512
    sheet.paste(im, (x, y))
    ImageDraw.Draw(sheet).text((x + 6, y + 6), label, fill=(255, 255, 0))
sheet.resize((256 * cols, 256 * len(EXAMPLES)), Image.LANCZOS).save(out / "examples_sheet.jpg", quality=88)
print("sheet", out / "examples_sheet.jpg")
