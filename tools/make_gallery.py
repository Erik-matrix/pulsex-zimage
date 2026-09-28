# Renderar README-galleriet igen med EXAKT de kommandon som star under bilderna (09-28), sa att
# de gar att aterskapa aven efter att berikningstabellen andrats. Kraver en varm encoder.
#   python tools/make_gallery.py <utkatalog>
import subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ZI = Path(__file__).resolve().parents[1] / "zimage.py"
GALLERY = [
    ("rapids", ["forsarna i kalixälven under midnattssolen", "--seed", "11"]),
    ("moose", ["en älg i en snöig granskog under norrskenet", "--seed", "11"]),
    ("woman", ["portrait of a young woman with freckles by a window, soft window light", "--seed", "4242"]),
    ("cat", ["a tabby cat on a windowsill, winter light, snow outside", "--seed", "4242"]),
    ("man-with-hat", ["portrait of an old man with a felt hat and a grey beard, warm evening light", "--seed", "11"]),
    ("open-house", ["interiörbild av ett gammalt konditori med kakelugn, spetsgardiner och prinsesstårta på porslin",
                    "--seed", "11", "--title", "Öppet hus", "--subtitle", "Lördag 10–14"]),
]
EXTRA = [("cake-close", ["närbild av en prinsesstårta på ett porslinsfat", "--seed", "4242"])]
done = []
for name, args in GALLERY + EXTRA:
    f = out / (name + ".png")
    r = subprocess.run([sys.executable, str(ZI)] + args + ["--size", "1024", "--enrich", "-o", str(f)],
                       capture_output=True, text=True, encoding="utf-8")
    print(name, "rc", r.returncode, flush=True)
    done.append((f, name))
sheet = Image.new("RGB", (512 * 4, 512 * 2))
for k, (f, label) in enumerate(done):
    x, y = (k % 4) * 512, (k // 4) * 512
    sheet.paste(Image.open(f).convert("RGB").resize((512, 512), Image.LANCZOS), (x, y))
    ImageDraw.Draw(sheet).text((x + 6, y + 6), label, fill=(255, 0, 0))
sheet.resize((1024, 512), Image.LANCZOS).save(out / "gallery_sheet.jpg", quality=90)
print("sheet", out / "gallery_sheet.jpg")
