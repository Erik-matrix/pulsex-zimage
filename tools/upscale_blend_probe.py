# 09-28 (512+up "behover lite mjukare framtoning och inte sa plastig"): samma 512-bild
# uppskalad med ESRGAN-andel s = 1.0 / 0.7 / 0.5 / 0.3 / 0.0 (resten LANCZOS), sida vid sida.
#   python tools/upscale_blend_probe.py <512.png> <utkatalog>
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zimage as Z

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
im = np.asarray(Image.open(src).convert("RGB"))
esr = Z.upscale_x4(im, 1024).astype(np.float32)
lz = np.asarray(Image.fromarray(im).resize((1024, 1024), Image.LANCZOS)).astype(np.float32)
levels = [1.0, 0.7, 0.5, 0.3, 0.0]
crops = []
for s in levels:
    b = np.clip(np.rint(s * esr + (1.0 - s) * lz), 0, 255).astype(np.uint8)
    Image.fromarray(b).save(out / ("blend_%03d.png" % int(s * 100)))
    g = b.astype(np.float32).mean(axis=2)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    print("s=%.1f  laplace-var %.0f" % (s, lap.var()))
    crops.append((s, Image.fromarray(b)))
# 1:1-utsnitt kring huvudet (samma ruta i alla)
box = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (200, 280, 600, 680)
w = box[2] - box[0]
sheet = Image.new("RGB", (w * len(crops), box[3] - box[1]))
for i, (s, c) in enumerate(crops):
    sheet.paste(c.crop(box), (i * w, 0))
    ImageDraw.Draw(sheet).text((i * w + 6, 6), "ESRGAN %d%%" % int(s * 100), fill=(255, 255, 0))
sheet.save(out / "blend_sheet.png")
print("sheet", out / "blend_sheet.png")
