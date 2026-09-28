# -*- coding: utf-8 -*-
"""Ritar zimage.ico: mork rundad platta, gul blixt. Passar PulseX.

Ritas i 1024 px och nedskalas till ikonens fem storlekar - da blir kanterna rena aven
vid 16x16, vilket de inte blir om man ritar smatt direkt.

    python make_icon.py
"""
from PIL import Image, ImageDraw, ImageFilter

S = 1024                      # ritstorlek
R = int(S * 0.22)             # hornradie, ungefar som Windows appikoner

BG_TOP, BG_BOT = (28, 31, 42), (14, 15, 22)      # mork, lite bla
BOLT_TOP, BOLT_BOT = (255, 214, 92), (255, 176, 20)   # gul -> barnsten


def vertical_gradient(size, top, bottom):
    g = Image.new("RGB", (1, size), top)
    d = ImageDraw.Draw(g)
    for y in range(size):
        t = y / max(1, size - 1)
        d.point((0, y), tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)))
    return g.resize((size, size), Image.BILINEAR)


def rounded_mask(size, radius, supersample=4):
    m = Image.new("L", (size * supersample, size * supersample), 0)
    ImageDraw.Draw(m).rounded_rectangle(
        [0, 0, size * supersample - 1, size * supersample - 1],
        radius=radius * supersample, fill=255)
    return m.resize((size, size), Image.LANCZOS)


# Blixten, normaliserad till 0..1 och sedan skalad. Asymmetrisk med flit - en blixt som
# ar spegelsymmetrisk ser ut som ett Z.
BOLT = [(0.60, 0.06), (0.24, 0.55), (0.45, 0.55), (0.37, 0.94), (0.76, 0.43), (0.55, 0.43)]


def draw_bolt(size, supersample=4):
    """Returnerar en alfamask for blixten."""
    n = size * supersample
    m = Image.new("L", (n, n), 0)
    ImageDraw.Draw(m).polygon([(x * n, y * n) for x, y in BOLT], fill=255)
    return m.resize((size, size), Image.LANCZOS)


def build(size):
    bg = vertical_gradient(size, BG_TOP, BG_BOT).convert("RGBA")
    bg.putalpha(rounded_mask(size, int(size * 0.22)))

    bolt_mask = draw_bolt(size)

    # Mjukt sken bakom blixten sa den inte ser utstansad ut mot den morka plattan.
    glow = bolt_mask.filter(ImageFilter.GaussianBlur(max(1, size // 22)))
    glow_layer = Image.new("RGBA", (size, size), BOLT_BOT + (0,))
    glow_layer.putalpha(glow.point(lambda v: int(v * 0.45)))
    bg = Image.alpha_composite(bg, glow_layer)

    bolt = vertical_gradient(size, BOLT_TOP, BOLT_BOT).convert("RGBA")
    bolt.putalpha(bolt_mask)
    return Image.alpha_composite(bg, bolt)


master = build(S)
sizes = [16, 32, 48, 64, 128, 256]
imgs = [master.resize((n, n), Image.LANCZOS) for n in sizes]
imgs[-1].save("zimage.ico", format="ICO", sizes=[(n, n) for n in sizes])
master.resize((256, 256), Image.LANCZOS).save("zimage_icon_preview.png")
print("zimage.ico klar:", ", ".join("%dx%d" % (n, n) for n in sizes))
