# 09-28 (512+up "mjukare framtoning, inte sa plastig"): ESRGAN ritar pals som hårda,
# malade kanter. Blanda i malupplosningen: ut = d*ESRGAN + (1-d)*LANCZOS(512 -> mal).
# Matt pa samma tiger (D:\prov\2026-09-28_blend): Laplace-varians 688 (d=1) / 310 (0,5) / 156 (0).
# zimage.json defaults.upscale_detail, forval 0.5.
import io, json
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
JS = [r"C:\PulseCore\PulseX\zimage_dit\zimage.json", r"C:\PulseCore\PulseX\zimage_dit\zimage.example.json"]
s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit("py: ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


rep('''    img = Image.fromarray(big, "RGB")
    if target != img.size[0]:
        img = img.resize((target, target), Image.LANCZOS)
    return np.asarray(img)''', '''    img = Image.fromarray(big, "RGB")
    if target != img.size[0]:
        img = img.resize((target, target), Image.LANCZOS)
    # upscale_detail: ESRGAN's share. Pure ESRGAN paints fur and skin with hard, plastic-looking
    # edges; mixing in a plain Lanczos enlargement of the same 512 image softens that while most
    # of the detail stays (09-28, same tiger: Laplace variance 688 at 1.0, 310 at 0.5, 156 at 0).
    d = float(DEFAULTS.get("upscale_detail", 0.5))
    if d >= 1.0:
        return np.asarray(img)
    soft = np.asarray(Image.fromarray(im, "RGB").resize((target, target), Image.LANCZOS), dtype=np.float32)
    mix = d * np.asarray(img, dtype=np.float32) + (1.0 - d) * soft
    return np.clip(np.rint(mix), 0, 255).astype(np.uint8)''')
rep('"unload_after_s": 60, "purge_on_start": "standby", "purge_on_exit": "standby"}',
    '"unload_after_s": 60, "purge_on_start": "standby", "purge_on_exit": "standby",\n             "upscale_detail": 0.5}')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

for J in JS:
    j = json.load(open(J, encoding="utf-8"))
    nd = {}
    for k, v in j["defaults"].items():
        nd[k] = v
        if k == "upscale_to":
            nd["upscale_detail"] = 0.5
    j["defaults"] = nd
    if "_help" in j:
        j["_help"]["defaults.upscale_detail"] = ("0..1, how much of the Real-ESRGAN result is kept; the rest is a plain "
                                               "Lanczos enlargement. 1 = sharpest (can look plastic), 0 = softest.")
    io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
print("OK")
