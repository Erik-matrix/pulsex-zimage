# 09-28 ("de nyare modellerna kor 480p, 854x480 ... skruva NPU:n till 842x480?"):
# --size tar nu aven BxH och 480p/720p. Sidorna maste vara delbara med 16 (VAE x8, patch x2),
# sa 854x480 blir 848x480 (407 040 px, 1590 bildtoken mot 1024 vid 512 och 4096 vid 1024).
# Kvadrat forblir en int (512/1024) sa alla befintliga jamforelser (px == 512) haller.
#   python tools/patch_size_wh.py
import io

PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        raise SystemExit("py: ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


# 1) hjalpare + build_inputs
rep('''def build_inputs(cap, px, seed, workdir):
    """Skriver allt harnessen laser. cap_ids ar 1..Scap; img_ids0 ar Scap+1."""
    lh = px // VAEF
    ht = wt = lh // PATCH
''', '''# --- image size: 512 / 1024 (square, an int) or W x H, e.g. 848x480 (a tuple) -------------------
# Both sides must divide by 16 (VAE x8, patch x2): 854x480 is not possible, 848x480 is.
SIZE_ALIASES = {"480p": (848, 480), "720p": (1280, 720)}


def parse_size(v):
    """'512' / '1024' -> int, '848x480' / '480p' / '720p' -> (w, h). ValueError otherwise."""
    t = str(v).strip().lower()
    if t in SIZE_ALIASES:
        return SIZE_ALIASES[t]
    if "x" in t:
        w, h = (int(p) for p in t.split("x", 1))
    else:
        w = h = int(t)
    if w % 16 or h % 16 or not (256 <= w <= 2048 and 256 <= h <= 2048):
        raise ValueError("size must be 256..2048 per side and divisible by 16 (e.g. 848x480), got %s" % v)
    return w if w == h else (w, h)


def size_wh(px):
    return (px, px) if isinstance(px, int) else tuple(px)


def size_str(px):
    w, h = size_wh(px)
    return str(w) if w == h else "%dx%d" % (w, h)


_LAT_HW = None   # (lh, lw) of the latent build_inputs made; the VAEs read the shape from here


def lat_hw(lat):
    if _LAT_HW and _LAT_HW[0] * _LAT_HW[1] * INCH == lat.size:
        return _LAT_HW
    h = int(round((lat.size // INCH) ** 0.5))
    return h, h


def build_inputs(cap, px, seed, workdir):
    """Skriver allt harnessen laser. cap_ids ar 1..Scap; img_ids0 ar Scap+1."""
    global _LAT_HW
    w, h = size_wh(px)
    lh, lw = h // VAEF, w // VAEF
    ht, wt = lh // PATCH, lw // PATCH
    _LAT_HW = (lh, lw)
''')
rep('''    rng.standard_normal((INCH, lh, lh)).astype(np.float32).tofile(workdir / "lat_init.f32")''',
    '''    rng.standard_normal((INCH, lh, lw)).astype(np.float32).tofile(workdir / "lat_init.f32")''')

# 2) taef1 + full VAE: formen ur build_inputs
rep('''    H = int(round((lat.size // INCH) ** 0.5))
    fi = Path(tmp) / "taef1_lat.f32"
    fo = Path(tmp) / "taef1_out.rgb"
    lat.reshape(INCH, H, H).astype(np.float32).tofile(str(fi))''', '''    H, W = lat_hw(lat)
    fi = Path(tmp) / "taef1_lat.f32"
    fo = Path(tmp) / "taef1_out.rgb"
    lat.reshape(INCH, H, W).astype(np.float32).tofile(str(fi))''')
rep('''    r = subprocess.run([str(TAEF1X), str(fi), str(fo)], capture_output=True, text=True, env=env)''',
    '''    r = subprocess.run([str(TAEF1X), str(fi), str(fo), "--hw", "%dx%d" % (H, W)], capture_output=True, text=True, env=env)''')
rep('''    px = H * VAEF
    return np.fromfile(str(fo), dtype=np.uint8).reshape(px, px, 3)''',
    '''    return np.fromfile(str(fo), dtype=np.uint8).reshape(H * VAEF, W * VAEF, 3)''')
rep('''    H = int(round((lat.size // INCH) ** 0.5))
    z = (torch.from_numpy(lat.reshape(1, INCH, H, H)) / SCALE + SHIFT)''',
    '''    H, W = lat_hw(lat)
    z = (torch.from_numpy(lat.reshape(1, INCH, H, W)) / SCALE + SHIFT)''')

# 3) texter
rep('''    log(f"DiT {px}px, {steps} steps, {su} tokens", t)''', '''    log(f"DiT {size_str(px)}px, {steps} steps, {su} tokens", t)''')
rep('''    size = "%d x %d" % (px, px)''', '''    size = "%d x %d" % size_wh(px)''')
rep('''    print("    " + Y + "--size 512|1024" + O + "   resolution, or just " + Y + "--512" + O + " / " + Y + "--1024"
          + O + " (now " + str(a.size) + ")")''', '''    print("    " + Y + "--size 512|1024|480p" + O + "  resolution, or just " + Y + "--512" + O + " / " + Y + "--1024"
          + O + " / " + Y + "--480p" + O + " (848 x 480, wide) (now " + size_str(a.size) + ")")''')

# 4) repl-flaggor
rep('''                if w in ("--512", "--1024"):                   # kortform (09-28: "--1024" foll igenom)
                    px = int(w[2:]); k += 1; continue
                if w == "--size" and k + 1 < len(words) and words[k + 1] in ("512", "1024"):
                    px = int(words[k + 1]); k += 2; continue
                if w == "--size":
                    # Tyst svald ValueError har lat "--size N" se ut att fungera medan
                    # upplosningen stod kvar - anvandaren fick fel bild utan att veta om det.
                    print("  " + Y + "--size" + O + " takes 512 or 1024 - keeping " + str(px) + ".")
                    k += 2; continue''', '''                if w in ("--512", "--1024", "--480p", "--720p"):   # kortform (09-28: "--1024" foll igenom)
                    px = parse_size(w[2:]); k += 1; continue
                if w == "--size":
                    # Tyst svald ValueError har lat "--size N" se ut att fungera medan
                    # upplosningen stod kvar - anvandaren fick fel bild utan att veta om det.
                    try:
                        px = parse_size(words[k + 1] if k + 1 < len(words) else "")
                    except ValueError:
                        print("  " + Y + "--size" + O + " takes 512, 1024, 480p or WxH divisible by 16 (848x480)"
                              " - keeping " + size_str(px) + ".")
                    k += 2; continue''')
rep('''                print("  " + D + "(settings updated: size " + str(px) + ", upscale "''',
    '''                print("  " + D + "(settings updated: size " + size_str(px) + ", upscale "''')

# 5) argparse
rep('''    ap.add_argument("--size", type=int, default=int(DEFAULTS["size"]), choices=(512, 1024),
                    help="512 is ~3.5x cheaper than 1024 and good for trying prompts")''',
    '''    ap.add_argument("--size", type=parse_size, default=parse_size(DEFAULTS["size"]),
                    help="512, 1024, 480p (= 848x480) or WxH divisible by 16; 512 is ~3.5x cheaper than 1024")''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

# 6) zimage.ps1: -Size ar en strang; zimage.py validerar
p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"
for a, b in [
    ("    [ValidateSet(0, 512, 1024)] [int] $Size = 0,",
     "    [ValidatePattern('^(|0|512|1024|480p|720p|[0-9]+x[0-9]+)$')] [string] $Size = '',"),
    ("if (-not $Size)      { $Size      = [int] $D.size }",
     "if (-not $Size -or $Size -eq '0') { $Size = [string] $D.size }"),
    ("        @('-Size 512|1024',     'resolution of the generated image',                 \"now $Size\"),",
     "        @('-Size 512|1024|480p', 'resolution; 480p = 848 x 480 (wide, no upscaler)', \"now $Size\"),"),
    ("    if ($Size -ne 1024) {", "    if ($Size -ne '1024') {"),
]:
    if p.count(a) != 1:
        raise SystemExit("ps1: ankare %d ggr: %r" % (p.count(a), a[:70]))
    p = p.replace(a, b)
io.open(PS, "w", encoding="ascii", newline="").write(p)
print("OK")
