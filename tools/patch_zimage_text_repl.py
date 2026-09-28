# zimage.py: text pa bilden (gamla ZImage-affischen), egen radlasare (Ctrl+D avslutar direkt),
# ESRGAN-forvarmning i repl, --no-upscale, och --add-text for en befintlig bild.
# Ersatter allt fran "def save_image" till filens slut. LF. Backslash i C-strangar via chr(92) behovs ej
# har: allt nedan ar Python-kalla i en r-strang.
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
s = io.open(P, encoding="utf-8", newline="").read()
assert "def read_line" not in s
i = s.index("def save_image(im, out, quality):")
NEW = r'''# --- text on the image (the old ZImage poster text) ---------------------------------------------
# ** VARFOR TEXTEN LAGGS PA EFTERAT och inte skrivs av modellen (matt 2026-08-30 i gamla ZImage):
#    ett KORT ord i motivet fungerar ("VALKOMMEN" pa en skylt), men en affisch med rubrik + brodtext +
#    datum blev nonsens ("NORTYERN MPAIET", "39-84202") - modellen ritar TEXTUR SOM SER UT SOM TEXT.
#    Kompositering ger ratt stavning, skarpa kanter, och texten gar att ANDRA utan ny generering.
FONTS = [r"C:/Windows/Fonts/seguibl.ttf",    # Segoe UI Black
         r"C:/Windows/Fonts/segoeuib.ttf",   # Segoe UI Bold
         r"C:/Windows/Fonts/arialbd.ttf"]


def _font(px):
    from PIL import ImageFont
    for f in FONTS:
        try:
            return ImageFont.truetype(f, px)
        except Exception:
            pass
    return ImageFont.load_default()


def add_text(img, title, subtitle, pos="bottom"):
    """Title + subtitle on a PIL image, over a soft dark scrim so white text reads even on snow.
    Lines that do not fit shrink until they do (a long title used to be cut at the edge)."""
    from PIL import ImageDraw
    if not (title or subtitle):
        return img
    img = img.convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    marg = int(W * 0.055)
    width = W - 2 * marg

    def fit(text, start_px, min_px):
        f, px = _font(start_px), start_px
        while px > min_px:
            try:
                b = f.getbbox(text); w = b[2] - b[0]
            except Exception:
                w = len(text) * px * 0.55
            if w <= width:
                break
            px = int(px * 0.94); f = _font(px)
        return f, px

    ft, t_px = fit(title, max(18, int(W * 0.075)), max(12, int(W * 0.030))) if title else (None, 0)
    fs, s_px = fit(subtitle, max(12, int(W * 0.030)), max(9, int(W * 0.018))) if subtitle else (None, 0)
    gap = int(t_px * 0.28) if title else 0
    height = (t_px if title else 0) + (gap if title and subtitle else 0) + (s_px if subtitle else 0)
    band = height + 2 * marg
    y0 = 0 if pos == "top" else H - band
    step = max(1, band // 48)
    for k in range(0, band, step):
        t = k / max(1, band - 1)
        a = int(165 * ((1 - t) if pos == "top" else t) ** 1.35)
        d.rectangle([0, y0 + k, W, y0 + k + step], fill=(8, 12, 20, a))
    y = y0 + marg
    if title:
        d.text((marg, y), title, font=ft, fill=(255, 255, 255, 255),
               stroke_width=max(1, t_px // 26), stroke_fill=(0, 0, 0, 190))
        y += t_px + (gap if subtitle else 0)
    if subtitle:
        d.text((marg, y), subtitle, font=fs, fill=(232, 238, 246, 255),
               stroke_width=max(1, s_px // 22), stroke_fill=(0, 0, 0, 170))
    return img


def save_image(im, out, quality, title=None, subtitle=None, pos="bottom"):
    from PIL import Image
    out = Path(out)
    img = Image.fromarray(im) if not hasattr(im, "save") else im
    if title or subtitle:
        img = add_text(img, title, subtitle, pos)
    if out.suffix.lower() in (".jpg", ".jpeg"):
        # subsampling 0 = full colour resolution: text edges stay sharp (4:2:0 smears red/white text)
        img.save(out, quality=quality, subsampling=0 if (title or subtitle) else -1)
    else:
        img.save(out)
    return out


def describe(px, steps, seed, up, up_to):
    size = "%d x %d" % (px, px)
    return (C_CYAN + "zimage" + C_OFF + "  " + size + "  |  " + str(steps) + " steps  |  seed " + str(seed)
            + (("  |  upscale x4 -> " + str(up_to)) if up else ""))


VAE_NAME = {"taef1": "taef1, CPU", "full": "full VAE, CPU"}

# --- line input for the REPL -------------------------------------------------------------------
# Windows' console only hands Ctrl+D to input() AFTER Enter (as "\x04"), so "Ctrl+D quits" did not
# work. This reader takes the keys itself: Ctrl+D / Ctrl+Z quit at once, Ctrl+C too, Up/Down recall
# earlier lines, Left/Right/Home/End/Del edit, Esc clears. Long lines that wrap are redrawn correctly.
# Not a console (piped input, tests): plain input().
_HISTORY = []


def read_line(prompt, prompt_width):
    if os.name != "nt" or not sys.stdin.isatty():
        return input(prompt)
    import msvcrt, shutil
    cols = max(20, shutil.get_terminal_size((100, 30)).columns)
    buf, pos, hi = [], 0, len(_HISTORY)
    state = {"row": 0}

    def draw():
        if state["row"]:
            sys.stdout.write("\x1b[%dA" % state["row"])
        sys.stdout.write("\r" + prompt + "".join(buf) + "\x1b[J")
        end = prompt_width + len(buf)
        cur = prompt_width + pos
        end_row, cur_row = end // cols, cur // cols
        if end_row > cur_row:
            sys.stdout.write("\x1b[%dA" % (end_row - cur_row))
        sys.stdout.write("\x1b[%dG" % (cur % cols + 1))
        state["row"] = cur_row
        sys.stdout.flush()

    draw()
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            pos = len(buf); draw(); sys.stdout.write("\n"); sys.stdout.flush()
            line = "".join(buf)
            if line.strip() and (not _HISTORY or _HISTORY[-1] != line):
                _HISTORY.append(line)
            return line
        if ch in ("\x04", "\x1a"):                    # Ctrl+D, Ctrl+Z
            sys.stdout.write("\n"); sys.stdout.flush(); raise EOFError
        if ch == "\x03":                              # Ctrl+C
            sys.stdout.write("\n"); sys.stdout.flush(); raise KeyboardInterrupt
        if ch in ("\x00", "\xe0"):                    # arrows and friends
            k = msvcrt.getwch()
            if k == "K" and pos > 0: pos -= 1
            elif k == "M" and pos < len(buf): pos += 1
            elif k == "G": pos = 0
            elif k == "O": pos = len(buf)
            elif k == "S" and pos < len(buf): del buf[pos]
            elif k == "H" and hi > 0:
                hi -= 1; buf = list(_HISTORY[hi]); pos = len(buf)
            elif k == "P":
                if hi < len(_HISTORY) - 1:
                    hi += 1; buf = list(_HISTORY[hi])
                else:
                    hi = len(_HISTORY); buf = []
                pos = len(buf)
            draw(); continue
        if ch == "\x08":                              # Backspace
            if pos > 0:
                del buf[pos - 1]; pos -= 1
            draw(); continue
        if ch == "\x1b":                              # Esc
            buf, pos = [], 0; draw(); continue
        if ch < " ":
            continue
        buf.insert(pos, ch); pos += 1
        draw()


def _take_text(words, k):
    """The words after --title/--sub, up to the next --option."""
    out = []
    while k < len(words) and not words[k].startswith("--"):
        out.append(words[k]); k += 1
    return " ".join(out), k


def serve(a, tmp):
    """REPL: model, weights and NPU session stay warm between images."""
    Y, G, D, O = C_YELLOW, C_GREEN, C_DIM, C_OFF
    print()
    print(C_CYAN + "zimage - interactive mode" + O)
    print("  Everything stays loaded between images, so from the second image on you only")
    print("  pay for the generation itself.")
    print()
    print("  " + G + "type a prompt" + O + " and press Enter        " + D + "a lighthouse at dusk" + O)
    print("  add options at the end of the line    " + D + "a lighthouse at dusk --size 1024" + O)
    print()
    print("  " + D + "these stay in effect until you change them:" + O)
    print("    " + Y + "--size 512|1024" + O + "   resolution (now " + str(a.size) + ")")
    print("    " + Y + "--up" + O + " / " + Y + "--noup" + O + "      512 + Real-ESRGAN x4 -> " + str(a.upscale_to)
          + " on/off (now " + ("on" if a.upscale else "off") + ")")
    print("    " + Y + "--seed N" + O + "          start seed (each image adds 1)")
    print("  " + D + "these apply to this one image:" + O)
    print("    " + Y + "--title" + O + " words       text on the picture   " + D + "--title Summer Sale" + O)
    print("    " + Y + "--sub" + O + " words         smaller line under it " + D + "--sub Friday 6 pm" + O)
    print("    " + Y + "--top" + O + "               put the text at the top (bottom is default)")
    print()
    print("  " + Y + "Ctrl+D" + O + " or " + Y + ":q" + O + " quits.  Up/Down recalls earlier lines.  Images: "
          + D + os.getcwd() + O + " as " + Path(a.out).stem + "_NNNN" + Path(a.out).suffix)
    print()
    if a.vae == "full":
        vae_weights()                  # betala torch-laddningen nu, inte pa forsta prompten
    dit, n, px, up, seed0 = None, 0, a.size, a.upscale, a.seed
    try:
        while True:
            try:
                line = read_line(G + "> " + O, 2)
                if "\x04" in line:                     # piped input can still carry it
                    raise EOFError
                line = line.strip()
            except (EOFError, KeyboardInterrupt):
                print(D + "  bye - freeing NPU memory and cache..." + O); break
            if line in (":q", ":quit", "quit", "exit"):
                break
            if not line:
                continue
            words, prompt = line.split(), []
            title = subtitle = None
            tpos = "bottom"
            k = 0
            while k < len(words):
                w = words[k]
                if w == "--size" and k + 1 < len(words) and words[k + 1] in ("512", "1024"):
                    px = int(words[k + 1]); k += 2; continue
                if w == "--size":
                    # Tyst svald ValueError har lat "--size N" se ut att fungera medan
                    # upplosningen stod kvar - anvandaren fick fel bild utan att veta om det.
                    print("  " + Y + "--size" + O + " takes 512 or 1024 - keeping " + str(px) + ".")
                    k += 2; continue
                if w == "--seed" and k + 1 < len(words) and words[k + 1].lstrip("-").isdigit():
                    seed0 = int(words[k + 1]) - n; k += 2; continue
                if w in ("--up", "--upscale"):
                    up = True; k += 1; continue
                if w in ("--noup", "--no-upscale"):
                    up = False; k += 1; continue
                if w == "--title":
                    title, k = _take_text(words, k + 1); continue
                if w in ("--sub", "--subtitle"):
                    subtitle, k = _take_text(words, k + 1); continue
                if w == "--top":
                    tpos = "top"; k += 1; continue
                if w == "--bottom":
                    tpos = "bottom"; k += 1; continue
                if w.startswith("--"):
                    print("  " + D + "unknown option " + w + " - ignored" + O); k += 1; continue
                prompt.append(w); k += 1
            if not prompt:
                print("  " + D + "(settings updated: size " + str(px) + ", upscale " + ("on" if up else "off")
                      + " - now type a prompt)" + O)
                continue
            text = " ".join(prompt)
            do_up = up and px == 512
            n += 1
            out = Path(a.out).with_name(f"{Path(a.out).stem}_{n:04d}{Path(a.out).suffix}")
            st = Steps(4 if do_up else 3)
            t_img = time.perf_counter()
            try:
                t = time.perf_counter(); cap = cap_feats(text, tmp, a.verbose)
                st.done("encode the prompt  (Qwen3-4B, NPU)", t)
                su = build_inputs(cap, px, seed0 + n, tmp)
                if do_up and _ESR is None:
                    esrgan_prewarm()                          # sessionsstarten doljs bakom DiT:n
                t = time.perf_counter()
                if dit is None:
                    dit = DitServer(tmp, a.steps, a.verbose)      # forsta bilden gors av starten
                else:
                    dit.render(tmp)
                lat = np.fromfile(tmp / "lat_out.f32", dtype=np.float32)
                st.done("generate the image (Z-Image DiT, NPU)", t, "%d tokens, %d steps" % (su, a.steps))
                if not np.isfinite(lat).all():
                    print("  the latent contains NaN - skipping this one"); continue
                t = time.perf_counter(); im = decode(lat, tmp)
                st.done("decode to pixels   (%s)" % VAE_NAME.get(a.vae, a.vae), t)
                if do_up:
                    t = time.perf_counter(); im = upscale_x4(im, a.upscale_to)
                    st.done("upscale x4         (Real-ESRGAN, NPU)", t, "512 -> 2048 -> %d" % a.upscale_to)
                out = save_image(im, out, a.quality, title, subtitle, tpos)
                print("  " + G + "saved" + O + " %s  (%d x %d, seed %d)  in %.1f s" % (out, im.shape[1], im.shape[0], seed0 + n,
                                                                                    time.perf_counter() - t_img))
            except Exception as e:
                print(f"  failed: {e}")
    finally:
        if dit: dit.close()


def text_only(a):
    """zimage text: put a title/subtitle on an EXISTING image - no generation."""
    from PIL import Image
    src = Path(a.add_text)
    if not src.exists():
        sys.exit("zimage: no such image: %s" % src)
    if not (a.title or a.subtitle):
        sys.exit("zimage: give -Title and/or -Subtitle for the text")
    out = Path(a.out) if a.out_given else src.with_name(src.stem + "_text" + (src.suffix or ".jpg"))
    img = Image.open(src)
    out = save_image(img.convert("RGB"), out, a.quality, a.title, a.subtitle, a.text_pos)
    print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)" % (out, img.size[0], img.size[1]))


def main():
    ap = argparse.ArgumentParser(prog="zimage", description="Make an image from a prompt on the NPU.")
    ap.add_argument("prompt", nargs="?", default="", help="the prompt (empty with --serve)")
    ap.add_argument("-o", "--out", default=None, help="output file (.jpg or .png)")
    ap.add_argument("--size", type=int, default=int(DEFAULTS["size"]), choices=(512, 1024),
                    help="512 is ~3.5x cheaper than 1024 and good for trying prompts")
    ap.add_argument("--steps", type=int, default=int(DEFAULTS["steps"]), help="Z-Image-Turbo is distilled for 4")
    ap.add_argument("--seed", type=int, default=int(DEFAULTS["seed"]))
    ap.add_argument("-q", "--quality", type=int, default=int(DEFAULTS["quality"]), help="JPEG quality")
    ap.add_argument("--upscale", action="store_true", default=bool(DEFAULTS["upscale"]),
                    help="at 512: Real-ESRGAN x4 afterwards (the old ZImage recipe)")
    ap.add_argument("--no-upscale", dest="upscale", action="store_false", help="keep the 512 image as it is")
    ap.add_argument("--upscale-to", type=int, default=int(DEFAULTS["upscale_to"]), choices=(1024, 2048),
                    help="final size after the x4 upscaler")
    ap.add_argument("--title", default=None, help="text on the picture")
    ap.add_argument("--subtitle", default=None, help="smaller line under the title")
    ap.add_argument("--text-pos", default="bottom", choices=("top", "bottom"))
    ap.add_argument("--add-text", metavar="IMAGE", default=None, help="put the text on an existing image")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--keep", metavar="DIR", help="keep the intermediate files here")
    ap.add_argument("--serve", action="store_true", help="interactive mode (REPL)")
    a = ap.parse_args()
    a.out_given = a.out is not None
    if a.out is None:
        a.out = DEFAULTS["out"]
    if a.add_text:
        text_only(a)
        return
    a.vae = os.environ.get("ZI_VAE", DEFAULTS["vae"]).lower()
    os.environ["ZI_VAE"] = a.vae

    need = [DIT, ENC, TOK, BIN, TAEF1X if a.vae == "taef1" else VAE]
    if a.upscale:
        need += [ESRGAN, QNN_RT]
    missing = [p for p in need if not p.exists()]
    if missing:
        sys.exit("zimage: missing files (see %s):\n  %s" % (CONFIG_FILE, "\n  ".join(str(p) for p in missing)))

    t_all = time.perf_counter()
    tmp = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="zimage_"))
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        if a.serve:
            serve(a, tmp)
            return
        do_up = a.upscale and a.size == 512
        print(describe(a.size, a.steps, a.seed, do_up, a.upscale_to))
        st = Steps(4 if do_up else 3)
        t = time.perf_counter(); cap = cap_feats(a.prompt, tmp, a.verbose)
        st.done("encode the prompt  (Qwen3-4B, NPU)", t)
        release_encoder()
        su = build_inputs(cap, a.size, a.seed, tmp)
        if do_up:
            esrgan_prewarm()          # ~1 s sessionsstart doljs bakom DiT:n
        t = time.perf_counter(); lat, dit_finish = start_dit(tmp, a.steps, a.verbose)
        st.done("generate the image (Z-Image DiT, NPU)", t, "%d tokens, %d steps" % (su, a.steps))
        if not np.isfinite(lat).all():
            dit_finish()
            sys.exit("the latent contains NaN - aborting")
        # VAE:n (CPU) avkodar medan motorn river ned sina NPU-buffertar.
        t = time.perf_counter(); im = decode(lat, tmp)
        st.done("decode to pixels   (%s)" % VAE_NAME.get(a.vae, a.vae), t)
        t = time.perf_counter(); dit_finish()
        if a.verbose:
            log("DiT-nedstangning, vantan efter VAE:n", t)
        if do_up:
            # Efter dit_finish: motorns NPU-session ar stangd innan ESRGAN oppnar sin.
            t = time.perf_counter(); im = upscale_x4(im, a.upscale_to)
            st.done("upscale x4         (Real-ESRGAN, NPU)", t, "512 -> 2048 -> %d" % a.upscale_to)
        out = save_image(im, a.out, a.quality, a.title, a.subtitle, a.text_pos)
        print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)  in %.1f s" % (out, im.shape[1], im.shape[0],
                                                                            time.perf_counter() - t_all))
    finally:
        if not a.keep:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
'''
s = s[:i] + NEW
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
