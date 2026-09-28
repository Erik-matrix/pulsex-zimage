# zimage.py: (1) sokvagar och forval ur zimage.json, (2) --upscale = gamla ZImage-receptet for
# 512-laget (512 -> Real-ESRGAN x4 -> 2048 -> Lanczos -> 1024), (3) pedagogisk utskrift: numrerade
# steg med vad/var/tid, pa engelska. Filen ar LF. Backslash via chr(92) dar det behovs.
import io
P = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
s = io.open(P, encoding="utf-8", newline="").read()
assert "load_config" not in s


def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit("ankare %d ggr (vill %d): %r" % (n, count, old[:90]))
    s = s.replace(old, new)


# ---- 1. konfigurationen
rep('''# --- fasta sokvagar --------------------------------------------------------
MODELS = Path(r"C:\\PulseCore\\models\\zImage")
BIN    = Path(r"C:\\PulseCore\\PulseX\\ggml-hexagon\\build-wos\\bin")
DIT    = MODELS / "z-image-turbo-q4_0.gguf"
ENC    = MODELS / "qwen3-4b-zimage-q4_0.gguf"
TOK    = MODELS / "tokenizer"
VAE    = MODELS / "vae" / "ae.safetensors"
TAEF1X = Path(r"C:\\PulseCore\\src\\vae_engine\\taef1_decode.exe")
''', '''# --- sokvagar och forval: zimage.json (ZIMAGE_CONFIG pekar om) ---------------
# Relativa sokvagar tolkas mot json-filens katalog, {models} = paths.models. Saknas filen
# galler vardena nedan - samma som maskinen byggdes med.
HERE = Path(__file__).resolve().parent
_FALLBACK = {
    "paths": {
        "models": "C:/PulseCore/models/zImage",
        "dit": "{models}/z-image-turbo-q4_0.gguf",
        "encoder": "{models}/qwen3-4b-zimage-q4_0.gguf",
        "tokenizer": "{models}/tokenizer",
        "vae_full": "{models}/vae/ae.safetensors",
        "taef1": "C:/PulseCore/src/vae_engine/taef1_decode.exe",
        "bin": "C:/PulseCore/PulseX/ggml-hexagon/build-wos/bin",
        "esrgan_x4": "C:/PulseCore/models/sdxl-qnn/esrgan_x4.bin.wrap.onnx",
        "qnn_runtime": "C:/PulseCore/tools/qnn/2.46-runtime",
    },
    "defaults": {"out": "zimage.jpg", "size": 512, "steps": 4, "seed": 1234, "quality": 95,
                 "vae": "taef1", "upscale": False, "upscale_to": 1024},
    "server": {"port": 8099},
}


def load_config():
    p = Path(os.environ.get("ZIMAGE_CONFIG", str(HERE / "zimage.json")))
    cfg = {k: dict(v) for k, v in _FALLBACK.items()}
    if p.exists():
        user = json.loads(p.read_text(encoding="utf-8"))
        for k in cfg:
            cfg[k].update(user.get(k, {}))
    base = p.parent

    def res(v):
        q = Path(v)
        return q if q.is_absolute() else (base / q).resolve()

    models = res(cfg["paths"]["models"])
    paths = {k: (models if k == "models" else res(str(v).replace("{models}", str(models))))
             for k, v in cfg["paths"].items()}
    return paths, cfg["defaults"], cfg["server"], p


PATHS, DEFAULTS, SERVER, CONFIG_FILE = load_config()
MODELS = PATHS["models"]
BIN    = PATHS["bin"]
DIT    = PATHS["dit"]
ENC    = PATHS["encoder"]
TOK    = PATHS["tokenizer"]
VAE    = PATHS["vae_full"]
TAEF1X = PATHS["taef1"]
ESRGAN = PATHS["esrgan_x4"]
QNN_RT = PATHS["qnn_runtime"]
''')
rep('''SERVER_URL = os.environ.get("ZIMAGE_SERVER", "http://127.0.0.1:8099")''',
    '''SERVER_URL = os.environ.get("ZIMAGE_SERVER", "http://127.0.0.1:%d" % int(SERVER["port"]))''')

# ---- 2 + 3. ESRGAN, stegutskrift, ny serve() och main()
i = s.index("def serve(a, tmp):")
j = s.index('if __name__ == "__main__":')
NEW = r'''_ESR = None


def esrgan_session():
    """Real-ESRGAN x4 as a QNN context via ONNX Runtime's QNN EP - the old ZImage recipe for
    512-px images. Opened once per process (~2.3 s); in the REPL it stays warm."""
    global _ESR
    if _ESR is None:
        import onnxruntime as ort
        try:
            ort.disable_telemetry_events()
        except Exception:
            pass
        for d in (QNN_RT, Path(ort.__file__).parent / "capi"):
            if d.is_dir():
                os.add_dll_directory(str(d))
        so = ort.SessionOptions()
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        _ESR = ort.InferenceSession(str(ESRGAN), so, providers=["QNNExecutionProvider"],
                                    provider_options=[{"backend_path": str(QNN_RT / "QnnHtp.dll")}])
    return _ESR


def upscale_x4(im, target):
    """512 x 512 RGB (uint8) -> Real-ESRGAN x4 -> 2048 -> Lanczos -> target. Measured 09-27:
    3.4 s on the NPU, colours faithful (per-channel slope 0.998-1.010 vs Lanczos), more detail."""
    from PIL import Image
    if im.shape[0] != 512 or im.shape[1] != 512:
        raise ValueError("the x4 upscaler takes a 512 x 512 image, got %dx%d" % (im.shape[1], im.shape[0]))
    x = np.ascontiguousarray((im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    y = esrgan_session().run(None, {"image": x})[0]
    big = np.rint(np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(big, "RGB")
    if target != img.size[0]:
        img = img.resize((target, target), Image.LANCZOS)
    return np.asarray(img)


class Steps:
    """Numbered progress lines: what runs, where, and how long it took."""

    def __init__(self, n):
        self.i, self.n = 0, n

    def done(self, what, t0, note=""):
        self.i += 1
        dt = time.perf_counter() - t0
        print("  %s[%d/%d]%s %-40s %s%5.1f s%s%s" % (C_DIM, self.i, self.n, C_OFF, what, C_GREEN, dt, C_OFF,
                                                     ("   " + C_DIM + note + C_OFF) if note else ""), flush=True)


def save_image(im, out, quality):
    from PIL import Image
    out = Path(out)
    img = Image.fromarray(im)
    img.save(out, quality=quality) if out.suffix.lower() in (".jpg", ".jpeg") else img.save(out)
    return out


def describe(px, steps, seed, up, up_to):
    size = "%d x %d" % (px, px)
    return (C_CYAN + "zimage" + C_OFF + "  " + size + "  |  " + str(steps) + " steps  |  seed " + str(seed)
            + (("  |  upscale x4 -> " + str(up_to)) if up else ""))


VAE_NAME = {"taef1": "taef1, CPU", "full": "full VAE, CPU"}


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
    print("    " + Y + "--size 512|1024" + O + "   resolution (now " + str(a.size) + ")")
    print("    " + Y + "--up" + O + "              512 + Real-ESRGAN x4 -> " + str(a.upscale_to)
          + " (" + ("on" if a.upscale else "off") + ")")
    print("    " + Y + "--noup" + O + "            turn the upscaler off again")
    print("    " + Y + "--seed N" + O + "          start seed (each image adds 1)")
    print()
    print("  " + Y + ":q" + O + " or " + Y + "Ctrl+D" + O + " quits.   Images: " + D + os.getcwd() + O
          + " as " + Path(a.out).stem + "_NNNN" + Path(a.out).suffix)
    print()
    if a.vae == "full":
        vae_weights()                  # betala torch-laddningen nu, inte pa forsta prompten
    dit, n, px, up, seed0 = None, 0, a.size, a.upscale, a.seed
    try:
        while True:
            try:
                line = input(G + "> " + O)
                # Ctrl+D arrives as \x04 on a Windows console instead of raising EOFError.
                if "\x04" in line:
                    print(); print(D + "  bye - freeing NPU memory and cache..." + O); break
                line = line.strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if line in (":q", ":quit", "quit", "exit"):
                break
            if not line:
                continue
            words, prompt, bad = line.split(), [], False
            k = 0
            while k < len(words):
                w = words[k]
                if w == "--size" and k + 1 < len(words) and words[k + 1] in ("512", "1024"):
                    px = int(words[k + 1]); k += 2; continue
                if w == "--size":
                    # Tyst svald ValueError har lat "--size N" se ut att fungera medan
                    # upplosningen stod kvar - anvandaren fick fel bild utan att veta om det.
                    print("  " + Y + "--size" + O + " takes 512 or 1024 - keeping " + str(px) + "."); bad = True
                    k += 2; continue
                if w == "--seed" and k + 1 < len(words) and words[k + 1].lstrip("-").isdigit():
                    seed0 = int(words[k + 1]) - n; k += 2; continue
                if w in ("--up", "--upscale"):
                    up = True; k += 1; continue
                if w in ("--noup", "--no-upscale"):
                    up = False; k += 1; continue
                prompt.append(w); k += 1
            if not prompt:
                print("  " + D + "(settings updated: size " + str(px) + ", upscale " + ("on" if up else "off")
                      + " - now type a prompt)" + O)
                continue
            text = " ".join(prompt)
            do_up = up and px == 512
            if up and px != 512:
                print("  " + D + "note: the x4 upscaler works from 512; rendering 1024 natively." + O)
            n += 1
            out = Path(a.out).with_name(f"{Path(a.out).stem}_{n:04d}{Path(a.out).suffix}")
            st = Steps(4 if do_up else 3)
            t_img = time.perf_counter()
            try:
                t = time.perf_counter(); cap = cap_feats(text, tmp, a.verbose)
                st.done("encode the prompt  (Qwen3-4B, NPU)", t)
                su = build_inputs(cap, px, seed0 + n, tmp)
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
                out = save_image(im, out, a.quality)
                print("  " + G + "saved" + O + " %s  (%d x %d, seed %d)  in %.1f s" % (out, im.shape[1], im.shape[0], seed0 + n,
                                                                                    time.perf_counter() - t_img))
            except Exception as e:
                print(f"  failed: {e}")
    finally:
        if dit: dit.close()


def main():
    ap = argparse.ArgumentParser(prog="zimage", description="Make an image from a prompt on the NPU.")
    ap.add_argument("prompt", nargs="?", default="", help="the prompt (empty with --serve)")
    ap.add_argument("-o", "--out", default=DEFAULTS["out"], help="output file (.jpg or .png)")
    ap.add_argument("--size", type=int, default=int(DEFAULTS["size"]), choices=(512, 1024),
                    help="512 is ~3.5x cheaper than 1024 and good for trying prompts")
    ap.add_argument("--steps", type=int, default=int(DEFAULTS["steps"]), help="Z-Image-Turbo is distilled for 4")
    ap.add_argument("--seed", type=int, default=int(DEFAULTS["seed"]))
    ap.add_argument("-q", "--quality", type=int, default=int(DEFAULTS["quality"]), help="JPEG quality")
    ap.add_argument("--upscale", action="store_true", default=bool(DEFAULTS["upscale"]),
                    help="render 512, then Real-ESRGAN x4 (the old ZImage recipe)")
    ap.add_argument("--upscale-to", type=int, default=int(DEFAULTS["upscale_to"]), choices=(1024, 2048),
                    help="final size after the x4 upscaler")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--keep", metavar="DIR", help="keep the intermediate files here")
    ap.add_argument("--serve", action="store_true", help="interactive mode (REPL)")
    a = ap.parse_args()
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
        if a.upscale and a.size != 512:
            print(C_DIM + "note: the x4 upscaler works from 512; rendering 1024 natively." + C_OFF)
        print(describe(a.size, a.steps, a.seed, do_up, a.upscale_to))
        st = Steps(4 if do_up else 3)
        t = time.perf_counter(); cap = cap_feats(a.prompt, tmp, a.verbose)
        st.done("encode the prompt  (Qwen3-4B, NPU)", t)
        release_encoder()
        su = build_inputs(cap, a.size, a.seed, tmp)
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
        out = save_image(im, a.out, a.quality)
        print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)  in %.1f s" % (out, im.shape[1], im.shape[0],
                                                                            time.perf_counter() - t_all))
    finally:
        if not a.keep:
            import shutil; shutil.rmtree(tmp, ignore_errors=True)


'''
s = s[:i] + NEW + s[j:]
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
