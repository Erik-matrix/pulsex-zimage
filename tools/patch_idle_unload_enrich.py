# (1) ComfyUI-modellen i repl (09-27): efter N s utan prompt (zimage.json unload_after_s,
#     forval 60) slapps DiT-servern (~4,9 GB committed) och ESRGAN-sessionen (~0,9 GB), och
#     deras filsidor toms ur standby-cachen (zimage.exe --drop-cache). Nasta bild laddar om.
# (2) Berikning som VAL: --enrich (repl: klistrig, --noenrich), -Enrich i zimage.ps1,
#     zimage.json enrich=false. Den skickade prompten visas alltid.
import io, json
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
J = r"C:\PulseCore\PulseX\zimage_dit\zimage.json"
s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        raise SystemExit("py: ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


# --- hjalpfunktioner fore read_line
rep(r'''def read_line(prompt, prompt_width):
''', r'''def commit_mb():
    """System committed memory in MB (GetPerformanceInfo), -1 if unknown."""
    try:
        import ctypes
        class _PI(ctypes.Structure):
            _fields_ = ([("cb", ctypes.c_ulong)]
                        + [(k, ctypes.c_size_t) for k in ("CommitTotal", "CommitLimit", "CommitPeak",
                           "PhysicalTotal", "PhysicalAvailable", "SystemCache", "KernelTotal",
                           "KernelPaged", "KernelNonpaged", "PageSize")]
                        + [(k, ctypes.c_ulong) for k in ("HandleCount", "ProcessCount", "ThreadCount")])
        p = _PI(); p.cb = ctypes.sizeof(p)
        ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
        return p.CommitTotal * p.PageSize / 2**20
    except Exception:
        return -1.0


def unload_models(dit):
    """ComfyUI-style unload between images: close the DiT server (its NPU buffers are ~4.9 GB
    COMMITTED), drop the ESRGAN session (~0.9 GB), then drop their file pages from Windows'
    standby cache. The next image reloads them. Returns (committed MB freed, cache MB dropped)."""
    global _ESR
    c0 = commit_mb()
    if dit is not None:
        dit.close()
    with _ESR_LOCK:
        _ESR = None
    import gc
    gc.collect()
    dropped = 0
    exe = HERE / "zimage.exe"
    files = [str(DIT)] + [f for f in [str(ESRGAN).replace(".wrap.onnx", "")] if Path(f).exists()]
    if exe.exists():
        try:
            r = subprocess.run([str(exe), "--drop-cache"] + files, capture_output=True, text=True, timeout=60)
            for l in r.stdout.split(chr(10)):
                if l.startswith("dropped "):
                    dropped = int(l.split()[1])
        except Exception:
            pass
    c1 = commit_mb()
    return (c0 - c1 if c0 >= 0 and c1 >= 0 else 0.0), dropped


def read_line(prompt, prompt_width, idle=None):
    """idle = (seconds, callback): after that long without a key the callback runs once and
    may return a line to show above the prompt (the REPL uses it to unload the models)."""
''')
rep(r'''    draw()
    while True:
        ch = msvcrt.getwch()
''', r'''    draw()
    t_idle = time.monotonic()
    while True:
        if idle is not None:
            while not msvcrt.kbhit():
                if idle[0] > 0 and time.monotonic() - t_idle >= idle[0]:
                    msg = idle[1]()
                    idle = None
                    if msg:
                        if state["row"]:
                            sys.stdout.write("\x1b[%dA" % state["row"])
                        sys.stdout.write("\r\x1b[J" + msg + "\n")
                        state["row"] = 0
                        draw()
                    break
                time.sleep(0.02)
        ch = msvcrt.getwch()
''')

# --- serve: hjalptext, idle-callback, berikning
rep(r'''    print("    " + Y + "--seed N" + O + "          start seed (each image adds 1)")
''', r'''    print("    " + Y + "--seed N" + O + "          start seed (each image adds 1)")
    print("    " + Y + "--enrich" + O + " / " + Y + "--noenrich" + O + " add material words, like the old ZImage (now "
          + ("on" if a.enrich else "off") + ")")
''')
rep(r'''    dit, n, px, up, seed0 = None, 0, a.size, a.upscale, a.seed
    try:
        while True:
            try:
                line = read_line(G + "> " + O, 2)
''', r'''    dit, n, px, up, seed0 = None, 0, a.size, a.upscale, a.seed
    enr = a.enrich
    unload_s = int(DEFAULTS.get("unload_after_s", 60) or 0)
    if unload_s > 0:
        print("  " + D + "After %d s without a prompt the image model is unloaded and its memory given back;"
              % unload_s + O)
        print("  " + D + "the next image then takes a few seconds longer." + O)
        print()

    def _idle():
        nonlocal dit
        freed, dropped = unload_models(dit)
        dit = None
        return (D + "  (idle %d s: image model unloaded - %.1f GB memory and %.1f GB cache given back)"
                % (unload_s, freed / 1024.0, dropped / 1024.0) + O)

    try:
        while True:
            try:
                busy = dit is not None or _ESR is not None
                line = read_line(G + "> " + O, 2, (unload_s, _idle) if (busy and unload_s > 0) else None)
''')
rep(r'''                if w == "--top":
                    tpos = "top"; k += 1; continue
''', r'''                if w == "--top":
                    tpos = "top"; k += 1; continue
                if w == "--enrich":
                    enr = True; k += 1; continue
                if w in ("--noenrich", "--no-enrich"):
                    enr = False; k += 1; continue
''')
rep(r'''            text = " ".join(prompt)
''', r'''            text = " ".join(prompt)
            if enr:
                text, added = enrich_prompt(text)
                if added:
                    print("  " + D + "prompt sent: " + text + O)
''')

# --- make
rep(r'''        t = time.perf_counter(); cap = cap_feats(a.prompt, tmp, a.verbose)
''', r'''        if a.enrich:
            a.prompt, added = enrich_prompt(a.prompt)
            if added:
                print("  " + C_DIM + "prompt sent: " + a.prompt + C_OFF)
        t = time.perf_counter(); cap = cap_feats(a.prompt, tmp, a.verbose)
''')
rep(r'''    ap.add_argument("--no-upscale", dest="upscale", action="store_false", help="keep the 512 image as it is")
''', r'''    ap.add_argument("--no-upscale", dest="upscale", action="store_false", help="keep the 512 image as it is")
    ap.add_argument("--enrich", dest="enrich", action="store_true", default=bool(DEFAULTS.get("enrich", False)),
                    help="add material words to the prompt (the old ZImage table, cues.json)")
    ap.add_argument("--no-enrich", dest="enrich", action="store_false", help=argparse.SUPPRESS)
''')
rep(r'''PATHS, DEFAULTS, SERVER, CONFIG_FILE = load_config()
''', r'''PATHS, DEFAULTS, SERVER, CONFIG_FILE = load_config()
from enrich import enrich as enrich_prompt          # gamla ZImages materialordstabell (cues.json)
''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

# --- ps1
p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"


def rp(old, new):
    global p
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    if p.count(old) != 1:
        raise SystemExit("ps1: ankare %d ggr: %r" % (p.count(old), old[:70]))
    p = p.replace(old, new)


rp("""    [Alias('NoUp')] [switch] $NoUpscale,""", """    [Alias('NoUp')] [switch] $NoUpscale,
    [switch] $Enrich,""")
rp("""if ($Upscale) { $common += '--upscale' } else { $common += '--no-upscale' }""",
   """if ($Upscale) { $common += '--upscale' } else { $common += '--no-upscale' }
if ($Enrich)  { $common += '--enrich' }""")
rp("""        @('-Seed <n>',          'another number = another image for the same text',  "now $Seed"),""",
   """        @('-Seed <n>',          'another number = another image for the same text',  "now $Seed"),
        @('-Enrich',            'add material words (skin pores, wet sand ...)',     $(if ($D.enrich) { 'now on' } else { 'now off' })),""")
io.open(PS, "w", encoding="ascii", newline="").write(p)

j = json.load(open(J, encoding="utf-8"))
nd = {}
for k, v in j["defaults"].items():
    nd[k] = v
    if k == "upscale_to":
        nd["enrich"] = False
        nd["unload_after_s"] = 60
j["defaults"] = nd
io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
print("OK")
