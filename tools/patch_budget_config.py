# 09-28: packcachen + viktbudgeten in i zimage.py/zimage.json.
#   pack_cache (true): DiT-vikterna sparas en gang i NPU:ns tegelpackade layout bredvid GGUF:en
#                      (<dit>.hexpack, 3,46 GB) och kopieras sedan in i stallet for att packas om.
#   npu_weight_budget_mb (0): tak for DiT-vikterna pa NPU:n; 0 = allt strommas genom tva platser
#                      (194 MB), -1 = allt residentt (som forut).
# Matt 09-28, bit-identiska latenter: 512 9,7 s/+3,75 GB -> 8,8 s/+0,64 GB; 1024 36,4/+4,6 -> 35,6/+1,9.
#   python tools/patch_budget_config.py
import io, json

PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
JS = [r"C:\PulseCore\PulseX\zimage_dit\zimage.json", r"C:\PulseCore\PulseX\zimage_dit\zimage.example.json"]
s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new, cnt=1):
    global s
    if s.count(old) != cnt:
        raise SystemExit("py: ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


rep('''             "upscale_detail": 0.5}''', '''             "upscale_detail": 0.5, "pack_cache": True, "npu_weight_budget_mb": 0}''')

# miljo for DiT-motorn: bada startvagarna (engangs och server)
rep('''def start_dit(workdir, steps, verbose):''', '''def dit_weight_env():
    """ZI_PACKCACHE / ZI_WEIGHT_BUDGET_MB for zimage-dit-stream.exe (see zimage.json)."""
    env = {}
    if DEFAULTS.get("pack_cache", True):
        pack = DIT.with_suffix(".hexpack")
        env["ZI_PACKCACHE"] = str(pack)
        if not pack.exists():
            print("  " + C_DIM + "(once: preparing the NPU weight cache next to the model, ~5 s, 3.5 GB)" + C_OFF,
                  flush=True)
    b = int(DEFAULTS.get("npu_weight_budget_mb", 0))
    if b >= 0:
        env["ZI_WEIGHT_BUDGET_MB"] = str(b)
    return env


def start_dit(workdir, steps, verbose):''')
rep('''           "GGML_HEXAGON_PD_DUMP": os.environ.get("GGML_HEXAGON_PD_DUMP", "1"), "ZI_STEPS": str(steps)}
    p = subprocess.Popen([str(BIN / "zimage-dit-stream.exe"), str(DIT), str(workdir), "HTP0"],''',
    '''           "GGML_HEXAGON_PD_DUMP": os.environ.get("GGML_HEXAGON_PD_DUMP", "1"), "ZI_STEPS": str(steps),
           **dit_weight_env()}
    p = subprocess.Popen([str(BIN / "zimage-dit-stream.exe"), str(DIT), str(workdir), "HTP0"],''')
rep('''           "GGML_HEXAGON_PD_DUMP": os.environ.get("GGML_HEXAGON_PD_DUMP", "1"), "ZI_STEPS": str(steps), "ZI_SERVE": "1"}''',
    '''           "GGML_HEXAGON_PD_DUMP": os.environ.get("GGML_HEXAGON_PD_DUMP", "1"), "ZI_STEPS": str(steps), "ZI_SERVE": "1",
           **dit_weight_env()}''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

for J in JS:
    j = json.load(open(J, encoding="utf-8"))
    j["defaults"]["pack_cache"] = True
    j["defaults"]["npu_weight_budget_mb"] = 0
    if "_help" in j:
        j["_help"]["defaults.size"] = ("512 (default, with the x4 upscaler -> 1024), 1024, or a wide size: '480p' "
                                       "(= 848 x 480) or 'WxH' with both sides divisible by 16")
        j["_help"]["defaults.pack_cache"] = ("true: the image model's weights are stored once in the NPU's own tiled "
                                             "layout next to the model (.hexpack, 3.5 GB) and then just copied - "
                                             "no re-packing on every image")
        j["_help"]["defaults.npu_weight_budget_mb"] = ("NPU memory for the image model's weights. 0 (default): they "
                                                       "stream through two small slots (~0.2 GB) from the file cache "
                                                       "- ~3 GB less committed memory, same image, same speed. "
                                                       "-1: keep all 3.3 GB on the NPU")
    io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
print("OK")
