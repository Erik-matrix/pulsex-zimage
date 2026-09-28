# GitHub-redo (09-28 "gor klart json-konverteringen till Github"): zimage.json ar ENDA kallan
# for sokvagar. Koden bar inga maskinspecifika sokvagar langre; saknas zimage.json pekar felet pa
# zimage.example.json. Inbyggda forval foljer nu den aktuella konfigurationen (shift 1, cpu-encoder,
# upscale, enrich av, unload 60 s) - de gamla hade drivit (upscale False, shift saknades).
import io
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
s = io.open(PY, encoding="utf-8", newline="").read()
a = s.index("# --- sokvagar och forval: zimage.json")
b = s.index("PATHS, DEFAULTS, SERVER, CONFIG_FILE = load_config()")
new = r'''# --- sokvagar och forval: zimage.json (ZIMAGE_CONFIG pekar om) ---------------
# zimage.json ar ENDA kallan for sokvagar (kopiera zimage.example.json). Relativa sokvagar tolkas
# mot json-filens katalog, {models} = paths.models. Forvalen nedan galler for nycklar filen saknar.
HERE = Path(__file__).resolve().parent
_DEFAULTS = {"out": "zimage.jpg", "size": 512, "steps": 4, "shift": 1.0, "seed": 1234, "quality": 95,
             "vae": "taef1", "encoder": "cpu", "upscale": True, "upscale_to": 1024, "enrich": False,
             "unload_after_s": 60}
_PATH_KEYS = ("models", "dit", "encoder", "tokenizer", "vae_full", "taef1", "bin", "esrgan_x4", "qnn_runtime")


def load_config():
    p = Path(os.environ.get("ZIMAGE_CONFIG", str(HERE / "zimage.json")))
    if not p.exists():
        sys.exit("PulseX: no config file at %s - copy zimage.example.json to zimage.json and set the paths."
                 % p)
    user = json.loads(p.read_text(encoding="utf-8"))
    missing = [k for k in _PATH_KEYS if k not in user.get("paths", {})]
    if missing:
        sys.exit("PulseX: %s lacks paths.%s (see zimage.example.json)" % (p, ", paths.".join(missing)))
    defaults = dict(_DEFAULTS)
    defaults.update(user.get("defaults", {}))
    server = {"port": 8099}
    server.update(user.get("server", {}))
    base = p.parent

    def res(v):
        q = Path(v)
        return q if q.is_absolute() else (base / q).resolve()

    models = res(user["paths"]["models"])
    paths = {k: (models if k == "models" else res(str(v).replace("{models}", str(models))))
             for k, v in user["paths"].items()}
    return paths, defaults, server, p


'''
s = s[:a] + new + s[b:]
io.open(PY, "w", encoding="utf-8", newline="").write(s)

p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"
old = """if (-not (Test-Path $ConfigFile)) { Write-Host "PulseX: config file not found: $ConfigFile" -ForegroundColor Red; exit 1 }"""
new = """if (-not (Test-Path $ConfigFile)) {
    Write-Host "PulseX: config file not found: $ConfigFile" -ForegroundColor Red
    Write-Host "  copy zimage.example.json to zimage.json next to zimage.ps1 and set the paths for this machine." -ForegroundColor Yellow
    exit 1
}""".replace("\n", nl)
assert p.count(old) == 1
p = p.replace(old, new)
io.open(PS, "w", encoding="ascii", newline="").write(p)
print("OK")
