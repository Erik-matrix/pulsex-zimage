# Encodern pa CPU som forval (09-27). Matt: NPU-encodern haller +2 813 MB COMMITTED
# (rpcmem) sa lange den lever - i repl hela sessionen, ovanpa DiT-servern (+4,9 GB) = ~17,8 GB.
# CPU utan repack laser vikterna direkt ur den minnesmappade filen (filcache, ej commit):
# +554 MB. Kallstart 1,6 s mot 2,5-3,0; embedding 17 tok 0,39 mot 0,23 s, 71 tok 1,5 mot 0,3 s.
# cos 0,99999 mot NPU. Standardtradar (8) ar bast - 12 overbokar och blir 12x langsammare.
# ⛔ GPU (OpenCL) hjalper INTE: kopierar vikterna till egna buffertar (SOA-layout) = commit.
import io, json
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
J = r"C:\PulseCore\PulseX\zimage_dit\zimage.json"

p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"
def rp(old, new):
    global p
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    n = p.count(old)
    if n != 1: raise SystemExit("ps1: ankare %d ggr: %r" % (n, old[:70]))
    p = p.replace(old, new)

rp("""    [ValidateSet('', 'taef1', 'full')] [string] $Vae = '',""",
   """    [ValidateSet('', 'taef1', 'full')] [string] $Vae = '',
    [ValidateSet('', 'cpu', 'npu')] [string] $EncoderOn = '',""")
rp("""if (-not $Vae)       { $Vae       = [string] $D.vae }""",
   """if (-not $Vae)       { $Vae       = [string] $D.vae }
if (-not $EncoderOn) { $EncoderOn = if ($D.encoder) { [string] $D.encoder } else { 'cpu' } }""")
rp("""    W '  loading the text encoder ' DarkGray -NoNL""",
   """    W ("  loading the text encoder ({0}) " -f $EncoderOn.ToUpper()) DarkGray -NoNL""")
rp("""            '-ngl', '99', '-c', '512', '--host', '127.0.0.1', '--port', $Port,
            # ONLY the NPU. With a GPU backend present, -ngl 99 split the layers NPU/GPU:
            # cap_feats relL2 0.155 vs the HF reference (NPU alone 0.047), ready 5.9 s vs 3.2 s.
            '--device', 'HTP0', '--no-warmup',""",
   """            '-c', '512', '--host', '127.0.0.1', '--port', $Port, '--no-warmup') + $(
            if ($EncoderOn -eq 'npu') {
                # ONLY the NPU. With a GPU backend present, -ngl 99 split the layers NPU/GPU:
                # cap_feats relL2 0.155 vs the HF reference (NPU alone 0.047), ready 5.9 s vs 3.2 s.
                # Its weights are copied into NPU memory: +2.8 GB COMMITTED while it runs.
                @('-ngl', '99', '--device', 'HTP0')
            } else {
                # CPU without repacking reads the weights straight from the mapped file (the
                # page cache), so it commits ~0.5 GB instead of 2.8. Ready 1.6 s; a long prompt
                # costs ~1.2 s more than on the NPU. Default threads (8) are best.
                @('-ngl', '0', '--device', 'none', '--no-repack')
            }) + @(""")
rp("""        @('-Vae taef1|full',    'decoder: taef1 is fast, full is the reference',     "now $Vae"),""",
   """        @('-Vae taef1|full',    'decoder: taef1 is fast, full is the reference',     "now $Vae"),
        @('-EncoderOn cpu|npu', 'text encoder: cpu uses ~2.3 GB less memory, npu is ~1 s faster', "now $EncoderOn"),""")
rp("""    W ("  size {0} | upscale {1} -> {2} | steps {3} | seed {4} | quality {5} | decoder {6} | port {7}" -f `
            $D.size, $(if ($D.upscale) { 'on' } else { 'off' }), $D.upscale_to, $D.steps, $D.seed, $D.quality, $D.vae, $Port)""",
   """    W ("  size {0} | upscale {1} -> {2} | steps {3} | seed {4} | quality {5} | decoder {6} | encoder {7} | port {8}" -f `
            $D.size, $(if ($D.upscale) { 'on' } else { 'off' }), $D.upscale_to, $D.steps, $D.seed, $D.quality, $D.vae, $EncoderOn, $Port)""")
rp("""function Invoke-Engine([string[]] $engineArgs) {
    $env:ZI_VAE = $Vae
""", """function Invoke-Engine([string[]] $engineArgs) {
    $env:ZI_VAE = $Vae
    $env:ZI_ENCODER_ON = $EncoderOn       # also for the no-server fallback in zimage.py
""")
io.open(PS, "w", encoding="ascii", newline="").write(p)

s = io.open(PY, encoding="utf-8", newline="").read()
old = '''           "-ngl", "99", "-c", "512"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)'''
assert s.count(old) == 1
s = s.replace(old, '''           "-c", "512"]
    # Samma val som servern (zimage.ps1 -EncoderOn): cpu laser vikterna ur filcachen (+0,5 GB
    # commit), npu kopierar dem till rpcmem (+2,8 GB).
    if os.environ.get("ZI_ENCODER_ON", "cpu") == "npu":
        cmd += ["-ngl", "99", "--device", "HTP0"]
    else:
        cmd += ["-ngl", "0", "--device", "none", "--no-repack"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

j = json.load(open(J, encoding="utf-8"))
d = j["defaults"]
nd = {}
for k, v in d.items():
    nd[k] = v
    if k == "vae": nd["encoder"] = "cpu"
j["defaults"] = nd
io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
print("OK")
