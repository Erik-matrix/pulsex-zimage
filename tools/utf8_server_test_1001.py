# utf8_server_test_1001.py — 2026-10-01: the CPU text encoder (llama-server, started by zimage.ps1 -EncoderOn cpu)
# with paths outside ASCII. Each case stops any running server first, so the case starts its own with its own paths.
#   python utf8_server_test_1001.py ref|model|temp [tag]
#   ref   : ordinary paths (the reference image for this encoder - the CPU encoder differs slightly from the NPU one)
#   model : the text encoder GGUF in C:\PulseCore\models\zImage_srv_åäö (a hard link)
#   temp  : TEMP (the server's log, the work files) under a folder with Ł
# 512, no upscale, seed 1234, PNG to D:\prov\2026-10-01_zimage_utf8_server\.
import json, os, subprocess, sys
from pathlib import Path

ZD = Path(r"C:\PulseCore\PulseX\zimage_dit")
OUT = Path(r"D:\prov\2026-10-01_zimage_utf8_server")
OUT.mkdir(parents=True, exist_ok=True)
case = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
c = json.loads((ZD / "zimage.json").read_text(encoding="utf-8"))
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
if case == "model":
    c["paths"]["encoder"] = "C:/PulseCore/models/zImage_srv_åäö/qwen3-4b-zimage-q4_0.gguf"
elif case == "temp":
    t = OUT / "Temp_Łódź"; t.mkdir(exist_ok=True)
    env["TEMP"] = env["TMP"] = str(t)
elif case != "ref":
    sys.exit("case ref | model | temp")
cfg = OUT / ("cfg_%s.json" % case)
cfg.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
env["ZIMAGE_CONFIG"] = str(cfg)
ps = [r"C:\Program Files\PowerShell\7\pwsh.exe", "-NoProfile", "-File", str(ZD / "zimage.ps1")]
subprocess.run(ps + ["stop"], env=env, capture_output=True, timeout=120)
png = OUT / ("%s_%s.png" % (case, tag))
p = subprocess.run(ps + ["make", "a close up picture of a cat in jungle", "-EncoderOn", "cpu", "-NoUp", "-Out", str(png)],
                   cwd=str(OUT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
log = p.stdout + p.stderr
(OUT / ("%s_%s.log" % (case, tag))).write_text(log, encoding="utf-8")
subprocess.run(ps + ["stop"], env=env, capture_output=True, timeout=120)
print("case %s: rc=%d, png %s" % (case, p.returncode, "written" if png.exists() else "MISSING"))
keys = ("error", "fail", "never came", "cannot", "encode the prompt", "saved", "traceback")
print("\n".join(l for l in log.split("\n") if any(k in l.lower() for k in keys))[-1200:])
