# utf8_path_test_1001.py — 2026-10-01: does the Z-Image chain work when a path has non-ASCII letters?
# (the CCTV folder-picker lesson: a path handed around as ANSI in one place and UTF-8 in another breaks.)
#   python utf8_path_test_1001.py <case> [tag]
#   case model : the DiT GGUF in C:\PulseCore\models\zImage_test_åäö (a hard link), pack cache off (no 3.5 GB write)
#   case temp  : ASCII models, but TEMP (the work folder for latents/decoder files) under a folder with åäö
#   case temp_pl : TEMP with a letter outside the Windows-1252 code page (Ł) - a user name in Polish, Czech, ...
# Everything else as the user's zimage.json; 512, no upscale, seed 1234; PNG to D:\prov\2026-10-01_zimage_utf8\.
import json, os, subprocess, sys
from pathlib import Path

ZD = Path(r"C:\PulseCore\PulseX\zimage_dit")
OUT = Path(r"D:\prov\2026-10-01_zimage_utf8")
OUT.mkdir(parents=True, exist_ok=True)
case = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else "run"
c = json.loads((ZD / "zimage.json").read_text(encoding="utf-8"))
env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
if case == "model":
    c["paths"]["dit"] = "C:/PulseCore/models/zImage_test_åäö/z-image-turbo-ultrareal06-q4_0.gguf"
    c["defaults"]["pack_cache"] = False
elif case in ("temp", "temp_pl"):
    t = OUT / ("Temp_åäö" if case == "temp" else "Temp_Łódź")
    t.mkdir(exist_ok=True)
    env["TEMP"] = env["TMP"] = str(t)
else:
    sys.exit("case model | temp | temp_pl")
cfg = OUT / ("cfg_%s.json" % case)
cfg.write_text(json.dumps(c, indent=2, ensure_ascii=False), encoding="utf-8")
env["ZIMAGE_CONFIG"] = str(cfg)
png = OUT / ("%s_%s.png" % (case, tag))
p = subprocess.run([sys.executable, str(ZD / "zimage.py"), "a close up picture of a cat in jungle", "--size", "512",
                    "--no-upscale", "--seed", "1234", "-o", str(png), "-v"], cwd=str(ZD), env=env,
                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
log = p.stdout + p.stderr
(OUT / ("%s_%s.log" % (case, tag))).write_text(log, encoding="utf-8")
print("case %s: rc=%d, png %s" % (case, p.returncode, "written" if png.exists() else "MISSING"))
print("\n".join(l for l in log.split("\n") if any(k in l.lower() for k in ("error", "fail", "cannot", "kan inte", "gguf", "rc=", "saved", "traceback", "unable")))[-2500:])
