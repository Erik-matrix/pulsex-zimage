# Shift 1,0 som forval (09-27, punkt 4 "1024 gor inte lika fina bilder som gamla ZImage").
# Gamla motorn korde shift 1,0 = linjart schema [1 .75 .5 .25 0] (dit_full_engine.cpp:3569-3582:
# "3,0 matt samst, hittar pa textur"). Vi korde motorns inbyggda 3,0 = [1 .9 .75 .5 0].
# MATT 09-27, D:\prov\2026-09-27_quality: shift 1 ger mer detalj pa 5/5 motiv (Laplace-varians
# +12..+65 %), samma komposition; 1024 nativ ansikte 230 -> 379, hudporer/fransar synliga.
# zimage.json defaults.shift styr; en satt ZI_SHIFT i miljon vinner (for A/B).
import io, json
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
CA = r"C:\PulseCore\PulseX\zimage_dit\cascade2.py"
J = r"C:\PulseCore\PulseX\zimage_dit\zimage.json"
s = io.open(PY, encoding="utf-8", newline="").read()
old = 'PATHS, DEFAULTS, SERVER, CONFIG_FILE = load_config()\n'
assert s.count(old) == 1
s = s.replace(old, old + '''# Sigma-shift for DiT-motorn (zimage_stream.cpp laser ZI_SHIFT; utan den kor den 3,0).
# 1,0 = gamla ZImages linjara schema - mer hud/pals-detalj, matt 09-27. Miljon vinner.
os.environ.setdefault("ZI_SHIFT", "%g" % float(DEFAULTS.get("shift", 1.0)))
''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

j = json.load(open(J, encoding="utf-8"))
nd = {}
for k, v in j["defaults"].items():
    nd[k] = v
    if k == "steps":
        nd["shift"] = 1.0
j["defaults"] = nd
io.open(J, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")
print("OK")
