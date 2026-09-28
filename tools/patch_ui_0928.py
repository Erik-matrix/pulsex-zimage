# 09-28: (1) "--1024" gav "unknown option" och en 512-bild -> --512/--1024 som kortformer,
# (2) "like the old ZImage" far inte sta kvar i UI/README - neutral text overallt anvandaren ser.
import io, json
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"
EX = r"C:\PulseCore\PulseX\zimage_dit\zimage.example.json"
EN = r"C:\PulseCore\PulseX\zimage_dit\enrich.py"

s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new):
    global s
    if s.count(old) != 1:
        raise SystemExit("py: ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


rep('''                if w == "--size" and k + 1 < len(words) and words[k + 1] in ("512", "1024"):''',
    '''                if w in ("--512", "--1024"):                   # kortform (09-28: "--1024" foll igenom)
                    px = int(w[2:]); k += 1; continue
                if w == "--size" and k + 1 < len(words) and words[k + 1] in ("512", "1024"):''')
rep('''    print("    " + Y + "--size 512|1024" + O + "   resolution (now " + str(a.size) + ")")''',
    '''    print("    " + Y + "--size 512|1024" + O + "   resolution, or just " + Y + "--512" + O + " / " + Y + "--1024"
          + O + " (now " + str(a.size) + ")")''')
rep('''" add material words, like the old ZImage (now "''', '''" add material words (skin pores, wet sand ...) (now "''')
rep('''help="at 512: Real-ESRGAN x4 afterwards (the old ZImage recipe)")''', '''help="at 512: Real-ESRGAN x4 afterwards")''')
rep('''help="add material words to the prompt (the old ZImage table, cues.json)")''',
    '''help="add material words to the prompt (cues.json)")''')
rep('''    Never overwrites: the old ZImage did not, and a new session used to start again at _0001."""''',
    '''    Never overwrites an earlier image, also not one from an earlier session."""''')
rep('''    """Real-ESRGAN x4 as a QNN context via ONNX Runtime's QNN EP - the old ZImage recipe for''',
    '''    """Real-ESRGAN x4 as a QNN context via ONNX Runtime's QNN EP - the upscaler for''')
rep('''# --- text on the image (the old ZImage poster text) ---------------------------------------------''',
    '''# --- text on the image ------------------------------------------------------------------------''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

p = io.open(PS, encoding="ascii", newline="").read()
old = "'512, then Real-ESRGAN x4 -> 1024 (old ZImage)',   '~19 s'"
assert p.count(old) == 1
p = p.replace(old, "'512, then Real-ESRGAN x4 -> 1024',               '~19 s'")
io.open(PS, "w", encoding="ascii", newline="").write(p)

j = json.load(open(EX, encoding="utf-8"))
j["_help"]["defaults.shift"] = "sigma shift; 1.0 = linear schedule [1, .75, .5, .25], measurably more skin/fur detail than 3.0"
j["_help"]["defaults.enrich"] = "append material words to the prompt (cues.json), e.g. 'skin pores, catchlight in the eyes'"
io.open(EX, "w", encoding="utf-8", newline="\n").write(json.dumps(j, indent=2, ensure_ascii=False) + "\n")

e = io.open(EN, encoding="utf-8", newline="").read()
old = '"""Prompt enrichment, ported 1:1 from the old ZImage (apps/zimage/zimage.cpp enrich()).'
assert e.count(old) == 1
e = e.replace(old, '"""Prompt enrichment: append material words that Z-Image-Turbo responds to.')
io.open(EN, "w", encoding="utf-8", newline="").write(e)
print("OK")
