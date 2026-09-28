# Hamtar gamla ZImages promptberikning (apps/zimage/zimage.cpp: CUES, CUES_FRAM, PHOTO_WORDS,
# PHOTO_ADD) till cues.json, sa att tabellen inte skrivs av for hand och kan redigeras utan kod.
# Gamla appen lade till MATERIALORD: modellen ar destillerad till CFG=1, "photorealistic" biter
# inte, men "skin pores, catchlight in the eyes" gor det (cfg1-needs-material-words).
import io, json, re
SRC = r"C:\PulseCore\apps\zimage\zimage.cpp"
OUT = r"C:\PulseCore\PulseX\zimage_dit\cues.json"
s = io.open(SRC, encoding="utf-8", errors="replace").read()

NL = chr(10)
WS = "[ " + chr(9) + chr(13) + NL + "]*"
STR = '"([^"]*)"'

def table(name):
    i = s.index("static const PCue %s[] = {" % name)
    j = s.index(NL + "};", i)
    body = re.sub("//[^" + NL + "]*", "", s[i:j])
    out = []
    for m in re.finditer("[{]" + WS + STR + WS + "," + WS + "((" + STR + WS + ")+)[}]", body):
        out.append([m.group(1), "".join(re.findall(STR, m.group(2)))])
    return out

cues, fram = table("CUES"), table("CUES_FRAM")
pw = s[s.index("PHOTO_WORDS[]"):]; pw = pw[:pw.index("}")]
photo_words = re.findall(r'"([^"]*)"', pw)
photo_add = re.findall(STR, s[s.index("PHOTO_ADD ="):])[0]
n_src = len(re.findall(r'^\s*\{"', s[s.index("static const PCue CUES[]"):s.index("\n};", s.index("static const PCue CUES[]"))], re.M))
assert len(cues) == n_src, (len(cues), n_src)
doc = {
    "_about": "Prompt enrichment ported from the old ZImage (apps/zimage/zimage.cpp). Z-Image-Turbo is distilled to CFG=1: words like 'photorealistic' do nothing, MATERIAL words do. When a key appears in the prompt as a word, its fragments are appended (most specific key first) while they fit the token budget. 'front' fragments name things that must EXIST; plain ones describe how things look.",
    "budget_tokens": 22,
    "front": fram, "cues": cues,
    "photo_words": photo_words, "photo_add": photo_add,
}
io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
print("cues", len(cues), "front", len(fram), "photo", photo_words)
