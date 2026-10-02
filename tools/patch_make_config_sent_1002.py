# 2026-10-02: two things the window showed.
#   * "sent:" is printed when the ENRICHMENT changed the prompt, and without the LoRA's trigger word - the window shows
#     the line to the user ("Sent: ..."), and "l3n0v0, " in front of every prompt is not something the user wrote or
#     needs to see. (Before: every picture said "sent: l3n0v0, ...".)
#   * zimage.json is looked for beside the program and then one folder up - the published layout has the programs in
#     bin\ and zimage.json above it. zimage-make and the window (flux_gui.cpp, Z-Image build) do the same.
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def patch(rel, pairs):
    p = ROOT / rel
    s = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert s.count(old) == 1, (rel, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", rel)


patch("zimage_dit/zimage_make.cpp", [
    ("    if (full != prompt) { printf(\"sent: %s\\n\", full.c_str()); fflush(stdout); }\n",
     "    if (sent != prompt) { printf(\"sent: %s\\n\", sent.c_str()); fflush(stdout); }      // what the cue table made of it (the trigger word is not shown)\n"),
    ("    if (const char * e = getenv(\"ZIMAGE_CONFIG\")) if (*e) cfg_path = e;\n",
     "    if (!exists(cfg_path) && exists(exe_dir() + \"\\\\..\\\\zimage.json\")) cfg_path = exe_dir() + \"\\\\..\\\\zimage.json\";      // bin\\ under the folder with zimage.json\n"
     "    if (const char * e = getenv(\"ZIMAGE_CONFIG\")) if (*e) cfg_path = e;\n"),
])
patch("flux_dit/flux_gui.cpp", [
    ("        cfg = e && *e ? e : bin + \"\\\\\" CFG_FILE;\n",
     "        cfg = e && *e ? e : bin + \"\\\\\" CFG_FILE;\n"
     "        if (!(e && *e) && !file_exists(cfg) && file_exists(bin + \"\\\\..\\\\\" CFG_FILE)) cfg = bin + \"\\\\..\\\\\" CFG_FILE;      // bin\\ under the folder with the settings file\n"),
])
