# (1) PulseX i UI-text, fonsterrubrik (kommandot/filerna heter fortfarande zimage - 09-27).
# (2) Skriv aldrig over en bild: utan uttryckligt -Out valjer bade make och repl nasta lediga
#     <stem>_NNNN<suffix> (gamla ZImage skrev inte over; nya borjade pa _0001 varje session).
import io, re
PY = r"C:\PulseCore\PulseX\zimage_dit\zimage.py"
PS = r"C:\PulseCore\PulseX\zimage_dit\zimage.ps1"

s = io.open(PY, encoding="utf-8", newline="").read()


def rep(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit("py: ankare %d ggr (vill %d): %r" % (n, count, old[:80]))
    s = s.replace(old, new)


rep('''    return (C_CYAN + "zimage" + C_OFF + "  " + size''', '''    return (C_CYAN + "PulseX" + C_OFF + "  " + size''')
rep('''    print(C_CYAN + "zimage - interactive mode" + O)''', '''    print(C_CYAN + "PulseX - interactive mode" + O)''')
rep('''        sys.exit("zimage: no such image: %s" % src)''', '''        sys.exit("PulseX: no such image: %s" % src)''')
rep('''        sys.exit("zimage: give -Title and/or -Subtitle for the text")''', '''        sys.exit("PulseX: give -Title and/or -Subtitle for the text")''')
rep('''        sys.exit("zimage: missing files (see %s):''', '''        sys.exit("PulseX: missing files (see %s):''')
rep('''    ap = argparse.ArgumentParser(prog="zimage", description="Make an image from a prompt on the NPU.")''',
    '''    ap = argparse.ArgumentParser(prog="PulseX", description="Make an image from a prompt on the NPU.")''')

# nasta lediga nummer
rep('''def describe(px, steps, seed, up, up_to):''', '''def next_free(template):
    """<stem>_NNNN<suffix> with NNNN one higher than the highest that already exists next to it.
    Never overwrites: the old ZImage did not, and a new session used to start again at _0001."""
    t = Path(template)
    rx = re.compile(re.escape(t.stem) + r"_(\\d{4,})" + re.escape(t.suffix) + r"$", re.IGNORECASE)
    folder = t.parent if str(t.parent) not in ("", ".") else Path.cwd()
    hi = 0
    if folder.is_dir():
        for f in folder.iterdir():
            m = rx.match(f.name)
            if m:
                hi = max(hi, int(m.group(1)))
    return t.with_name("%s_%04d%s" % (t.stem, hi + 1, t.suffix))


def describe(px, steps, seed, up, up_to):''')
if "\nimport re" not in s and "import re\n" not in s:
    rep("import argparse, json, os, subprocess, sys, tempfile, time\n",
        "import argparse, json, os, re, subprocess, sys, tempfile, time\n")

rep('''    print("  Up/Down recalls earlier lines.  Images: " + D + os.getcwd() + O + " as "
          + Path(a.out).stem + "_NNNN" + Path(a.out).suffix)''',
    '''    print("  Up/Down recalls earlier lines.  Images: " + D + os.getcwd() + O + " as "
          + Path(a.out).stem + "_NNNN" + Path(a.out).suffix + " (never overwritten)")''')
rep('''            n += 1
            out = Path(a.out).with_name(f"{Path(a.out).stem}_{n:04d}{Path(a.out).suffix}")''',
    '''            n += 1
            out = next_free(a.out)             # nasta lediga - skriver aldrig over en tidigare bild''')
rep('''        out = save_image(im, a.out, a.quality, a.title, a.subtitle, a.text_pos)
        print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)  in %.1f s" % (out, im.shape[1], im.shape[0],''',
    '''        # Utan -Out: nasta lediga zimage_NNNN.jpg. Med -Out: exakt den filen (uttryckligt val).
        out = save_image(im, a.out if a.out_given else next_free(a.out), a.quality, a.title, a.subtitle, a.text_pos)
        print("  " + C_GREEN + "saved" + C_OFF + " %s  (%d x %d)  in %.1f s" % (out, im.shape[1], im.shape[0],''')
io.open(PY, "w", encoding="utf-8", newline="").write(s)

p = io.open(PS, encoding="ascii", newline="").read()
nl = "\r\n" if "\r\n" in p else "\n"


def rp(old, new, count=1):
    global p
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    n = p.count(old)
    if n != count:
        raise SystemExit("ps1: ankare %d ggr (vill %d): %r" % (n, count, old[:80]))
    p = p.replace(old, new)


rp("""$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot""", """$ErrorActionPreference = 'Stop'
$Root    = $PSScriptRoot
# The window says PulseX; the command stays "zimage" (09-27).
try { $Host.UI.RawUI.WindowTitle = 'PulseX' } catch {}""")
rp("""if (-not (Test-Path $ConfigFile)) { Write-Host "zimage: config file not found: $ConfigFile" -ForegroundColor Red; exit 1 }""",
   """if (-not (Test-Path $ConfigFile)) { Write-Host "PulseX: config file not found: $ConfigFile" -ForegroundColor Red; exit 1 }""")
rp("""    W 'zimage' White -NoNL; W " $Version - turns a sentence into a picture, on this PC's NPU.\"""",
   """    W 'PulseX' White -NoNL; W " $Version - turns a sentence into a picture, on this PC's NPU.\"""")
rp("""    W '    -> zimage.jpg in the current folder. Change the words, run it again.' DarkGray""",
   """    W '    -> zimage_0001.jpg in the current folder, then _0002 ... - nothing is overwritten.' DarkGray""")
rp("""    'version' { W "zimage $Version" White -NoNL;""", """    'version' { W "PulseX $Version" White -NoNL;""")
rp("""        @('-Out <file>',        'file name, .jpg or .png',                           "now $Out"),""",
   """        @('-Out <file>',        'exact file name (.jpg/.png); without it: next free', "zimage_NNNN.jpg"),""")
# -o bara om anvandaren gav -Out, annars valjer motorn nasta lediga nummer
rp("""$common = @('--size', $Size, '--steps', $Steps, '--seed', $Seed, '-q', $Quality, '-o', $Out,
            '--upscale-to', $UpscaleTo)""",
   """$common = @('--size', $Size, '--steps', $Steps, '--seed', $Seed, '-q', $Quality, '--upscale-to', $UpscaleTo)
if ($PSBoundParameters.ContainsKey('Out')) { $common += @('-o', $Out) }""")
io.open(PS, "w", encoding="ascii", newline="").write(p)
print("OK")
