# 2026-10-02: README.md - the installer. A short section in front of "The window" (the easy way in), one row in the
# file table, and how the setup file is built.
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
p = HERE / "README.md"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)


SECTION = """## Install

The easy way is the setup file from the [latest release](https://github.com/Erik-matrix/pulsex-zimage/releases/latest),
`PulseX-Z-Image-Setup-<version>.exe`: next, next, and the window (see below) is in the Start menu with its picture
model. No Python, no command line.

- It installs for you alone, without administrator rights, and it **downloads the picture model while it installs**
  (5.8 GB from [Hugging Face](https://huggingface.co/Pexqman/Z-Image-Turbo-Q4_0-GGUF), every file checked against its
  SHA-256). Have about 12 GB free. "The program only" skips the download - then put the three model files in the
  program folder's `models\\` yourself.
- The first picture takes longer than the following ones: the model is prepared for the NPU once.
- The setup file is not code-signed, so Windows SmartScreen asks first: *More info* › *Run anyway*.
- The same limit as everywhere in PulseX applies: the NPU module is self-signed (see *Setup*), so the installed
  program makes pictures only on a computer that accepts self-signed NPU modules.
- The enlarging (2×, 4×) is not part of the installation yet: it needs the QNN runtime and the upscaler's compiled
  graph, which the setup does not carry. Without them the window makes pictures at their own size.
- Uninstalling removes the program and the model; your pictures (`Pictures\\PulseX Z-Image`) stay.

The setup is made with [Inno Setup](https://jrsoftware.org/isinfo.php) from `installer/pulsex_zimage.iss`:
`ISCC.exe installer\\pulsex_zimage.iss` in a clone writes it to `installer\\output\\`.

"""
rep("## The window: PulseX Z-Image\n", SECTION + "## The window: PulseX Z-Image\n")
rep("| `window/` |", "| `installer/` | the setup file's script (Inno Setup), the `zimage.json` it installs and the text of its first page |\n| `window/` |")
p.write_text(s, encoding="utf-8", newline="\n")
print("README.md: the Install section and the installer row")
