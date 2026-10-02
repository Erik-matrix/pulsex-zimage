# 2026-10-02: README.md gets the window - a section after the examples, the word list's link in the configuration
# table, and the new files in the file table. The screenshot is docs/window.png (a run of the window's own test hook).
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
p = HERE / "README.md"
s = p.read_text(encoding="utf-8")


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)


SECTION = """## The window: PulseX Z-Image

`bin\\pulsex-zimage.exe` is the same thing as a window, for people who do not want a command line: describe the
picture, choose a shape, press **Create picture**. It needs no Python - it runs `bin\\zimage-make.exe`, the whole
chain (text encoder, diffusion transformer, decoder, upscaler, text on the picture) as one C++ program, and reads the
same `zimage.json` (beside the program, or one folder up as in this repository).

![PulseX Z-Image](docs/window.png)

- **Shapes**: square 512 × 512 (about 11 s), portrait 512 × 768 and landscape 768 × 512 (14 s), wide 912 × 512
  (17 s), large square 1024 × 1024 (40 s). The finished picture can be enlarged 2× or 4× (under a second more).
- **Text on the picture**: a title and a smaller line, written on the finished picture - spelled as typed. *Add the
  text* writes it on a picture that already exists; *Enlarge 2×* enlarges one.
- **Swedish words** ("renarna vandrar över höstfjället", "en älg i en snöig granskog") get their English word added,
  from the word list `cues.tsv` - the table form of `cues.json`, named in `zimage.json` as `paths.cues`. The check box
  is under *More settings*, with the variation number, the number of steps and PNG instead of JPEG.
- Pictures are saved in `Pictures\\PulseX Z-Image`, never overwritten; every picture remembers its description.
- English, with Swedish behind the button in the title bar.

Two things to know. The same variation number gives **another picture** than `zimage make -Seed` (the window draws
its start noise from another generator); with the same noise the two chains make the same pixels. And enlarging needs
the upscaler's compiled context beside the QuickSRNet model (`…_ctx_qnn.bin`) - `zimage make` creates it the first
time it upscales.

On the command line the same program is `bin\\zimage-make.exe "a red fox in the snow" --up 1024 --out fox.jpg`
(`--size WxH`, `--seed`, `--steps`, `--enrich nouns|full`, `--title`, `--sub`, `--top`).

"""
rep("## Setup\n", SECTION + "## Setup\n")
rep("| `enrich.py`, `cues.json` | prompt enrichment |\n",
    "| `enrich.py`, `cues.json` | prompt enrichment |\n"
    "| `cues.tsv` | the same word list as a table, for `zimage-make.exe` and the window (`paths.cues`); made from `cues.json` |\n"
    "| `zimage_make.cpp` | `zimage-make.exe`, the whole chain without Python - what the window runs |\n"
    "| `window/` | the window (`flux_gui.cpp`, built with `PULSEX_ZIMAGE`) and the headers it shares with PulseX Image; `bin\\pulsex-zimage.exe` |\n")
p.write_text(s, encoding="utf-8", newline="\n")
(HERE / "docs").mkdir(exist_ok=True)
shutil.copyfile(r"D:\prov\2026-10-02_zimage_gui\gui\run_512x768_saved_up1024_0ref_text_shot.png", HERE / "docs" / "window.png")
print("README.md: the window section, the file table; docs/window.png")
