# 09-28: README Memory - encodern 2,3 GB, DiT-servern slapper sina buffertar mellan bilderna.
#   python tools/patch_readme_encoder_slim.py
import io

P = r"C:\PulseCore\PulseX\zimage_dit\README.md"
s = io.open(P, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"


def rep(old, new):
    global s
    old, new = old.replace("\n", nl), new.replace("\n", nl)
    if s.count(old) != 1:
        raise SystemExit("ankare %d ggr: %r" % (s.count(old), old[:70]))
    s = s.replace(old, new)


rep("| its graph buffers (512 / 480p / 1024) | 0.15 / 0.24 / 0.6 GB |",
    "| its graph buffers (512 / 480p / 1024) | 0.15 / 0.24 / 0.6 GB — only while an image is made; the REPL frees them between images |")
rep("| Text encoder on the NPU (default) | 2.8 GB — only for the ~2 s it is loaded; it starts while you type and exits after encoding |",
    "| Text encoder on the NPU (default) | 2.3 GB — only for the ~2 s it is loaded; it starts while you type and exits after encoding |")
rep("""(~1.4 s, hidden behind your typing), encodes when you press Enter (~0.9 s), frees its memory and
exits cleanly. It is never kept next to the image model. The image model itself is unloaded after
an idle minute (the next image then takes a few seconds longer).""",
    """(~1.3 s, hidden behind your typing), encodes when you press Enter (~0.8 s), frees its memory and
exits cleanly. Only its transformer layers go to the NPU: the output layer (the same 304 MB table as
the input embedding; a text encoder never needs logits) stays a mapped view of the model file, and
its buffers are sized to the prompt, not to 512 tokens. Between images the image model keeps only
its NPU session and gives back its slots and graph buffers, so the encoder does not sit on top of
it. The image model itself is unloaded after an idle minute (the next image then takes a few
seconds longer).

Measured in the REPL, two 848 × 480 images with a rest after each, committed memory above what
it was before PulseX started: **+0.3 GB between images, +2.6 GB at the peak** (while the encoder
lives), back to zero after quitting.""")
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
