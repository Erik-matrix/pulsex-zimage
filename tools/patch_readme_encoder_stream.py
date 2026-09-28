# 09-28: README - encodern strommar sina lager (zimage_encode_stream.h): ~0,27 GB i stallet for 2,3.
#   python tools/patch_readme_encoder_stream.py
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


rep("| Text encoder on the NPU (default) | 2.3 GB — only for the ~2 s it is loaded; it starts while you type and exits after encoding |",
    "| Text encoder on the NPU (default) | 0.27 GB — its layers stream through two slots; ~1 s per prompt, then it exits |")
rep("""(~1.3 s, hidden behind your typing), encodes when you press Enter (~0.8 s), frees its memory and
exits cleanly. Only its transformer layers go to the NPU: the output layer (the same 304 MB table as
the input embedding; a text encoder never needs logits) stays a mapped view of the model file, and
its buffers are sized to the prompt, not to 512 tokens. Between images the image model keeps only
its NPU session and gives back its slots and graph buffers, so the encoder does not sit on top of
it.""", """(~0.35 s), encodes when you press Enter (~0.6 s) and exits (~0.1 s). Like the image model, its
layers stream: llama.cpp only tokenizes, and the 35 layers the image model needs run one at a time
through two ~54 MB slots, copied from `qwen3-4b-zimage-q4_0.hexpack` (the layers in the NPU's
tiled layout, written next to the model the first time, 2 GB). Layer 36 and the output head are
never loaded — the image model reads the input of the last layer. The encodings match the
llama.cpp path (`ZI_ENC_STREAM=0`, the whole model on the NPU, 2.3 GB) to within the NPU's own
noise. Between images the image model keeps only its NPU session and gives back its slots and
graph buffers.""")
rep("""it was before PulseX started: **+0.3 GB between images, +2.6 GB at the peak** (while the encoder
lives), back to zero after quitting.""", """it was before PulseX started: **+0.3 GB between images, +1.0 GB at the peak**, back to zero after
quitting. (With the whole encoder on the NPU the peak was +2.6 GB, and with the image model's
weights resident as well, +6.8 GB.)""")
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("OK")
